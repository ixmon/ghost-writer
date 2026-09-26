#!/usr/bin/env python3
"""GhostWriter Web — FastAPI server for the GhostWriter book generation pipeline."""

import bootstrap

if __name__ == "__main__":
    bootstrap.ensure_runtime(("fastapi", "httpx", "uvicorn", "yaml"))

import argparse
import os
import re
import sys
import sysconfig
import glob
import yaml
import asyncio
import subprocess
import signal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from settings import load_settings, redact_url

# ─── Custom YAML Dumper that forces double quotes on risky strings ──────
class QuotedDumper(yaml.SafeDumper):
    pass

def _str_representer(dumper, data):
    # Force double quotes for strings containing ': ' (common in long key_events)
    # or newlines, to prevent YAML parsing surprises.
    if '\n' in data or re.search(r':\s', data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

QuotedDumper.add_representer(str, _str_representer)

# ─── Configuration ──────────────────────────────────────────────────────

# Library, LLM endpoint, and web bind address come from settings.py
# (config file, then environment, then defaults).
_SETTINGS = load_settings()
LIBRARY_DIR = _SETTINGS.library
_model_bit = f" model={_SETTINGS.llm_model}" if _SETTINGS.llm_model else ""
print(f"[GhostWriter] Library: {LIBRARY_DIR}")
print(f"[GhostWriter] LLM: {redact_url(_SETTINGS.llm_base_url)}{_model_bit}")
if _SETTINGS.config_path:
    print(f"[GhostWriter] Config: {_SETTINGS.config_path}")

app = FastAPI(title="GhostWriter Web", version="0.1.0")

def _find_static_dir() -> str:
    """Static files sit next to this module, or under the install data dir.

    An editable install from a clone hits the first path. A regular
    `pip install` hits the copy installed as package data.
    """
    candidates = [
        Path(__file__).resolve().parent / "static",
        Path(sysconfig.get_path("data")) / "share" / "ghostwriter" / "static",
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return str(candidate)
    return str(candidates[0])


STATIC_DIR = _find_static_dir()
if os.path.isdir(STATIC_DIR) and os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"[GhostWriter] Static files not found at {STATIC_DIR}", file=sys.stderr)
    print("[GhostWriter] From a git clone, install with: pip install -e .", file=sys.stderr)


# ─── Helpers ────────────────────────────────────────────────────────────

def get_book_dir(name: str) -> str:
    """Resolve a book name to its directory path."""
    book_dir = os.path.join(LIBRARY_DIR, name)
    if not os.path.isdir(book_dir):
        raise HTTPException(404, f"Book '{name}' not found")
    return book_dir


def scan_chapters(book_dir: str) -> list:
    """Scan a book directory for chapter files."""
    chapters = []
    for f in sorted(glob.glob(os.path.join(book_dir, "chapter_*.md"))):
        basename = os.path.basename(f)
        # Skip versioned backups (chapter_01_v1.md, etc.)
        if re.match(r"chapter_\d+\.md$", basename):
            match = re.match(r"chapter_(\d+)\.md$", basename)
            if match:
                num = int(match.group(1))
                size = os.path.getsize(f)
                word_count = 0
                if size > 0:
                    with open(f) as fh:
                        word_count = len(fh.read().split())
                # Check for review file
                review_path = os.path.join(book_dir, f"review_{num:02d}.md")
                has_review = os.path.exists(review_path)
                # Check for version backups
                versions = sorted(glob.glob(
                    os.path.join(book_dir, f"chapter_{num:02d}_v*.md")
                ))
                chapters.append({
                    "number": num,
                    "file": basename,
                    "size": size,
                    "word_count": word_count,
                    "has_review": has_review,
                    "versions": len(versions),
                    "status": "empty" if size == 0 else "draft" if not has_review else "reviewed",
                })
    return chapters


def find_config_for_book(book_dir: str, book_name: str) -> Optional[str]:
    """Find the YAML config for a book — check inside the dir and in parent."""
    import re as _re

    def _is_versioned(path):
        """Check if a filename is a .vN.yaml backup."""
        return bool(_re.search(r'\.v\d+\.ya?ml$', os.path.basename(path)))

    def _pick_primary(files):
        """From a list of yaml files, prefer non-versioned ones."""
        primary = [f for f in files if not _is_versioned(f)]
        if primary:
            return sorted(primary)[0]
        # Fallback: return highest version if only backups exist
        return sorted(files)[-1] if files else None

    # Check inside book dir
    yamls = glob.glob(os.path.join(book_dir, "*.yaml")) + glob.glob(os.path.join(book_dir, "*.yml"))
    if yamls:
        picked = _pick_primary(yamls)
        if picked:
            return picked
    # Check parent dir for matching name
    parent = os.path.dirname(book_dir)
    for ext in ("yaml", "yml"):
        candidate = os.path.join(parent, f"{book_name}.{ext}")
        if os.path.exists(candidate):
            return candidate
    # Check all yamls in parent for matching title
    parent_yamls = glob.glob(os.path.join(parent, "*.yaml")) + glob.glob(os.path.join(parent, "*.yml"))
    # Filter out versioned backups for title matching
    parent_yamls = [f for f in parent_yamls if not _is_versioned(f)]
    for f in parent_yamls:
        try:
            with open(f) as fh:
                config = yaml.safe_load(fh)
            if isinstance(config, dict):
                title = config.get("title", "")
                slug = title.lower().replace(" ", "_").replace("'", "")
                if slug == book_name:
                    return f
        except Exception:
            continue
    return None


# ─── API: Runtime settings ──────────────────────────────────────────────

@app.get("/api/settings")
def api_settings():
    """Public runtime settings. The API key is never included."""
    return load_settings().public_dict()


# ─── API: Library ───────────────────────────────────────────────────────

@app.get("/api/books")
def list_books():
    """List all book projects in the library."""
    books = []
    if not os.path.isdir(LIBRARY_DIR):
        return {"books": []}

    for entry in sorted(os.listdir(LIBRARY_DIR)):
        entry_path = os.path.join(LIBRARY_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        # Must have at least one chapter file or a config
        chapters = scan_chapters(entry_path)
        config_path = find_config_for_book(entry_path, entry)

        if not chapters and not config_path:
            continue

        # Get title from config if available
        title = entry.replace("_", " ").title()
        config_data = None
        if config_path:
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                if isinstance(config_data, dict):
                    title = config_data.get("title", title)
            except Exception:
                pass

        total_words = sum(c["word_count"] for c in chapters)
        books.append({
            "name": entry,
            "title": title,
            "genre": config_data.get("genre", "") if config_data else "",
            "chapter_count": len(chapters),
            "total_words": total_words,
            "has_config": config_path is not None,
            "config_file": os.path.basename(config_path) if config_path else None,
        })
    return {"books": books}


# ─── API: Config ────────────────────────────────────────────────────────

@app.get("/api/books/{name}/config")
def get_config(name: str):
    """Get the YAML config for a book as JSON."""
    book_dir = get_book_dir(name)
    config_path = find_config_for_book(book_dir, name)
    if not config_path:
        raise HTTPException(404, f"No config found for '{name}'")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {"config": config, "config_path": config_path}


class ConfigUpdate(BaseModel):
    config: dict


@app.put("/api/books/{name}/config")
def update_config(name: str, update: ConfigUpdate):
    """Save updated config as YAML with auto-backup."""
    book_dir = get_book_dir(name)
    config_path = find_config_for_book(book_dir, name)
    if not config_path:
        raise HTTPException(404, f"No config found for '{name}'")

    # Auto-backup before overwriting
    import shutil, re as _re
    base, ext = os.path.splitext(config_path)
    # Strip any existing .vN suffix to avoid stacking like .v1.v2
    base = _re.sub(r'\.v\d+$', '', base)
    version = 1
    while os.path.exists(f"{base}.v{version}{ext}"):
        version += 1
    backup_path = f"{base}.v{version}{ext}"
    shutil.copy2(config_path, backup_path)

    with open(config_path, "w") as f:
        yaml.dump(update.config, f, Dumper=QuotedDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False, width=120)
    return {"status": "saved", "path": config_path, "backup": backup_path}


# ─── API: Generate Synopsis ─────────────────────────────────────────────


class SynopsisRequest(BaseModel):
    chapter_number: int
    title: str = ""
    key_events: list[str] = []
    pov_character: str = ""
    emotional_beat: str = ""
    location: str = ""
    host: Optional[str] = None
    port: Optional[int] = None


@app.post("/api/books/{name}/generate-synopsis")
async def generate_synopsis(name: str, req: SynopsisRequest):
    """Use LLM to generate a chapter synopsis from key_events."""
    book_dir = get_book_dir(name)
    config_path = find_config_for_book(book_dir, name)
    if not config_path:
        raise HTTPException(404, f"No config found for '{name}'")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    book_title = config.get("title", "Untitled")
    genre = config.get("genre", "Fiction")

    events_text = "\n".join(f"- {e}" for e in req.key_events) if req.key_events else "(no events listed)"

    prompt = f"""Write a concise chapter synopsis (2-4 sentences) for the following chapter of "{book_title}" ({genre}).

Chapter {req.chapter_number}: "{req.title}"
POV Character: {req.pov_character or 'N/A'}
Location: {req.location or 'N/A'}
Emotional Beat: {req.emotional_beat or 'N/A'}

Key Events:
{events_text}

Write ONLY the synopsis text — no labels, no markdown, no quotes. It should read as a narrative summary that a writer would use as a guide for drafting the chapter."""

    llm_url, llm_headers = load_settings().connection(req.host, req.port)
    synopsis_settings = load_settings()

    async def stream():
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            payload = {
                "model": synopsis_settings.llm_model or "any",
                "messages": [
                    {"role": "system", "content": "You are a skilled fiction editor. Write concise, evocative chapter synopses."},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            if not synopsis_settings.strict_openai:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                async with client.stream(
                    "POST", llm_url, json=payload, headers=llm_headers, timeout=120.0
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if "choices" not in data:
                            continue
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                        # Skip reasoning tokens
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ─── API: Chapters ──────────────────────────────────────────────────────

@app.get("/api/books/{name}/chapters")
def list_chapters(name: str):
    """List all chapters for a book."""
    book_dir = get_book_dir(name)
    chapters = scan_chapters(book_dir)
    return {"chapters": chapters}


@app.get("/api/books/{name}/chapters/{num}")
def get_chapter(name: str, num: int):
    """Get the content of a specific chapter."""
    book_dir = get_book_dir(name)
    chapter_path = os.path.join(book_dir, f"chapter_{num:02d}.md")
    if not os.path.exists(chapter_path):
        raise HTTPException(404, f"Chapter {num} not found")

    with open(chapter_path) as f:
        content = f.read()

    # Also get version list
    versions = sorted(glob.glob(os.path.join(book_dir, f"chapter_{num:02d}_v*.md")))
    version_info = []
    for v in versions:
        vname = os.path.basename(v)
        vmatch = re.match(r"chapter_\d+_v(\d+)\.md$", vname)
        if vmatch:
            version_info.append({
                "version": int(vmatch.group(1)),
                "file": vname,
                "size": os.path.getsize(v),
            })

    return {
        "number": num,
        "content": content,
        "word_count": len(content.split()),
        "versions": version_info,
    }


class ChapterUpdate(BaseModel):
    content: str


@app.put("/api/books/{name}/chapters/{num}")
def update_chapter(name: str, num: int, update: ChapterUpdate):
    """Save updated chapter content."""
    book_dir = get_book_dir(name)
    chapter_path = os.path.join(book_dir, f"chapter_{num:02d}.md")
    with open(chapter_path, "w") as f:
        f.write(update.content)
    return {"status": "saved", "word_count": len(update.content.split())}


# ─── API: Reviews ───────────────────────────────────────────────────────

@app.get("/api/books/{name}/reviews/{num}")
def get_review(name: str, num: int):
    """Get the review for a specific chapter."""
    book_dir = get_book_dir(name)
    review_path = os.path.join(book_dir, f"review_{num:02d}.md")
    if not os.path.exists(review_path):
        raise HTTPException(404, f"Review for chapter {num} not found")
    with open(review_path) as f:
        content = f.read()
    return {"number": num, "content": content}


class ReviewUpdate(BaseModel):
    content: str


@app.put("/api/books/{name}/reviews/{num}")
def update_review(name: str, num: int, update: ReviewUpdate):
    """Save updated review content."""
    book_dir = get_book_dir(name)
    review_path = os.path.join(book_dir, f"review_{num:02d}.md")
    with open(review_path, "w") as f:
        f.write(update.content)
    return {"status": "saved"}


# ─── API: Command Execution (SSE) ──────────────────────────────────────

GHOSTWRITER_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghostwriter.py")

# Track running jobs: book_name -> subprocess
active_jobs: dict = {}


def _ghostwriter_argv(host: Optional[str], port: Optional[int],
                      max_tokens: Optional[int], temperature: Optional[float]) -> list[str]:
    """CLI prefix that pins the child to this process's resolved endpoint.

    The child runs with its cwd set to a book directory, so it must not
    discover a different ghostwriter.toml there.
    """
    settings = load_settings()
    host = settings.llm_host if not host else host
    port = settings.llm_port if port is None else int(port)
    if max_tokens is None:
        max_tokens = settings.max_tokens
    if temperature is None:
        temperature = settings.temperature
    argv = [sys.executable, GHOSTWRITER_PY]
    if settings.uses_configured_endpoint(host, port):
        argv += ["--base-url", settings.llm_base_url]
    else:
        argv += ["--host", host, "--port", str(port)]
    argv += ["--model", settings.llm_model]
    argv += ["--strict-openai" if settings.strict_openai else "--no-strict-openai"]
    argv += ["--max-tokens", str(max_tokens), "--temperature", str(temperature)]
    return argv


def _ghostwriter_env() -> dict:
    settings = load_settings()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GHOSTWRITER_IGNORE_CWD_CONFIG": "1"}
    # Always set, including empty, so a book-directory config cannot attach
    # a different credential to this run.
    env["GHOSTWRITER_API_KEY"] = settings.llm_api_key
    return env


class RunCommand(BaseModel):
    command: str          # compose, review, revise, audit, polish, distill, evolve, restructure, pdf
    chapter: Optional[int] = None  # specific chapter (None = all)
    host: Optional[str] = None
    port: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    strategy: str = "general"     # revision strategy for revise/polish
    min_score: int = 80           # polish convergence threshold
    max_rounds: int = 3           # polish/evolve max iterations
    evolve_mode: str = "full"     # evolve mode: expand/refine/full
    threshold: int = 6            # restructure: min key_events to trigger split


@app.post("/api/books/{name}/run")
async def run_command(name: str, cmd: RunCommand, request: Request):
    """Run a GhostWriter command and stream output via SSE."""
    book_dir = get_book_dir(name)
    config_path = find_config_for_book(book_dir, name)
    if not config_path:
        raise HTTPException(404, f"No config found for '{name}'")

    # Cancel any existing job for this book
    if name in active_jobs:
        try:
            active_jobs[name].terminate()
        except Exception:
            pass
        del active_jobs[name]

    # Build the GhostWriter CLI command.
    # Global args must come BEFORE the subcommand for argparse.
    if not os.path.isfile(GHOSTWRITER_PY):
        raise HTTPException(500, f"GhostWriter CLI not found at {GHOSTWRITER_PY}")
    args = _ghostwriter_argv(cmd.host, cmd.port, cmd.max_tokens, cmd.temperature)

    if cmd.command == "compose":
        args += ["compose", config_path, "--output", "."]
        if cmd.chapter is not None:
            args += ["--chapter", str(cmd.chapter)]
    elif cmd.command == "review":
        args += ["review", config_path, "--output", "."]
        if cmd.chapter is not None:
            args += ["--chapter", str(cmd.chapter)]
    elif cmd.command == "revise":
        args += ["revise", config_path, "--output", ".",
                 "--strategy", cmd.strategy]
        if cmd.chapter is not None:
            args += ["--chapters", str(cmd.chapter)]
    elif cmd.command == "audit":
        args += ["audit", config_path, "--output", ".", "--fix"]
    elif cmd.command == "polish":
        args += ["polish", config_path, "--output", ".",
                 "--strategy", cmd.strategy,
                 "--min-score", str(cmd.min_score),
                 "--max-rounds", str(cmd.max_rounds)]
        if cmd.chapter is not None:
            args += ["--chapter", str(cmd.chapter)]
    elif cmd.command == "distill":
        args += ["distill", config_path, "--output", "."]
    elif cmd.command == "evolve":
        args += ["evolve", config_path,
                 "--mode", cmd.evolve_mode,
                 "--max-rounds", str(cmd.max_rounds)]
    elif cmd.command == "restructure":
        args += ["restructure", config_path,
                 "--threshold", str(cmd.threshold)]
    elif cmd.command == "pdf":
        args += ["pdf", "--config", config_path]
    else:
        raise HTTPException(400, f"Unknown command: {cmd.command}")

    async def event_stream():
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_ghostwriter_env(),
            cwd=book_dir,
        )
        active_jobs[name] = proc

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                yield f"data: {text}\n\n"

            await proc.wait()
            yield f"data: [EXIT:{proc.returncode}]\n\n"
        except asyncio.CancelledError:
            proc.terminate()
            yield f"data: [CANCELLED]\n\n"
        finally:
            active_jobs.pop(name, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/books/{name}/cancel")
async def cancel_command(name: str):
    """Cancel a running command for a book."""
    if name in active_jobs:
        proc = active_jobs[name]
        try:
            proc.terminate()
            await asyncio.sleep(0.5)
            if proc.returncode is None:
                proc.kill()
        except Exception:
            pass
        active_jobs.pop(name, None)
        return {"status": "cancelled"}
    return {"status": "no_job"}


@app.get("/api/books/{name}/job_status")
async def job_status(name: str):
    """Check if a command is running for a book."""
    if name in active_jobs:
        proc = active_jobs[name]
        return {"running": proc.returncode is None, "pid": proc.pid}
    return {"running": False}


# ─── API: Chat (Tool-Calling AI) ───────────────────────────────────────

import json
import httpx

# Tool schemas for the book AI (OpenAI function-calling format)
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_config",
            "description": "Read a section of the book config. Sections: metadata, style, characters, chapters, events, worldbuilding, all",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["metadata", "style", "characters", "chapters", "events", "worldbuilding", "all"],
                        "description": "Which config section to read"
                    }
                },
                "required": ["section"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_config",
            "description": "Search the full book config for a keyword. Returns matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or name to search for (case-insensitive)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_chapter",
            "description": "Load the full text of a specific chapter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_number": {
                        "type": "integer",
                        "description": "Chapter number to read (1-indexed)"
                    }
                },
                "required": ["chapter_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_config_update",
            "description": "Propose a config change. Actions: add_chapter, update_chapter, add_character, update_field, replace_section. Put ALL changes in ONE call. For add_chapter/update_chapter pass chapter fields in json_data. For add_character pass character fields in json_data. For update_field pass the value in json_data. For replace_section pass the section content in json_data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What this change does"
                    },
                    "action": {
                        "type": "string",
                        "description": "One of: add_chapter, update_chapter, add_character, update_field, replace_section"
                    },
                    "chapter_number": {
                        "type": "integer",
                        "description": "For update_chapter: which chapter number to update"
                    },
                    "field_name": {
                        "type": "string",
                        "description": "For update_field/replace_section: the field or section name"
                    },
                    "insert_after": {
                        "type": "integer",
                        "description": "For add_chapter: insert after this chapter number (0=beginning)"
                    },
                    "json_data": {
                        "type": "string",
                        "description": "JSON string with the data. For chapters: {\"title\":\"...\",\"synopsis\":\"...\",\"pov_character\":\"...\",\"emotional_beat\":\"...\",\"location\":\"...\",\"key_events\":[...]}. For characters: {\"name\":\"...\",\"role\":\"...\",\"description\":\"...\"}. For update_field: the raw value as JSON."
                    }
                },
                "required": ["description", "action", "json_data"]
            }
        }
    },
]


def _execute_chat_tool(tool_name: str, args: dict, book_name: str, book_dir: str, config: dict, config_path: str) -> tuple[str, dict | None]:
    """Execute a chat tool call. Returns (result_text, proposal_or_none)."""
    if tool_name == "get_config":
        section = args.get("section", "all")
        if section == "all":
            return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False), None
        elif section == "metadata":
            meta = {k: config.get(k) for k in ["title", "genre", "setting", "pov", "target_audience", "synopsis"]
                    if k in config}
            return yaml.dump(meta, default_flow_style=False, allow_unicode=True), None
        elif section in config:
            return yaml.dump({section: config[section]}, default_flow_style=False, allow_unicode=True), None
        else:
            return f"Section '{section}' not found in config. Available: {', '.join(config.keys())}", None

    elif tool_name == "search_config":
        query = args.get("query", "").lower()
        config_yaml = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        lines = config_yaml.split("\n")
        matches = []
        for i, line in enumerate(lines):
            if query in line.lower():
                # Show context: 1 line before and after
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "\n".join(f"  {lines[j]}" for j in range(start, end))
                matches.append(f"Line {i + 1}:\n{context}")
        if matches:
            return f"Found {len(matches)} match(es) for '{query}':\n\n" + "\n\n".join(matches[:10]), None
        return f"No matches found for '{query}' in the config.", None

    elif tool_name == "read_chapter":
        num = args.get("chapter_number", 1)
        chapter_path = os.path.join(book_dir, f"chapter_{num:02d}.md")
        if not os.path.exists(chapter_path):
            return f"Chapter {num} not found.", None
        with open(chapter_path) as f:
            content = f.read()
        if not content.strip():
            return f"Chapter {num} exists but is empty.", None
        # Truncate if very long
        if len(content) > 12000:
            content = content[:12000] + f"\n\n... [truncated, {len(content)} total chars]"
        return f"# Chapter {num}\n\n{content}", None

    elif tool_name == "propose_config_update":
        import copy
        description = args.get("description", "Config update")
        action = args.get("action", "")
        json_data_str = args.get("json_data", "{}")

        # Parse json_data
        try:
            data = json.loads(json_data_str) if isinstance(json_data_str, str) else json_data_str
        except json.JSONDecodeError as e:
            return f"Error: invalid json_data: {e}", None

        # Work on a deep copy so the original config isn't mutated
        merged = copy.deepcopy(config)

        if action == "add_chapter":
            if not isinstance(data, dict) or not data.get("title"):
                return "Error: json_data must be a JSON object with at least a title", None

            chapters = merged.get("chapters", [])
            new_ch = {"number": len(chapters) + 1}
            for field in ["title", "synopsis", "pov_character", "emotional_beat", "location", "key_events"]:
                if field in data:
                    new_ch[field] = data[field]

            insert_after = args.get("insert_after")
            if insert_after is not None and 0 <= insert_after < len(chapters):
                chapters.insert(insert_after, new_ch)
                for i, ch in enumerate(chapters):
                    if isinstance(ch, dict):
                        ch["number"] = i + 1
            else:
                chapters.append(new_ch)

            merged["chapters"] = chapters

        elif action == "update_chapter":
            ch_num = args.get("chapter_number")
            if not ch_num:
                return "Error: chapter_number is required for update_chapter", None

            chapters = merged.get("chapters", [])
            target = None
            for ch in chapters:
                if isinstance(ch, dict) and ch.get("number") == ch_num:
                    target = ch
                    break
            if not target:
                return f"Error: chapter {ch_num} not found", None

            if isinstance(data, dict):
                for field, value in data.items():
                    target[field] = value

        elif action == "add_character":
            if not isinstance(data, dict) or not data.get("name"):
                return "Error: json_data must be a JSON object with at least a name", None

            chars = merged.get("characters", [])
            chars.append(data)
            merged["characters"] = chars

        elif action == "update_field":
            field_name = args.get("field_name", "")
            if not field_name:
                return "Error: field_name is required for update_field", None
            merged[field_name] = data

        elif action == "replace_section":
            field_name = args.get("field_name", "")
            if not field_name:
                return "Error: field_name is required for replace_section", None
            merged[field_name] = data

        else:
            return f"Unknown action: {action}. Use: add_chapter, update_chapter, add_character, update_field, replace_section", None

        # Validate serialization
        try:
            json.dumps(merged, default=str)
        except (TypeError, ValueError) as e:
            return f"Merged config is not serializable: {e}", None

        proposal = {
            "description": description,
            "section": "full",
            "section_data": merged,
        }
        return f"📋 Proposed: {description}", proposal

    return f"Unknown tool: {tool_name}", None


def _build_chat_system_prompt(config: dict, book_dir: str) -> str:
    """Build the system prompt for the book AI chat."""
    # Flatten metadata if nested
    if "metadata" in config and isinstance(config["metadata"], dict):
        for k, v in config["metadata"].items():
            if k not in config:
                config[k] = v

    title = config.get("title", "Untitled")
    genre = config.get("genre", "Unknown")
    setting = config.get("setting", "")
    synopsis = config.get("synopsis", "")

    # Chapter summary
    chapters = scan_chapters(book_dir)
    chapter_lines = []
    config_chapters = config.get("chapters", [])
    if isinstance(config_chapters, dict):
        config_chapters = list(config_chapters.values())
    for ch in chapters:
        num = ch["number"]
        ch_title = ""
        if config_chapters and num <= len(config_chapters):
            ch_info = config_chapters[num - 1]
            if isinstance(ch_info, dict):
                ch_title = ch_info.get("title", "")
        status = ch["status"]
        words = ch["word_count"]
        chapter_lines.append(f"  Ch {num}: {ch_title} — {words:,} words ({status})")

    chapters_overview = "\n".join(chapter_lines) if chapter_lines else "  No chapters written yet."

    # Character names
    chars = config.get("characters", [])
    char_names = ", ".join(
        c.get("name", "?") for c in chars if isinstance(c, dict)
    ) if chars else "None defined"

    return f"""You are the Book AI assistant for "{title}".

BOOK OVERVIEW:
  Title: {title}
  Genre: {genre}
  Setting: {setting}
  Synopsis: {synopsis}

CHARACTERS: {char_names}

CHAPTERS:
{chapters_overview}

YOUR ROLE:
- Answer questions about the book's plot, characters, world, and structure
- Help the author brainstorm and develop their story
- When asked to make changes, use the propose_config_update tool with the appropriate action
- Use get_config to read detailed config sections before making changes
- Use search_config to find specific details
- Use read_chapter to load chapter text when discussing specific scenes

CONFIG SCHEMA (required fields for the compose pipeline):
  The following top-level keys are REQUIRED:
    title: string            # Book title (MUST be top-level, NOT nested under metadata)
    genre: string
    synopsis: string         # Book-level synopsis
    pov: string              # e.g. "third-person limited" or "first-person"
    characters:              # List of character dicts
      - name: string
        role: string
        description: string
    chapters:                # List of chapter dicts
      - number: int          # Sequential chapter number
        title: string
        synopsis: string     # IMPORTANT: use "synopsis", NOT "summary" — this drives prose generation
        pov_character: string # Name matching a character entry
        emotional_beat: string # e.g. "grief mixed with determination"
        location: string     # Where the chapter takes place
        key_events:          # List of plot points that MUST happen in this chapter
          - string           # e.g. "The detective finds the hidden lab"

  OPTIONAL top-level keys:
    meta:
      target_words_per_chapter: int  # Default 2500

  All other keys are freeform — add whatever enriches the story (world details,
  themes, style notes, character quirks, locations, technology, etc.).
  The more detail you provide, the richer the generated prose will be.

TOOL USAGE RULES:
- Always read the relevant config section with get_config BEFORE proposing changes
- Use propose_config_update with a SINGLE call — combine ALL changes into one
- Available actions: add_chapter, update_chapter, add_character, update_field, replace_section
- For add_chapter: provide chapter_data with title, synopsis, pov_character, etc.
- For update_chapter: provide chapter_number and the fields to change in chapter_data
- For add_character: provide character_data with name, role, description
- For update_field: provide field_name and field_value
- NEVER call propose_config_update more than once per response
- Use "synopsis" NOT "summary" for chapter descriptions — compose reads "synopsis"
- Be creative and helpful, but respect the author's vision"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    host: Optional[str] = None
    port: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    context_size: int = 65536
    enable_thinking: bool = True


MAX_TOOL_ROUNDS = 6


@app.post("/api/books/{name}/chat")
async def chat_with_book(name: str, req: ChatRequest):
    """Chat with the book AI — tool-calling agent with SSE streaming."""
    book_dir = get_book_dir(name)
    config_path = find_config_for_book(book_dir, name)
    if not config_path:
        raise HTTPException(404, f"No config found for '{name}'")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    system_prompt = _build_chat_system_prompt(config, book_dir)
    chat_settings = load_settings()
    llm_url, llm_headers = chat_settings.connection(req.host, req.port)

    # Build initial messages
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    async def event_stream():
        nonlocal config  # May be updated by proposals
        turn_messages = messages.copy()

        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            for round_num in range(MAX_TOOL_ROUNDS + 1):
                # Estimate prompt tokens (~4 chars/token) and cap max_tokens
                prompt_chars = sum(len(m.get("content", "") or "") for m in turn_messages)
                prompt_tokens_est = prompt_chars // 3  # conservative estimate
                context_limit = req.context_size
                remaining = max(2048, context_limit - prompt_tokens_est)
                requested_max = req.max_tokens if req.max_tokens is not None else chat_settings.max_tokens
                effective_max_tokens = min(requested_max, remaining)

                payload = {
                    "model": chat_settings.llm_model or "any",
                    "messages": turn_messages,
                    "stream": True,
                    "temperature": req.temperature,
                    "max_tokens": effective_max_tokens,
                    "tools": CHAT_TOOLS,
                }
                if not chat_settings.strict_openai:
                    payload["chat_template_kwargs"] = {"enable_thinking": req.enable_thinking}

                full_text = ""
                tool_calls_accum = {}

                try:
                    async with client.stream(
                        "POST", llm_url, json=payload, headers=llm_headers, timeout=180.0
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if "choices" not in data:
                                continue

                            delta = data["choices"][0].get("delta", {})

                            # Text content — stream to browser
                            token = delta.get("content", "")
                            if token:
                                full_text += token
                                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                            # Reasoning/thinking (Qwen 3.5)
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                yield f"data: {json.dumps({'type': 'thinking', 'content': reasoning})}\n\n"

                            # Tool call deltas — accumulate
                            if "tool_calls" in delta:
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta.get("index", 0)
                                    if idx not in tool_calls_accum:
                                        tool_calls_accum[idx] = {
                                            "id": tc_delta.get("id", f"call_{idx}"),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    tc = tool_calls_accum[idx]
                                    if tc_delta.get("id"):
                                        tc["id"] = tc_delta["id"]
                                    func = tc_delta.get("function", {})
                                    if func.get("name"):
                                        tc["function"]["name"] = func["name"]
                                    if func.get("arguments"):
                                        tc["function"]["arguments"] += func["arguments"]

                except httpx.HTTPStatusError as e:
                    detail = ""
                    try:
                        detail = (await e.response.aread()).decode("utf-8", errors="replace")[:300]
                    except Exception:
                        detail = ""
                    message = f"LLM API error: {e.response.status_code} {detail}".strip()
                    yield f"data: {json.dumps({'type': 'error', 'content': message})}\n\n"
                    return
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    return

                # Build final tool calls list
                tool_calls = [tool_calls_accum[idx] for idx in sorted(tool_calls_accum.keys())]

                if tool_calls:
                    # Add assistant message with tool calls to turn state
                    turn_messages.append({
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": tool_calls,
                    })

                    has_proposal = False
                    for tc in tool_calls:
                        tc_id = tc["id"]
                        tool_name = tc["function"]["name"]
                        try:
                            raw_args = tc["function"]["arguments"]
                            tool_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Notify browser about tool execution
                        yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name, 'args': tool_args})}\n\n"

                        # Execute the tool
                        result_text, proposal = _execute_chat_tool(
                            tool_name, tool_args, name, book_dir, config, config_path
                        )

                        # If it's a config proposal, send it to the browser
                        if proposal:
                            has_proposal = True
                            yield f"data: {json.dumps({'type': 'proposal', 'description': proposal['description'], 'section': proposal['section'], 'section_data': proposal['section_data']}, default=str)}\n\n"

                        # Notify browser about tool result
                        yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'content': result_text[:2000]})}\n\n"

                        # Add tool result to turn state for continuation
                        truncated = result_text[:4000]
                        if len(result_text) > 4000:
                            truncated += f"\n... ({len(result_text)} chars total)"
                        turn_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": truncated,
                        })

                    # Stop after a proposal — don't let the model call more tools
                    if has_proposal:
                        break

                    # Continue the loop for another LLM call
                    continue
                else:
                    # No tool calls — done
                    break

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── API: Genesis ──────────────────────────────────────────────────────

class GenesisRequest(BaseModel):
    prompt: str
    host: Optional[str] = None
    port: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@app.post("/api/genesis")
async def run_genesis(req: GenesisRequest):
    """Generate a new book config from a creative prompt via SSE."""
    import re as _re

    # Create a slug from the prompt for the output path
    slug = _re.sub(r'[^a-z0-9]+', '_', req.prompt.lower().strip())[:60].strip('_')
    output_dir = os.path.join(LIBRARY_DIR, slug)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{slug}.yaml")

    if not os.path.isfile(GHOSTWRITER_PY):
        raise HTTPException(500, f"GhostWriter CLI not found at {GHOSTWRITER_PY}")
    args = _ghostwriter_argv(req.host, req.port, req.max_tokens, req.temperature)
    args += ["genesis", req.prompt, "--output", output_path]

    async def event_stream():
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_ghostwriter_env(),
        )

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                yield f"data: {text}\n\n"

            await proc.wait()
            if proc.returncode == 0:
                yield f"data: [BOOK:{slug}]\n\n"
            yield f"data: [EXIT:{proc.returncode}]\n\n"
        except asyncio.CancelledError:
            proc.terminate()
            yield f"data: [CANCELLED]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



# ─── Root → index.html ─────────────────────────────────────────────────

@app.get("/")
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {
        "message": "GhostWriter Web — static files not found. "
                   "From a git clone, install with: pip install -e ."
    }


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    """Serve the web UI. Bind address defaults to 0.0.0.0:8501."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="GhostWriter web UI")
    parser.add_argument("--host", default=settings.web_host,
                        help="Interface to bind (default from config, else 0.0.0.0). "
                             "This is the web app, not the LLM.")
    parser.add_argument("--port", type=int, default=settings.web_port,
                        help="Port to bind (default 8501)")
    parser.add_argument("--no-reload", action="store_true",
                        help="Do not restart when source files change")
    args = parser.parse_args()
    import uvicorn
    print(f"[GhostWriter] Web UI listening on {args.host}:{args.port}")
    print(f"[GhostWriter] Open http://127.0.0.1:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()

