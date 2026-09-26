#!/usr/bin/env python3
"""
GhostWriter — AI-powered fiction writing system using local LLMs.

Pipeline:
  1. GENESIS:  prompt → structured YAML config (characters, settings, events, outline)
  2. OUTLINE:  config → detailed chapter-by-chapter outline
  3. COMPOSE:  outline → full chapters, with character-perspective simulation
  4. REVIEW:   chapters → self-critique and revision pass

Usage:
  # Generate config from a prompt
  python3 ghostwriter.py genesis "A noir detective story set in 1940s LA..."

  # Generate book from existing config
  python3 ghostwriter.py compose my_story.yaml

  # Full pipeline (genesis → compose)
  python3 ghostwriter.py write "A noir detective story set in 1940s LA..."

  # Review/revise an existing draft
  python3 ghostwriter.py review my_story.yaml
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from settings import load_settings, redact_url, set_settings


# ─── defaults ───────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMP = 0.88
DEFAULT_TOP_P = 0.92
DEFAULT_MIN_P = 0.08
DEFAULT_REPEAT_PENALTY = 1.22

CHAPTER_TARGET_WORDS = 2500   # target words per chapter
CONTINUATION_OVERLAP = 200    # chars of overlap when continuing a chapter
REPETITION_WINDOW = 150       # chars to check for repeating chunks
REPETITION_THRESHOLD = 3      # how many times a chunk must repeat to trigger

# ─── config helpers ────────────────────────────────────────────────────

def get_chapters_list(config: dict) -> list:
    """Extract chapters from config, handling all format variations."""
    ch = config.get("chapters", [])
    # Unwrap double-nesting: chapters: { chapters: [...] }
    if isinstance(ch, dict):
        ch = ch.get("chapters", list(ch.values()))
    if not isinstance(ch, list):
        ch = list(ch) if ch else []
    return ch


def ch_get(ch_info, key: str, default=""):
    """Safely get a key from a chapter info entry (may be str or dict)."""
    if isinstance(ch_info, dict):
        return ch_info.get(key, default)
    return default


# ─── text processing ───────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove thinking/reasoning tags from LLM output."""
    # <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # [Start thinking]...[End thinking] tags
    text = re.sub(r'\[Start thinking\].*?\[End thinking\]', '', text, flags=re.DOTALL)
    # Unclosed thinking tags — strip from <think> to end
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Start thinking\].*$', '', text, flags=re.DOTALL)
    return text.strip()


def detect_repetition(text: str, window: int = REPETITION_WINDOW,
                       threshold: int = REPETITION_THRESHOLD) -> tuple:
    """Detect if text has entered a repetition loop.
    Returns (is_looping, repeated_chunk, clean_text).

    Uses two strategies:
    1. Check if a tail chunk of the text appears multiple times in the recent portion
    2. Check if the same sentence is repeated several times at the end
    """
    if len(text) < window * 2:
        return False, "", text

    # Strategy 1: Check various tail chunk sizes for repeats
    for chunk_size in (120, 80, 50):
        if len(text) < chunk_size * 3:
            continue
        tail = text[-chunk_size:]
        # Search in the region before the tail
        search_end = len(text) - chunk_size
        count = 0
        pos = max(0, search_end - chunk_size * 6)
        while pos < search_end:
            idx = text.find(tail, pos, search_end)
            if idx == -1:
                break
            count += 1
            pos = idx + 1
            if count >= threshold - 1:  # -1 because the tail itself is one occurrence
                # Truncate: keep up to the first repeat + one copy
                clean = text[:idx + chunk_size].rstrip()
                return True, tail[:60], clean

    # Strategy 2: Check for repeated sentences at the end
    tail_text = text[-2000:] if len(text) > 2000 else text
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', tail_text) if len(s.strip()) > 20]
    if len(sentences) >= 4:
        last = sentences[-1]
        repeat_count = sum(1 for s in sentences[-6:] if s == last)
        if repeat_count >= 3:
            # Find second-to-last occurrence and truncate after it
            idx = text.rfind(last)
            if idx > 0:
                prev_idx = text.rfind(last, 0, idx)
                if prev_idx > 0:
                    clean = text[:prev_idx + len(last)].rstrip()
                    return True, last[:60], clean

    # Strategy 3: Run-on detection — model degenerates into stream-of-consciousness
    # without sentence-ending punctuation. If the tail has no period/!/? for 600+ chars,
    # it's degenerate output.
    last_sentence_end = max(
        text.rfind('.'),
        text.rfind('!'),
        text.rfind('?'),
        text.rfind('."'),
        text.rfind('!"'),
        text.rfind('?"'),
    )
    if last_sentence_end > 0:
        runon_length = len(text) - last_sentence_end
        if runon_length > 600:
            # Truncate to the last clean sentence
            clean = text[:last_sentence_end + 1].rstrip()
            return True, f"[run-on: {runon_length} chars without sentence end]", clean

    # Strategy 4: Soft-loop detection — model paraphrases its earlier content.
    # Uses word-level shingling to detect when the second half echoes the first.
    # Triggers around 8000 chars where Gemma 4 and similar models lose position.
    SOFT_LOOP_MIN_CHARS = 6000
    SHINGLE_SIZE = 3
    SOFT_LOOP_THRESHOLD = 0.40  # 40% shingle overlap = paraphrasing

    if len(text) >= SOFT_LOOP_MIN_CHARS:
        mid = len(text) // 2
        # Find sentence boundary nearest to midpoint
        for offset in range(0, 500):
            if mid + offset < len(text) and text[mid + offset] in '.!?':
                mid = mid + offset + 1
                break
            if mid - offset > 0 and text[mid - offset] in '.!?':
                mid = mid - offset + 1
                break

        first_half = text[:mid].lower()
        second_half = text[mid:].lower()

        # Extract word shingles
        words_first = re.findall(r'\b\w+\b', first_half)
        words_second = re.findall(r'\b\w+\b', second_half)

        if len(words_second) >= SHINGLE_SIZE * 3:
            shingles_first = set()
            for i in range(len(words_first) - SHINGLE_SIZE + 1):
                shingles_first.add(tuple(words_first[i:i + SHINGLE_SIZE]))

            shingles_second = set()
            for i in range(len(words_second) - SHINGLE_SIZE + 1):
                shingles_second.add(tuple(words_second[i:i + SHINGLE_SIZE]))

            if shingles_second:
                overlap = len(shingles_first & shingles_second)
                similarity = overlap / len(shingles_second)

                if similarity >= SOFT_LOOP_THRESHOLD:
                    # Truncate to the midpoint — keep only the first telling
                    clean = text[:mid].rstrip()
                    return True, f"[soft-loop: {similarity:.0%} shingle overlap between halves]", clean

    return False, "", text


# ─── LLM client ────────────────────────────────────────────────────────

def llm_chat(messages: list, host: str, port: int,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             temperature: float = DEFAULT_TEMP,
             top_p: float = DEFAULT_TOP_P,
             min_p: float = DEFAULT_MIN_P,
             repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
             stop: list = None) -> str:
    """Send a streaming chat completion request to an OpenAI-compatible server.

    host/port select the endpoint. When they match the configured endpoint,
    the configured base URL is used, so https:// and non-default paths work.
    Monitors output for repetition loops and aborts early if detected.
    """
    settings = load_settings()
    url = settings.chat_url(host, port)
    payload = {
        "model": settings.llm_model or "any",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    # llama.cpp sampling extensions. Strict OpenAI-compatible servers 400 on these.
    if not settings.strict_openai:
        payload["min_p"] = min_p
        payload["repeat_penalty"] = repeat_penalty
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers=settings.request_headers(host, port),
    )

    content = ""
    check_interval = 500  # check for repetition every N chars
    last_check_len = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        content += token

                        # Periodic repetition check
                        if len(content) - last_check_len >= check_interval:
                            last_check_len = len(content)
                            is_loop, chunk_text, clean = detect_repetition(content)
                            if is_loop:
                                print(f"\n  ⚠ REPETITION DETECTED: \"{chunk_text}...\"")
                                print(f"    Truncating at {len(clean)} chars (was {len(content)})")
                                content = clean
                                break
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        print(f"  ERROR: LLM request to {url} failed: HTTP {e.code} {detail}", file=sys.stderr)
        return ""
    except urllib.error.URLError as e:
        print(f"  ERROR: LLM request to {url} failed: {e.reason}", file=sys.stderr)
        return ""

    # Strip thinking tags from output
    content = strip_thinking(content)
    return content


def llm_generate(system: str, user: str, **kwargs) -> str:
    """Convenience: system + user → response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return llm_chat(messages, **kwargs)


# ─── config schema ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "title": "",
    "genre": "",
    "tone": "",
    "pov": "third-person limited",
    "setting": {
        "time_period": "",
        "locations": [],
        "world_details": "",
    },
    "characters": [],
    "themes": [],
    "events": [],
    "chapters": [],
    "meta": {
        "target_chapter_count": 12,
        "target_words_per_chapter": CHAPTER_TARGET_WORDS,
        "style_notes": "",
    },
}

CHARACTER_TEMPLATE = {
    "name": "",
    "role": "",  # protagonist, antagonist, supporting, etc.
    "physical": "",
    "psychological": "",
    "backstory": "",
    "voice": "",  # how they speak — dialect, vocabulary, mannerisms
    "arc": "",    # character development across the story
    "relationships": [],
}


# ─── GENESIS: prompt → config ──────────────────────────────────────────

GENESIS_SYSTEM = """You are a master story architect. Given a creative prompt, you produce a detailed story configuration in YAML format.

Output ONLY valid YAML (no markdown fences, no commentary). The YAML must follow this exact structure:

title: "Story Title"
genre: "genre"
tone: "tone description"
pov: "first-person|third-person limited|third-person omniscient"
setting:
  time_period: "when"
  locations:
    - name: "Place Name"
      description: "vivid sensory description of this place"
      mood: "the emotional atmosphere — foreboding, serene, electric, etc."
      sensory: "sounds, smells, textures, light quality that define this place"
  world_details: "any world-building context"
characters:
  - name: "Full Name"
    role: "protagonist|antagonist|supporting|mentor|love interest"
    physical: "detailed physical description — height, build, face, hair, distinguishing marks, how they move, what they wear"
    psychological: "inner world — core desires, deepest fears, defense mechanisms, contradictions, what they lie to themselves about"
    backstory: "relevant history that shaped who they are"
    voice: "how they speak — dialect, vocabulary, sentence patterns, verbal tics, what they avoid saying"
    arc: "how they change across the story — what breaks, what heals, what they learn or refuse to learn"
    relationships:
      - target: "Other Character Name"
        nature: "relationship description — the tension, the history, the unspoken"
themes:
  - "theme 1"
  - "theme 2"
scenes:
  - name: "Scene Name"
    location: "Place Name"
    description: "what this specific scene looks, sounds, and feels like"
    mood: "the emotional temperature of this scene"
    time_of_day: "morning|afternoon|evening|night|dawn|dusk"
    weather: "weather/atmosphere if relevant"
events:
  - chapter: 1
    summary: "what happens — be specific about actions, discoveries, confrontations"
    characters_present: ["Name1", "Name2"]
    emotional_beat: "tension|release|revelation|climax|betrayal|loss|hope|dread"
    consequences: "what this event changes — what can never go back to how it was"
chapters:
  - number: 1
    title: "Chapter Title"
    synopsis: "2-3 sentence synopsis with specific plot beats"
    pov_character: "Name"
    scene: "Scene Name"
    time: "when in the story timeline"
meta:
  target_chapter_count: 12
  target_words_per_chapter: 2500
  style_notes: "any specific prose style guidance"

Create compelling, three-dimensional characters with genuine flaws and contradictions. Physical descriptions should be vivid enough to visualize. Psychological profiles should reveal the gap between who they appear to be and who they really are. Plan meaningful character arcs. Scenes should be atmospheric and immersive. Events should have real consequences that ripple forward. Make each chapter synopsis specific enough to write from.

CRITICAL RULES:
- You MUST generate entries for ALL chapters up to target_chapter_count (typically 12).
- The chapters list must have one entry per chapter, each with a unique synopsis.
- The events list must cover the full arc — do not stop at chapter 3.
- Each chapter synopsis should be specific enough to write from — include WHO does WHAT and WHY it matters.
- Do NOT abbreviate or say "continue similarly" — enumerate every single chapter."""


def genesis(prompt: str, host: str, port: int, **kwargs) -> dict:
    """Generate a story config from a creative prompt."""
    print("📖 GENESIS: Generating story configuration...")
    print(f"   Prompt: {prompt[:80]}...")

    response = llm_generate(
        GENESIS_SYSTEM,
        f"Create a detailed story configuration for the following premise. You MUST define ALL 12 chapters with full synopses — do not stop early:\n\n{prompt}",
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", 8192),
        temperature=kwargs.get("temperature", 0.9),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )

    # Parse YAML from response — with progressive fixup attempts
    cleaned = response.strip()

    # Strip thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\[Start thinking\].*?\[End thinking\]', '', cleaned, flags=re.DOTALL)

    # Strip markdown fences
    cleaned = re.sub(r'^```ya?ml\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    config = None
    parse_error = None

    # Attempt 1: raw parse
    try:
        config = yaml.safe_load(cleaned)
        if not isinstance(config, dict):
            raise ValueError("YAML did not parse to a dictionary")
    except Exception as e:
        parse_error = e

    # Attempt 2: fix unclosed quotes (common LLM issue)
    if config is None:
        try:
            # Find lines with odd number of quotes and close them
            lines = cleaned.split('\n')
            fixed_lines = []
            for line in lines:
                if line.count('"') % 2 != 0:
                    line = line.rstrip() + '"'
                fixed_lines.append(line)
            fixed = '\n'.join(fixed_lines)
            config = yaml.safe_load(fixed)
            if not isinstance(config, dict):
                raise ValueError("YAML did not parse to a dictionary")
            print("  ✓ Fixed unclosed quotes in YAML")
        except Exception:
            pass

    # Attempt 3: truncate at the last valid top-level key
    if config is None:
        try:
            # Find last line that starts a top-level key (no leading whitespace)
            lines = cleaned.split('\n')
            last_valid = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if line and not line[0].isspace() and ':' in line:
                    # Try parsing up to and including this section
                    candidate = '\n'.join(lines[:i])
                    try:
                        test = yaml.safe_load(candidate)
                        if isinstance(test, dict):
                            config = test
                            print(f"  ✓ Truncated YAML at line {i} to fix parse error")
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    # Attempt 3b: fix unquoted colons in list items (prose with colons)
    if config is None:
        try:
            repaired = _yaml_repair(cleaned)
            config = yaml.safe_load(repaired)
            if not isinstance(config, dict):
                raise ValueError("YAML did not parse to a dictionary")
            print("  ✓ Fixed unquoted colons in YAML list items")
        except Exception:
            pass

    if config is None:
        print(f"  ⚠ YAML parse failed: {parse_error}")
        print(f"  Saving raw response for manual editing...")
        config = {"_raw_response": response, "_parse_error": str(parse_error)}

    # Fix dict entries in key_events
    if isinstance(config, dict) and "_parse_error" not in config:
        config = _fix_key_events_dicts(config)

    # ─── Backfill missing chapters ───
    # The model often generates fewer chapter entries than target_chapter_count.
    # Auto-generate placeholder entries from events + characters.
    if isinstance(config, dict) and "_parse_error" not in config:
        target_count = config.get("meta", {}).get("target_chapter_count", 12)
        chapters = get_chapters_list(config)
        events = config.get("events", [])
        characters = config.get("characters", [])

        if len(chapters) < target_count:
            print(f"  ⚠ Genesis produced {len(chapters)}/{target_count} chapters — backfilling...")

            # Get character names for POV rotation
            char_names = [c.get("name", "Unknown") for c in characters if c.get("role") in ("protagonist", "supporting", None, "")]
            if not char_names:
                char_names = [c.get("name", "Unknown") for c in characters[:3]] if characters else ["Narrator"]

            for ch_num in range(len(chapters) + 1, target_count + 1):
                # Match event if available
                matching_event = None
                for ev in events:
                    if isinstance(ev, dict) and ev.get("chapter") == ch_num:
                        matching_event = ev
                        break

                if matching_event:
                    synopsis = matching_event.get("summary", f"Continue the story — chapter {ch_num}")
                    emotional_beat = matching_event.get("emotional_beat", "continuation")
                else:
                    # Generate a progressive synopsis based on story arc position
                    arc_position = ch_num / target_count
                    if arc_position < 0.25:
                        synopsis = f"Chapter {ch_num}: Deepen the world and character relationships. Introduce new tensions."
                    elif arc_position < 0.5:
                        synopsis = f"Chapter {ch_num}: Rising action — complications multiply. Stakes escalate."
                    elif arc_position < 0.75:
                        synopsis = f"Chapter {ch_num}: Major turning point. Something breaks that cannot be repaired."
                    elif arc_position < 0.92:
                        synopsis = f"Chapter {ch_num}: Racing toward climax. Confrontations become unavoidable."
                    else:
                        synopsis = f"Chapter {ch_num}: Resolution. The aftermath of what was broken and what survived."
                    emotional_beat = "continuation"

                pov_char = char_names[(ch_num - 1) % len(char_names)]

                chapters.append({
                    "number": ch_num,
                    "title": f"Chapter {ch_num}",
                    "synopsis": synopsis,
                    "pov_character": pov_char,
                    "scene": "",
                    "time": "",
                })

            config["chapters"] = chapters
            print(f"  ✓ Backfilled to {len(chapters)} chapters (edit story_config.yaml to customize synopses)")

    return config


# ─── COMPOSE: config → chapters ────────────────────────────────────────

def build_character_context(config: dict) -> str:
    """Build a character reference sheet from config."""
    chars = config.get("characters", [])
    if not chars:
        return "No character information available."

    lines = ["CHARACTER REFERENCE:"]
    for c in chars:
        name = c.get("name", "Unknown")
        lines.append(f"\n{name} ({c.get('role', 'unknown role')})")
        if c.get("physical"):
            lines.append(f"  Appearance: {c['physical']}")
        if c.get("psychological"):
            lines.append(f"  Psychology: {c['psychological']}")
        elif c.get("personality"):  # backwards compat with old configs
            lines.append(f"  Personality: {c['personality']}")
        if c.get("voice"):
            lines.append(f"  Voice: {c['voice']}")
        if c.get("arc"):
            lines.append(f"  Arc: {c['arc']}")
        for rel in c.get("relationships", []):
            lines.append(f"  → {rel.get('target', '?')}: {rel.get('nature', '?')}")

    return "\n".join(lines)


def build_story_context(config: dict) -> str:
    """Build the story bible for the system prompt."""
    parts = []

    parts.append(f"TITLE: {config.get('title', 'Untitled')}")
    parts.append(f"GENRE: {config.get('genre', 'fiction')}")
    parts.append(f"TONE: {config.get('tone', 'literary')}")
    parts.append(f"POV: {config.get('pov', 'third-person limited')}")

    setting = config.get("setting", {})
    if setting:
        parts.append(f"\nSETTING: {setting.get('time_period', '')}")
        for loc in setting.get("locations", []):
            if isinstance(loc, dict):
                parts.append(f"  - {loc.get('name', '')}: {loc.get('description', '')}")
            else:
                parts.append(f"  - {loc}")
        if setting.get("world_details"):
            parts.append(f"  World: {setting['world_details']}")

    parts.append(f"\nTHEMES: {', '.join(config.get('themes', []))}")

    parts.append("\n" + build_character_context(config))

    # Add style notes
    meta = config.get("meta", {})
    if meta.get("style_notes"):
        parts.append(f"\nSTYLE: {meta['style_notes']}")

    return "\n".join(parts)


def build_chapter_prompt(config: dict, chapter_info: dict, chapter_num: int,
                          previous_chapters: list) -> tuple:
    """Build system + user prompts for writing a chapter."""
    story_context = build_story_context(config)

    # Get the POV character for this chapter
    pov_char_name = chapter_info.get("pov_character", "")
    pov_char = None
    for c in config.get("characters", []):
        if c.get("name", "").lower() == pov_char_name.lower():
            pov_char = c
            break

    system = f"""You are a masterful fiction writer. You are writing a novel.

{story_context}

WRITING RULES:
- Write in {config.get('pov', 'third-person limited')} point of view
- Show, don't tell. Use vivid sensory details.
- Dialogue should reflect each character's unique voice.
- Vary sentence length and structure for rhythm.
- End the chapter at a natural dramatic beat — do not summarize or wrap up neatly.
- Do NOT include chapter headers, titles, or "Chapter N" markers.
- Do NOT add author notes, meta-commentary, or break the fourth wall.
- Write ONLY the prose content of the chapter.
- Target approximately {config.get('meta', {}).get('target_words_per_chapter', CHAPTER_TARGET_WORDS)} words."""

    # Build the user prompt with context from previous chapters
    user_parts = []

    # Summary of story so far
    if previous_chapters:
        user_parts.append("STORY SO FAR:")
        for i, prev in enumerate(previous_chapters):
            # Use last ~500 chars of each previous chapter as context
            snippet = prev[-500:] if len(prev) > 500 else prev
            user_parts.append(f"\n[Chapter {i+1} ending]: ...{snippet}")

    # Current chapter instructions
    user_parts.append(f"\nNow write Chapter {chapter_num}: \"{chapter_info.get('title', '')}\"")
    user_parts.append(f"Synopsis: {chapter_info.get('synopsis', chapter_info.get('summary', 'Continue the story.'))}")

    if pov_char:
        user_parts.append(f"POV character: {pov_char['name']}")
        user_parts.append(f"Their current emotional state should reflect: {chapter_info.get('emotional_beat', 'continuation')}")

    if chapter_info.get("location"):
        user_parts.append(f"Location: {chapter_info['location']}")

    # Inject key_events from the chapter entry itself
    for ke in chapter_info.get("key_events", []):
        if isinstance(ke, str):
            user_parts.append(f"Key event: {ke}")
        elif isinstance(ke, dict):
            user_parts.append(f"Key event: {ke.get('summary', ke.get('description', str(ke)))}")

    # Find relevant events from top-level events list
    for event in config.get("events", []):
        if isinstance(event, dict) and event.get("chapter") == chapter_num:
            user_parts.append(f"Key event: {event.get('summary', '')}")
            if event.get("emotional_beat"):
                user_parts.append(f"Emotional beat: {event['emotional_beat']}")

    user_parts.append("\nWrite the full chapter now.")

    return system, "\n".join(user_parts)


def write_chapter_with_continuation(config: dict, chapter_info: dict, chapter_num: int,
                                      previous_chapters: list, host: str, port: int,
                                      **kwargs) -> str:
    """Write a chapter, using continuation if it gets cut off."""
    raw_target = config.get("meta", {}).get("target_words_per_chapter", CHAPTER_TARGET_WORDS)
    # Handle string ranges like "5000-7500" — take the midpoint
    if isinstance(raw_target, str):
        nums = re.findall(r'\d+', raw_target)
        target_words = sum(int(n) for n in nums) // len(nums) if nums else CHAPTER_TARGET_WORDS
    else:
        target_words = int(raw_target)
    max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

    system, user = build_chapter_prompt(config, chapter_info, chapter_num, previous_chapters)

    full_text = ""
    continuation_count = 0
    max_continuations = 4

    # Retry logic for thinking-only responses (model returns only <think> tags)
    max_empty_retries = 2
    for empty_retry in range(max_empty_retries + 1):
        if continuation_count == 0:
            retry_system = system
            if empty_retry > 0:
                retry_system += "\n\nIMPORTANT: Do NOT use <think>, </think>, or any reasoning/thinking tags. Write the chapter prose DIRECTLY with no preamble."
                print(f"     ⚠ Empty response detected — retrying without thinking tags (attempt {empty_retry + 1})...")

            response = llm_generate(
                retry_system, user,
                host=host, port=port,
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", DEFAULT_TEMP),
                top_p=kwargs.get("top_p", DEFAULT_TOP_P),
                min_p=kwargs.get("min_p", DEFAULT_MIN_P),
                repeat_penalty=kwargs.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
            )

            if response.strip():
                full_text = response
                break
            else:
                if empty_retry < max_empty_retries:
                    continue
                else:
                    print(f"     ⚠ Chapter {chapter_num} returned empty after {max_empty_retries + 1} attempts")
                    print(f"       Hint: The model may be exhausting its token budget on internal reasoning.")
                    print(f"       Try: --max-tokens 16384")
                    return ""

    word_count = len(full_text.split())
    print(f"     → {word_count} words", end="")

    if word_count >= target_words * 0.8:
        print(" ✓")
        return full_text.strip()

    # Continuation loop for short chapters
    while continuation_count < max_continuations:
        continuation_count += 1
        print(f" (continuing {continuation_count}/{max_continuations})...")

        cont_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": full_text[-CONTINUATION_OVERLAP:]},
            {"role": "user", "content": "Continue writing from exactly where you left off. Do not repeat any text. Do not add chapter headers or meta-commentary."},
        ]
        response = llm_chat(
            cont_messages,
            host=host, port=port,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", DEFAULT_TEMP),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            min_p=kwargs.get("min_p", DEFAULT_MIN_P),
            repeat_penalty=kwargs.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
        )

        if not response.strip():
            break

        full_text += response
        word_count = len(full_text.split())
        print(f"     → {word_count} words", end="")

        if word_count >= target_words * 0.8:
            print(" ✓")
            break

    return full_text.strip()


# ─── CHARACTER PERSPECTIVE ──────────────────────────────────────────────

def character_think(character: dict, situation: str, config: dict,
                     host: str, port: int, **kwargs) -> str:
    """Simulate a character's internal monologue about a situation."""
    system = f"""You ARE {character['name']}. Think and respond as this character would.

Your psychology: {character.get('psychological', character.get('personality', 'unknown'))}
Your backstory: {character.get('backstory', 'unknown')}
Your voice pattern: {character.get('voice', 'neutral')}
Your current arc: {character.get('arc', 'unchanged')}

Relationships:
{chr(10).join(f"- {r.get('target', '?')}: {r.get('nature', '?')}" for r in character.get('relationships', []))}

Respond in first person as {character['name']}. Be authentic to who you are — your flaws, desires, fears. Don't be noble unless that's genuinely who you are."""

    response = llm_generate(
        system,
        f"The situation: {situation}\n\nWhat are you thinking? How do you feel? What will you do?",
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", 1024),
        temperature=kwargs.get("temperature", 0.95),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.15),
    )
    return response


# ─── REVIEW pass ────────────────────────────────────────────────────────

# ─── AUDIT ──────────────────────────────────────────────────────────────

AUDIT_SYSTEM = """You are an expert fiction editor and story consultant reviewing a book's configuration (outline, characters, chapters) BEFORE prose is written.

Evaluate the following aspects and provide specific, actionable feedback:

1. **Plot Structure** — Does the story have a clear inciting incident, rising action, climax, and resolution? Are there stakes?
2. **Character Arcs** — Does each major character change or grow? Are motivations clear?
3. **Pacing** — Are there too many action/quiet chapters in a row? Does the emotional arc vary?
4. **Continuity** — Do chapter events flow logically? Are there plot holes or impossible timelines?
5. **Foreshadowing** — Are major reveals properly set up in earlier chapters?
6. **Underdeveloped Areas** — What needs more detail? Missing scenes? Thin character motivations?

End your review with exactly this format on its own line:
SCORE: XX/100

Where XX reflects overall story readiness (below 60 = major issues, 60-79 = workable but needs improvement, 80+ = ready to compose)."""


def audit_config(config: dict, host: str, port: int, **kwargs) -> str:
    """Run LLM plot critique on config."""
    import yaml as _yaml
    config_text = _yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    user = f"""Review this book configuration and provide detailed feedback on the story structure, character arcs, pacing, and areas that need improvement.

--- BOOK CONFIG ---
{config_text}
--- END CONFIG ---

Provide your editorial assessment."""

    return llm_generate(
        AUDIT_SYSTEM,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )


def parse_score(text: str) -> int:
    """Extract SCORE: XX/100 from review text. Returns -1 if not found."""
    import re
    m = re.search(r'SCORE:\s*(\d+)\s*/\s*100', text)
    return int(m.group(1)) if m else -1


def deterministic_audit(config: dict) -> list:
    """Run deterministic validation checks on config. Returns list of issue strings."""
    issues = []
    chapters = get_chapters_list(config)
    characters = config.get("characters", [])

    # Required fields
    if not config.get("title"):
        issues.append("❌ Missing: title")
    if not config.get("genre"):
        issues.append("⚠ Missing: genre")
    if not config.get("synopsis"):
        issues.append("⚠ Missing: book-level synopsis")
    if not characters:
        issues.append("❌ Missing: characters list")
    if not chapters:
        issues.append("❌ Missing: chapters list")

    # Character checks
    char_names = set()
    for i, c in enumerate(characters):
        if isinstance(c, dict):
            name = c.get("name", "")
            if not name:
                issues.append(f"⚠ Character {i+1}: missing name")
            else:
                char_names.add(name.lower())
            if not c.get("description"):
                issues.append(f"⚠ Character '{name}': missing description")
        else:
            issues.append(f"⚠ Character {i+1}: expected dict, got {type(c).__name__}")

    # Chapter checks
    for i, ch in enumerate(chapters):
        num = i + 1
        if not isinstance(ch, dict):
            issues.append(f"⚠ Chapter {num}: expected dict, got {type(ch).__name__}")
            continue
        if not ch.get("title"):
            issues.append(f"⚠ Chapter {num}: missing title")
        if not ch.get("synopsis") and not ch.get("summary"):
            issues.append(f"⚠ Chapter {num}: missing synopsis (compose won't have guidance)")
        pov = ch.get("pov_character", "")
        if pov and pov.lower() not in char_names:
            issues.append(f"⚠ Chapter {num}: pov_character '{pov}' not found in characters list")
        if not ch.get("key_events"):
            issues.append(f"💡 Chapter {num}: no key_events (compose will have less plot guidance)")

    return issues


def fix_config(config: dict) -> tuple:
    """Auto-fix structural issues in config. Returns (fixes_applied: int, config: dict)."""
    fixes = 0

    # Unwrap double-nested chapters
    raw_ch = config.get("chapters", [])
    if isinstance(raw_ch, dict) and "chapters" in raw_ch:
        config["chapters"] = raw_ch["chapters"]
        fixes += 1
        print("   🔧 Fixed: unwrapped double-nested chapters")

    chapters = config.get("chapters", [])

    # Convert string chapters to dicts
    if isinstance(chapters, list):
        for i, ch in enumerate(chapters):
            if isinstance(ch, str):
                chapters[i] = {
                    "title": f"Chapter {i+1}",
                    "synopsis": ch,
                    "key_events": [],
                }
                fixes += 1
                print(f"   🔧 Fixed: chapter {i+1} converted from string to dict")
            elif isinstance(ch, dict):
                # Normalize summary → synopsis
                if "summary" in ch and "synopsis" not in ch:
                    ch["synopsis"] = ch.pop("summary")
                    fixes += 1
                    print(f"   🔧 Fixed: chapter {i+1} 'summary' renamed to 'synopsis'")
                # Add key_events stub
                if "key_events" not in ch:
                    ch["key_events"] = []
                    fixes += 1
                    print(f"   🔧 Fixed: chapter {i+1} added empty key_events")
        config["chapters"] = chapters

    # Convert dict chapters to list
    elif isinstance(chapters, dict):
        chapter_list = []
        for key, val in chapters.items():
            if isinstance(val, dict):
                if "title" not in val:
                    val["title"] = str(key)
                chapter_list.append(val)
            else:
                chapter_list.append({"title": str(key), "synopsis": str(val), "key_events": []})
            fixes += 1
        config["chapters"] = chapter_list
        print(f"   🔧 Fixed: converted {len(chapter_list)} dict chapters to list")

    # Remove redundant top-level events (duplicates chapter-level data)
    if "events" in config and "chapters" in config:
        chapters = config.get("chapters", [])
        has_chapter_detail = any(
            isinstance(ch, dict) and (ch.get("key_events") or ch.get("synopsis"))
            for ch in chapters
        )
        if has_chapter_detail:
            del config["events"]
            fixes += 1
            print("   🔧 Fixed: removed redundant top-level 'events' (data already in chapters)")

    return fixes, config


# ─── REVIEW pass ────────────────────────────────────────────────────────

REVIEW_SYSTEM = """You are a sharp-eyed fiction editor. Review the chapter for:
1. Repetitive phrases or loops (especially toward the end)
2. Character voice consistency
3. Show vs tell violations
4. Pacing issues
5. Continuity errors with the story config

Provide specific, actionable feedback. Then write a REVISED version of any weak sections.
If the chapter is solid, say "APPROVED" and note what works well.

End your review with exactly this format on its own line:
SCORE: XX/100"""


# ─── REVISION STRATEGIES ───────────────────────────────────────────────

REVISION_STRATEGIES = {
    "general": """Improve everything — prose quality, pacing, dialogue, description, voice.
Fix all issues flagged in the review. Preserve what works, rewrite what doesn't.""",

    "weakest": """Identify the SINGLE WEAKEST ELEMENT in this chapter (could be dialogue, pacing,
description, voice, or plot) and focus your revision ONLY on that element.
Do NOT change anything else — preserve all other prose exactly as written.""",

    "dialogue": """Focus ONLY on improving dialogue. Make each character sound distinct and natural.
Remove any dialogue that sounds generic or AI-generated. Add subtext — characters should
sometimes say one thing and mean another. Preserve all non-dialogue prose exactly.""",

    "sensory": """Focus ONLY on adding sensory immersion. What does the scene smell like? What sounds
are in the background? What textures does the POV character notice? Weave these details
into existing prose naturally — do NOT add separate descriptive paragraphs.""",

    "pacing": """Focus ONLY on pacing. Cut sentences and passages that drag or repeat beats already established.
Expand moments that feel rushed — especially emotional turning points and revelations.
Scene transitions should feel natural, not abrupt.""",

    "voice": """Focus ONLY on the POV character's internal voice and perspective. Strengthen their
unique way of seeing the world. Add their reactions, judgments, and emotional responses
to what's happening. Make the reader feel like they're inside this specific person's head.""",
}


def review_chapter(chapter_text: str, chapter_info, config: dict,
                    host: str, port: int, **kwargs) -> str:
    """Self-review and optionally revise a chapter."""
    story_context = build_story_context(config)

    user = f"""Story context:
{story_context}

Chapter {ch_get(chapter_info, 'number', '?')}: "{ch_get(chapter_info, 'title', '')}"
Synopsis: {ch_get(chapter_info, 'synopsis', '') or ch_get(chapter_info, 'summary', '')}

--- CHAPTER TEXT ---
{chapter_text}
--- END ---

Review this chapter and provide feedback. If sections need revision, write the improved version."""

    response = llm_generate(
        REVIEW_SYSTEM,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )
    return response


# ─── REVISE pass ────────────────────────────────────────────────────────

REVISE_SYSTEM_BASE = """You are a masterful fiction writer revising your own work based on editorial feedback.

RULES:
- Preserve everything that works well — do not rewrite sections the review praised.
- Fix ONLY the specific issues flagged in the review.
- If the review flags repetitive/run-on sections, rewrite those passages with fresh prose.
- Maintain the same POV, tense, tone, and character voices.
- Keep approximately the same word count.
- Output ONLY the revised chapter text — no commentary, no headers, no editorial notes.
- Do NOT include "Chapter N" or any title markers."""


def revise_chapter(chapter_text: str, review_text: str, chapter_info,
                    config: dict, host: str, port: int,
                    strategy: str = "general", **kwargs) -> str:
    """Revise a chapter based on review feedback."""
    story_context = build_story_context(config)

    # Build system prompt with strategy
    strategy_text = REVISION_STRATEGIES.get(strategy, REVISION_STRATEGIES["general"])
    system = f"{REVISE_SYSTEM_BASE}\n\nREVISION FOCUS:\n{strategy_text}"

    user = f"""STORY CONTEXT:
{story_context}

CHAPTER {ch_get(chapter_info, 'number', '?')}: "{ch_get(chapter_info, 'title', '')}"
Synopsis: {ch_get(chapter_info, 'synopsis', '') or ch_get(chapter_info, 'summary', '')}

--- ORIGINAL CHAPTER ---
{chapter_text}
--- END ORIGINAL ---

--- EDITORIAL REVIEW ---
{review_text}
--- END REVIEW ---

Revise the chapter incorporating the editorial feedback. Keep what works, fix what doesn't. Write the complete revised chapter."""

    response = llm_generate(
        system,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.15),
    )
    return response


# ─── file I/O ───────────────────────────────────────────────────────────

def save_config(config: dict, path: str):
    """Save config as YAML with human-friendly chapter field ordering."""
    # Enforce readable field order for chapters
    CHAPTER_FIELD_ORDER = [
        "number", "title", "synopsis", "pov_character",
        "emotional_beat", "location", "key_events",
    ]
    if "chapters" in config:
        ordered_chapters = []
        for ch in config["chapters"]:
            if not isinstance(ch, dict):
                ordered_chapters.append(ch)
                continue
            ordered = {}
            for key in CHAPTER_FIELD_ORDER:
                if key in ch:
                    ordered[key] = ch[key]
            # Append any extra fields not in the canonical order
            for key, val in ch.items():
                if key not in ordered:
                    ordered[key] = val
            ordered_chapters.append(ordered)
        config = dict(config)
        config["chapters"] = ordered_chapters

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=120, sort_keys=False)
    print(f"   💾 Config saved: {path}")


def load_config(path: str) -> dict:
    """Load config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_chapter(text: str, chapter_num: int, output_dir: str):
    """Save a chapter to file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"chapter_{chapter_num:02d}.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"   💾 Chapter {chapter_num} saved: {path}")
    return path


def save_book(chapters: list, config: dict, output_dir: str):
    """Compile all chapters into a single book file."""
    os.makedirs(output_dir, exist_ok=True)
    title = config.get("title", "Untitled")
    path = os.path.join(output_dir, "book.md")

    with open(path, "w") as f:
        f.write(f"# {title}\n\n")

        chapter_infos = get_chapters_list(config)
        for i, chapter_text in enumerate(chapters):
            ch_num = i + 1
            ch_title = ""
            if i < len(chapter_infos):
                ci = chapter_infos[i]
                ch_title = ci.get("title", f"Chapter {ch_num}") if isinstance(ci, dict) else f"Chapter {ch_num}"
            else:
                ch_title = f"Chapter {ch_num}"

            f.write(f"## Chapter {ch_num}: {ch_title}\n\n")
            f.write(chapter_text)
            f.write("\n\n---\n\n")

    total_words = sum(len(c.split()) for c in chapters)
    print(f"\n   📚 Book compiled: {path}")
    print(f"   📊 Total: {len(chapters)} chapters, {total_words:,} words")
    return path


# ─── CLI commands ───────────────────────────────────────────────────────

def cmd_genesis(args):
    """Generate a story config from a prompt."""
    config = genesis(
        args.prompt, args.host, args.port,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if "_parse_error" in config:
        # Save raw response
        raw_path = args.output.replace(".yaml", "_raw.txt")
        with open(raw_path, "w") as f:
            f.write(config["_raw_response"])
        print(f"   ⚠ Raw response saved to {raw_path} for manual fixing")
        return

    # Store origin prompt and generation metadata
    from datetime import datetime, timezone
    meta = config.setdefault("meta", {})
    meta["origin_prompt"] = args.prompt
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()

    save_config(config, args.output)
    print(f"\n   ✅ Genesis complete! Edit {args.output} to refine, then run:")
    print(f"      python3 ghostwriter.py compose {args.output}")


def cmd_compose(args):
    """Write the book from a config file."""
    config = load_config(args.config)

    # Flatten: if metadata is a sub-dict, merge its keys to top level
    if "metadata" in config and isinstance(config["metadata"], dict):
        for k, v in config["metadata"].items():
            if k not in config:
                config[k] = v

    title = config.get("title", "Untitled")
    chapters_info = config.get("chapters", [])

    # Unwrap double-nesting: chapters: { chapters: [...] }
    if isinstance(chapters_info, dict) and "chapters" in chapters_info:
        chapters_info = chapters_info["chapters"]

    # Detect act-based nesting: look for act_1/act_2/etc with sub-chapters
    if not chapters_info:
        acts = []
        for key in sorted(config.keys()):
            val = config.get(key)
            if isinstance(val, dict) and "chapters" in val:
                act_chapters = val["chapters"]
                if isinstance(act_chapters, list):
                    acts.extend(act_chapters)
        if acts:
            chapters_info = acts
            print(f"   ℹ Found {len(acts)} chapters across act-based structure")

    # Normalize: if chapters is a dict (keyed by name), convert to list
    if isinstance(chapters_info, dict):
        normalized = []
        for key, val in chapters_info.items():
            entry = val if isinstance(val, dict) else {"summary": str(val)}
            if "title" not in entry:
                entry["title"] = str(key).replace("_", " ").title()
            normalized.append(entry)
        chapters_info = normalized

    # Normalize list entries: handle [{chapter_1: "summary"}, ...] format
    if chapters_info and isinstance(chapters_info, list):
        normalized = []
        for idx, item in enumerate(chapters_info):
            if isinstance(item, dict):
                # Single-key dict like {chapter_1: "Opening scene..."}
                if len(item) == 1 and not any(k in item for k in ("title", "summary", "number")):
                    key, val = next(iter(item.items()))
                    entry = {"title": str(key).replace("_", " ").title(), "summary": str(val)}
                    normalized.append(entry)
                else:
                    normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"title": f"Chapter {idx+1}", "summary": item})
            else:
                normalized.append({"title": f"Chapter {idx+1}", "summary": str(item)})
        chapters_info = normalized

    if not chapters_info:
        print("ERROR: No chapters defined in config. Run genesis first.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📖 COMPOSING: {title}")
    print(f"   Chapters: {len(chapters_info)}")
    print(f"   Target words/chapter: {config.get('meta', {}).get('target_words_per_chapter', CHAPTER_TARGET_WORDS)}")
    print()

    written_chapters = [None] * len(chapters_info)

    # Scan for existing chapters — track which are present
    existing_count = 0
    for i in range(len(chapters_info)):
        ch_path = os.path.join(output_dir, f"chapter_{i+1:02d}.md")
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                written_chapters[i] = f.read()
            existing_count += 1
            print(f"   📄 Found existing chapter {i+1}, keeping...")

    if existing_count:
        missing = [i+1 for i in range(len(chapters_info)) if written_chapters[i] is None]
        print(f"   📝 Missing chapters to write: {missing if missing else 'none'}")

    # If --chapter was specified, only write that chapter
    chapter_range = range(len(chapters_info))
    if hasattr(args, 'chapter') and args.chapter:
        idx = args.chapter - 1
        if 0 <= idx < len(chapters_info):
            chapter_range = [idx]
            # Force re-generation even if it exists
            written_chapters[idx] = None
        else:
            print(f"ERROR: Chapter {args.chapter} out of range (1-{len(chapters_info)})")
            sys.exit(1)

    for i in chapter_range:
        if written_chapters[i] is not None:
            continue  # already exists — skip

        ch_info = chapters_info[i]
        ch_num = i + 1
        ch_title = ch_info.get("title", f"Chapter {ch_num}")

        print(f"\n✍  Chapter {ch_num}/{len(chapters_info)}: \"{ch_title}\"")

        # Optional: character perspective pre-writing
        if args.character_think and ch_info.get("pov_character"):
            pov_name = ch_info["pov_character"]
            pov_char = None
            for c in config.get("characters", []):
                if c.get("name", "").lower() == pov_name.lower():
                    pov_char = c
                    break

            if pov_char:
                synopsis = ch_info.get("synopsis", "the next scene")
                print(f"   🧠 {pov_name} is thinking about: {synopsis[:60]}...")
                thought = character_think(
                    pov_char, synopsis, config,
                    host=args.host, port=args.port,
                    max_tokens=512,
                )
                # We don't use the thought directly but it primes the context
                print(f"   💭 \"{thought[:100]}...\"")

        # Write the chapter
        # Pass only existing chapters as context (filter out None gaps)
        prev_chapters = [c for c in written_chapters[:i] if c is not None]
        chapter_text = write_chapter_with_continuation(
            config, ch_info, ch_num, prev_chapters,
            host=args.host, port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Optional: review pass
        if args.review:
            print(f"   📝 Reviewing chapter {ch_num}...")
            review = review_chapter(
                chapter_text, ch_info, config,
                host=args.host, port=args.port,
                max_tokens=args.max_tokens,
            )
            review_path = os.path.join(output_dir, f"review_{ch_num:02d}.md")
            with open(review_path, "w") as f:
                f.write(review)
            print(f"   📝 Review saved: {review_path}")

        save_chapter(chapter_text, ch_num, output_dir)
        written_chapters[i] = chapter_text

    # Compile book (filter out any remaining None gaps)
    complete_chapters = [c for c in written_chapters if c is not None]
    save_book(complete_chapters, config, output_dir)


def cmd_write(args):
    """Full pipeline: genesis → compose."""
    # Genesis
    config_path = args.output + ".yaml" if args.output else "story_config.yaml"
    config = genesis(
        args.prompt, args.host, args.port,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if "_parse_error" in config:
        raw_path = config_path.replace(".yaml", "_raw.txt")
        with open(raw_path, "w") as f:
            f.write(config["_raw_response"])
        print(f"   ⚠ YAML parse failed. Raw response saved to {raw_path}")
        print(f"   Fix the YAML and run: python3 ghostwriter.py compose {config_path}")
        return

    save_config(config, config_path)

    # Compose
    title = config.get("title", "untitled")
    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")

    # Reuse args for compose
    args.config = config_path
    args.output = output_dir
    cmd_compose(args)


def cmd_review(args):
    """Review all chapters of an existing book."""
    config = load_config(args.config)
    chapters_info = get_chapters_list(config)
    title = config.get("title", "Untitled")

    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")

    print(f"\n📝 REVIEWING: {title}")

    # If --chapter was specified, only review that chapter
    if hasattr(args, 'chapter') and args.chapter:
        review_indices = [args.chapter - 1]
    else:
        review_indices = range(len(chapters_info))

    for i in review_indices:
        if i < 0 or i >= len(chapters_info):
            print(f"   ⚠ Chapter {i+1} out of range")
            continue
        ch_info = chapters_info[i]
        ch_num = i + 1
        ch_path = os.path.join(output_dir, f"chapter_{ch_num:02d}.md")

        if not os.path.exists(ch_path):
            print(f"   ⚠ Chapter {ch_num} not found: {ch_path}")
            continue

        # Stale review check: skip if review is newer than chapter
        review_path = os.path.join(output_dir, f"review_{ch_num:02d}.md")
        forced = hasattr(args, 'chapter') and args.chapter
        if os.path.exists(review_path) and not forced:
            ch_mtime = os.path.getmtime(ch_path)
            rv_mtime = os.path.getmtime(review_path)
            if rv_mtime >= ch_mtime:
                print(f"   ✓ Chapter {ch_num} review is fresh, skipping")
                continue
            else:
                print(f"   ♻ Chapter {ch_num} updated since last review, re-reviewing...")

        with open(ch_path) as f:
            chapter_text = f.read()

        ch_title = ch_info.get('title', '') if isinstance(ch_info, dict) else f'Chapter {ch_num}'
        print(f"\n📝 Reviewing chapter {ch_num}: \"{ch_title}\"")
        review = review_chapter(
            chapter_text, ch_info, config,
            host=args.host, port=args.port,
            max_tokens=args.max_tokens,
        )

        with open(review_path, "w") as f:
            f.write(review)
        print(f"   💾 Review saved: {review_path}")


def cmd_revise(args):
    """Revise chapters based on review feedback."""
    config = load_config(args.config)
    chapters_info = get_chapters_list(config)
    title = config.get("title", "Untitled")

    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")

    # Determine which chapters to revise
    if args.chapters:
        chapter_nums = [int(c) for c in args.chapters]
    else:
        chapter_nums = list(range(1, len(chapters_info) + 1))

    print(f"\n🔄 REVISING: {title}")
    print(f"   Chapters: {chapter_nums}")

    for ch_num in chapter_nums:
        if ch_num < 1 or ch_num > len(chapters_info):
            print(f"   ⚠ Chapter {ch_num} out of range, skipping")
            continue

        ch_info = chapters_info[ch_num - 1]
        ch_path = os.path.join(output_dir, f"chapter_{ch_num:02d}.md")
        review_path = os.path.join(output_dir, f"review_{ch_num:02d}.md")

        if not os.path.exists(ch_path):
            print(f"   ⚠ Chapter {ch_num} not found: {ch_path}")
            continue

        if not os.path.exists(review_path):
            print(f"   ⚠ Review for chapter {ch_num} not found: {review_path}")
            print(f"      Run: ghostwriter.py review {args.config}")
            continue

        with open(ch_path) as f:
            chapter_text = f.read()
        with open(review_path) as f:
            review_text = f.read()

        ch_title = ch_info.get('title', '') if isinstance(ch_info, dict) else f'Chapter {ch_num}'
        print(f"\n🔄 Revising chapter {ch_num}: \"{ch_title}\"")
        print(f"   Original: {len(chapter_text.split())} words")

        # Archive original
        version = 1
        while os.path.exists(os.path.join(output_dir, f"chapter_{ch_num:02d}_v{version}.md")):
            version += 1
        archive_path = os.path.join(output_dir, f"chapter_{ch_num:02d}_v{version}.md")
        with open(archive_path, "w") as f:
            f.write(chapter_text)
        print(f"   📦 Archived original as v{version}")

        # Revise
        revised = revise_chapter(
            chapter_text, review_text, ch_info, config,
            host=args.host, port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if revised.strip():
            with open(ch_path, "w") as f:
                f.write(revised)
            print(f"   ✅ Revised: {len(revised.split())} words → {ch_path}")
        else:
            print(f"   ⚠ Revision produced empty output, keeping original")

    # Re-compile book
    all_chapters = []
    for i in range(len(chapters_info)):
        ch_path = os.path.join(output_dir, f"chapter_{i+1:02d}.md")
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                all_chapters.append(f.read())
    if all_chapters:
        save_book(all_chapters, config, output_dir)


# ─── AUDIT command ──────────────────────────────────────────────────────

def cmd_audit(args):
    """Validate config and run LLM plot critique."""
    config = load_config(args.config)
    title = config.get("title", "Untitled")
    print(f"\n🔎 Auditing config for: {title}")

    # Phase 0: Auto-fix (if --fix)
    if getattr(args, "fix", False):
        print("\n── Auto-Fix ──")
        num_fixes, config = fix_config(config)
        if num_fixes > 0:
            import yaml as _yaml
            with open(args.config, "w") as f:
                _yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"\n   ✅ Applied {num_fixes} fix(es), saved to {args.config}")
        else:
            print("   No fixable issues found")

    # Phase 1: Deterministic checks
    issues = deterministic_audit(config)
    if issues:
        print(f"\n── Structural Issues ({len(issues)}) ──")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n   ✅ No structural issues found")

    # Phase 2: LLM critique (unless --syntax-only)
    if not getattr(args, "syntax_only", False):
        print("\n── LLM Plot Review ──")
        print("   (sending config to LLM for critique...)\n")
        review = audit_config(
            config, host=args.host, port=args.port,
            max_tokens=getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
            temperature=getattr(args, "temperature", 0.7),
        )
        print(review)

        score = parse_score(review)
        if score >= 0:
            print(f"\n   📊 Config readiness score: {score}/100")
            if score >= 80:
                print("   ✅ Ready to compose!")
            elif score >= 60:
                print("   ⚠ Workable, but consider the suggestions above")
            else:
                print("   ❌ Significant issues — address feedback before composing")

        # Save audit report
        output_dir = getattr(args, "output", None) or title.lower().replace(" ", "_").replace("'", "")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "audit_report.md")
        with open(report_path, "w") as f:
            f.write(f"# Audit Report: {title}\n\n")
            if issues:
                f.write("## Structural Issues\n\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            f.write("## LLM Plot Review\n\n")
            f.write(review)
            if score >= 0:
                f.write(f"\n\n**Score: {score}/100**\n")
        print(f"\n   💾 Report saved: {report_path}")


# ─── POLISH command ─────────────────────────────────────────────────────

def cmd_polish(args):
    """Automated review→revise loop with convergence."""
    config = load_config(args.config)
    chapters_info = get_chapters_list(config)
    title = config.get("title", "Untitled")

    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")
    os.makedirs(output_dir, exist_ok=True)

    min_score = getattr(args, "min_score", 80)
    max_rounds = getattr(args, "max_rounds", 3)
    strategy = getattr(args, "strategy", "general")
    chapter_filter = getattr(args, "chapter", None)

    print(f"\n✨ Polishing: {title}")
    print(f"   Strategy: {strategy} | Min score: {min_score} | Max rounds: {max_rounds}")

    # Determine which chapters to polish
    if chapter_filter:
        chapter_nums = [chapter_filter]
    else:
        chapter_nums = list(range(1, len(chapters_info) + 1))

    for ch_num in chapter_nums:
        ch_path = os.path.join(output_dir, f"chapter_{ch_num:02d}.md")
        if not os.path.exists(ch_path):
            print(f"\n   ⏭ Chapter {ch_num}: no file found, skipping")
            continue

        ch_info = chapters_info[ch_num - 1] if ch_num <= len(chapters_info) else {}
        ch_title = ch_get(ch_info, "title", f"Chapter {ch_num}")
        print(f"\n── Chapter {ch_num}: \"{ch_title}\" ──")

        with open(ch_path) as f:
            chapter_text = f.read()

        prev_score = -1
        for round_num in range(1, max_rounds + 1):
            # Review
            print(f"   Round {round_num}: reviewing...")
            review = review_chapter(
                chapter_text, ch_info, config,
                host=args.host, port=args.port,
                max_tokens=getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
                temperature=getattr(args, "temperature", 0.7),
            )

            score = parse_score(review)
            print(f"   Round {round_num}: score = {score}/100" if score >= 0 else f"   Round {round_num}: no score parsed")

            # Save review
            review_path = os.path.join(output_dir, f"review_{ch_num:02d}.md")
            with open(review_path, "w") as f:
                f.write(review)

            # Check convergence
            if score >= min_score:
                print(f"   ✅ Score {score} ≥ {min_score} — chapter polished!")
                break
            if prev_score >= 0 and score >= 0 and score - prev_score < 2:
                print(f"   ⏹ Score plateau ({prev_score} → {score}) — stopping")
                break
            if round_num >= max_rounds:
                print(f"   ⏹ Max rounds ({max_rounds}) reached — stopping")
                break

            prev_score = score

            # Revise
            print(f"   Round {round_num}: revising with strategy '{strategy}'...")

            # Archive current version
            version = round_num
            archive_path = os.path.join(output_dir, f"chapter_{ch_num:02d}_v{version}.md")
            with open(archive_path, "w") as f:
                f.write(chapter_text)

            revised = revise_chapter(
                chapter_text, review, ch_info, config,
                host=args.host, port=args.port,
                strategy=strategy,
                max_tokens=getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
                temperature=getattr(args, "temperature", 0.8),
            )
            revised = strip_thinking(revised)

            # Save revised
            with open(ch_path, "w") as f:
                f.write(revised)
            chapter_text = revised
            word_count = len(revised.split())
            print(f"   Round {round_num}: revised ({word_count:,} words), saved v{version} backup")

        print(f"   Done: chapter {ch_num}")

    # Recompile book
    all_chapters = []
    for i in range(len(chapters_info)):
        ch_path = os.path.join(output_dir, f"chapter_{i+1:02d}.md")
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                all_chapters.append(f.read())
    if all_chapters:
        save_book(all_chapters, config, output_dir)
    print("\n✨ Polish complete!")


# ─── DISTILL command ────────────────────────────────────────────────────

DISTILL_SYSTEM = """You are an expert story architect. You have just read the complete prose of a novel.

Your task is to extract a REFINED story configuration (YAML) from the actual prose — NOT from the original outline, but from what the story ACTUALLY BECAME.

For each chapter, provide:
- title: the chapter title
- synopsis: a detailed 2-4 sentence synopsis of what ACTUALLY HAPPENS in the prose
- pov_character: the POV character for this chapter
- key_events: a list of 3-6 specific plot events that occur (be concrete, not vague)
- emotional_beat: one word describing the dominant emotion
- location: where the chapter takes place

Also update the top-level fields:
- synopsis: an improved 3-5 sentence book synopsis based on the actual story
- themes: refined theme list based on what actually emerged in the prose
- characters: updated character descriptions reflecting how they actually developed

Preserve the existing structure (title, genre, tone, pov, setting, meta) but refine any field that the prose improved upon.

Output ONLY valid YAML — no commentary, no markdown fences."""


def next_version_path(config_path: str) -> str:
    """Find the next available version path: config.v1.yaml, config.v2.yaml, etc."""
    import re as _re
    base = config_path.rsplit(".", 1)[0]  # Remove .yaml
    # Strip any existing .vN suffix to avoid stacking like .v1.v2
    base = _re.sub(r'\.v\d+$', '', base)
    version = 1
    while True:
        vpath = f"{base}.v{version}.yaml"
        if not os.path.exists(vpath):
            return vpath
        version += 1


def config_diff(old_config: dict, new_config: dict) -> str:
    """Generate a structured changelog between two configs."""
    lines = []

    # Top-level field changes
    for key in ("title", "genre", "tone", "pov"):
        old_val = old_config.get(key, "")
        new_val = new_config.get(key, "")
        if old_val != new_val:
            lines.append(f"  {key}: \"{old_val}\" → \"{new_val}\"")

    # Synopsis
    old_syn = old_config.get("synopsis", "")
    new_syn = new_config.get("synopsis", "")
    if old_syn != new_syn:
        old_words = len(old_syn.split()) if old_syn else 0
        new_words = len(new_syn.split()) if new_syn else 0
        lines.append(f"  synopsis: {old_words} → {new_words} words")

    # Themes
    old_themes = set(old_config.get("themes", []))
    new_themes = set(new_config.get("themes", []))
    added = new_themes - old_themes
    removed = old_themes - new_themes
    if added:
        lines.append(f"  themes added: {', '.join(added)}")
    if removed:
        lines.append(f"  themes removed: {', '.join(removed)}")

    if lines:
        print("\n── Top-Level Changes ──")
        for line in lines:
            print(line)

    # Chapter-by-chapter diff
    old_chapters = get_chapters_list(old_config)
    new_chapters = get_chapters_list(new_config)
    max_ch = max(len(old_chapters), len(new_chapters))

    ch_lines = []
    for i in range(max_ch):
        old_ch = old_chapters[i] if i < len(old_chapters) else {}
        new_ch = new_chapters[i] if i < len(new_chapters) else {}

        if not isinstance(old_ch, dict):
            old_ch = {"synopsis": str(old_ch)}
        if not isinstance(new_ch, dict):
            new_ch = {"synopsis": str(new_ch)}

        ch_title = new_ch.get("title", old_ch.get("title", f"Chapter {i+1}"))
        changes = []

        # Synopsis length
        old_s = old_ch.get("synopsis", old_ch.get("summary", ""))
        new_s = new_ch.get("synopsis", new_ch.get("summary", ""))
        old_sw = len(old_s.split()) if old_s else 0
        new_sw = len(new_s.split()) if new_s else 0
        if old_sw != new_sw:
            pct = ((new_sw - old_sw) / old_sw * 100) if old_sw > 0 else 100
            changes.append(f"synopsis: {old_sw} → {new_sw} words ({pct:+.0f}%)")

        # Key events count
        old_ke = len(old_ch.get("key_events", []))
        new_ke = len(new_ch.get("key_events", []))
        if old_ke != new_ke:
            changes.append(f"key_events: {old_ke} → {new_ke}")

        # POV character
        old_pov = old_ch.get("pov_character", "")
        new_pov = new_ch.get("pov_character", "")
        if old_pov != new_pov:
            changes.append(f"pov: \"{old_pov}\" → \"{new_pov}\"")

        # Emotional beat
        old_eb = old_ch.get("emotional_beat", "")
        new_eb = new_ch.get("emotional_beat", "")
        if old_eb != new_eb:
            changes.append(f"emotional_beat: \"{old_eb}\" → \"{new_eb}\"")

        if changes:
            ch_lines.append(f"  Chapter {i+1}: \"{ch_title}\"")
            for c in changes:
                ch_lines.append(f"    {c}")

    if ch_lines:
        print(f"\n── Chapter Changes ({max_ch} chapters) ──")
        for line in ch_lines:
            print(line)

    if not lines and not ch_lines:
        print("\n   No significant changes detected")


def cmd_distill(args):
    """Extract refined config from existing prose chapters."""
    config = load_config(args.config)
    chapters_info = get_chapters_list(config)
    title = config.get("title", "Untitled")

    output_dir = args.output or title.lower().replace(" ", "_").replace("'", "")

    print(f"\n🧪 Distilling config from prose: {title}")

    # Read all chapter files
    all_prose = []
    for i in range(len(chapters_info)):
        ch_path = os.path.join(output_dir, f"chapter_{i+1:02d}.md")
        if os.path.exists(ch_path):
            with open(ch_path) as f:
                text = f.read()
            all_prose.append(f"--- CHAPTER {i+1} ---\n{text}")
            print(f"   Read: chapter_{i+1:02d}.md ({len(text.split()):,} words)")
        else:
            print(f"   ⏭ chapter_{i+1:02d}.md not found, skipping")

    if not all_prose:
        print("   ❌ No chapter files found. Run compose first.")
        return

    # Build context: include existing config for structure reference
    import yaml as _yaml
    old_config_text = _yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    user = f"""Here is the ORIGINAL config for reference:

--- ORIGINAL CONFIG ---
{old_config_text}
--- END ORIGINAL ---

Here is the COMPLETE PROSE of the novel:

{chr(10).join(all_prose)}

--- END PROSE ---

Extract a refined story configuration from the actual prose. Improve synopses, add key_events from what actually happens, and update character descriptions to reflect their development in the prose. Output valid YAML only."""

    print(f"\n   Sending {len(all_prose)} chapters to LLM for distillation...")

    response = llm_generate(
        DISTILL_SYSTEM,
        user,
        host=args.host, port=args.port,
        max_tokens=getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
        temperature=getattr(args, "temperature", 0.6),
        top_p=getattr(args, "top_p", DEFAULT_TOP_P),
        min_p=getattr(args, "min_p", DEFAULT_MIN_P),
        repeat_penalty=getattr(args, "repeat_penalty", 1.1),
    )

    # Parse the new config
    import re
    cleaned = response.strip()
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'^```ya?ml\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    new_config = None
    for attempt, text in enumerate([cleaned, _yaml_repair(cleaned)]):
        try:
            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict):
                raise ValueError("YAML did not parse to a dictionary")
            new_config = parsed
            if attempt > 0:
                print("   🔧 YAML repaired successfully")
            break
        except Exception as e:
            if attempt == 0:
                print(f"   ⚠ Initial YAML parse failed, attempting repair: {e}")
            else:
                print(f"   ⚠ Failed to parse distilled config after repair: {e}")
                raw_path = os.path.join(output_dir, "distill_raw.txt")
                with open(raw_path, "w") as f:
                    f.write(response)
                print(f"   Raw response saved: {raw_path}")
                return

    if new_config is None:
        return

    new_config = _fix_key_events_dicts(new_config)

    # Preserve origin_prompt from old config
    old_meta = config.get("meta", {})
    new_meta = new_config.setdefault("meta", {})
    if "origin_prompt" in old_meta:
        new_meta["origin_prompt"] = old_meta["origin_prompt"]
    from datetime import datetime, timezone
    new_meta["distilled_at"] = datetime.now(timezone.utc).isoformat()
    new_meta["distilled_from"] = os.path.basename(args.config)

    # Structured diff
    config_diff(config, new_config)

    # Version the old config
    version_path = next_version_path(args.config)
    import shutil
    shutil.copy2(args.config, version_path)
    print(f"\n   📦 Old config backed up: {version_path}")

    # Save new config
    save_config(new_config, args.config)
    print(f"   ✅ Distilled config saved: {args.config}")
    print(f"\n   Next steps:")
    print(f"      Review the config, then re-compose with:")
    print(f"      python3 ghostwriter.py compose {args.config}")


# ─── EVOLVE command ─────────────────────────────────────────────────────

EVOLVE_MODES = {
    "expand": """You are an expert story architect EXPANDING a book outline.

RULES — READ CAREFULLY:
- You may ONLY ADD new content. Never modify or remove existing text.
- You may add new chapters (insert between existing ones or append)
- You may add new key_events to existing chapters
- You may add new characters
- You may add detail to sparse synopses by APPENDING to them (keep original text)
- You may add emotional_beat, location, pov_character if missing
- NEVER change existing titles, synopses, character descriptions, or key_events
- Renumber chapters sequentially after insertions

Output the COMPLETE config as valid YAML — no commentary, no markdown fences.""",

    "refine": """You are an expert story architect REFINING a book outline.

RULES — READ CAREFULLY:
- You may REWRITE any existing field to improve quality
- Sharpen synopses to be more specific and compelling
- Strengthen character motivations and arcs
- Improve emotional beats for better pacing variety
- Fix pacing: ensure action/quiet chapters alternate
- Add foreshadowing connections between chapters
- DO NOT add or remove chapters — keep the exact same chapter count
- DO NOT add or remove characters

Output the COMPLETE config as valid YAML — no commentary, no markdown fences.""",

    "full": """You are an expert story architect performing a FULL EVOLUTION of a book outline.

You may:
- Add new chapters, characters, key_events
- Rewrite and strengthen existing synopses, character descriptions, emotional beats
- Improve pacing by reordering or splitting chapters
- Add foreshadowing and thematic connections
- Fill gaps in the narrative
- Strengthen character arcs with more specific development markers

Your goal is to produce the BEST POSSIBLE story configuration.
Renumber chapters sequentially.

Output the COMPLETE config as valid YAML — no commentary, no markdown fences.""",
}

# Mode-specific analysis prompts (Phase 1: produce prose, NOT YAML)
ANALYSIS_PROMPTS = {
    "expand": """You are a senior acquisitions editor evaluating a book outline for EXPANSION opportunities.

Analyze the book and produce a strategic editorial brief covering:
1. THE ELEVATOR PITCH: 1-2 sentences that would sell this book to a publisher
2. WHAT'S WORKING: The strongest elements of the current outline
3. GAPS & OPPORTUNITIES: What's missing? Where could new chapters, events, or characters fill narrative holes?
4. PER-CHAPTER NOTES: For each chapter (by number), note what could be ADDED — new key_events, richer details, missing emotional beats. Do NOT suggest removing or changing existing content.
5. CHARACTER GAPS: Any characters who need more development or new characters who should be introduced?

Remember: EXPAND mode is ADDITIVE ONLY. Never suggest removing or rewriting existing content.
Output ONLY your editorial brief as prose — no YAML, no code fences.""",

    "refine": """You are a senior acquisitions editor evaluating a book outline for QUALITY refinement.

Analyze the book and produce a strategic editorial brief covering:
1. THE ELEVATOR PITCH: 1-2 sentences that would sell this book to a publisher
2. WHAT'S WORKING: The strongest elements of the current outline
3. WEAKNESSES: What's the weakest aspect? Pacing issues? Flat characters? Unclear motivations?
4. PER-CHAPTER NOTES: For each chapter (by number), note what could be IMPROVED — sharper synopses, stronger emotional beats, better foreshadowing, tighter pacing. Same chapter count, same characters.
5. ARC ASSESSMENT: How well do the character arcs develop across chapters? What needs strengthening?

Remember: REFINE mode keeps the same structure. No adding/removing chapters or characters.
Output ONLY your editorial brief as prose — no YAML, no code fences.""",

    "full": """You are a senior acquisitions editor performing a comprehensive evaluation of a book outline.

Analyze the book and produce a strategic editorial brief covering:
1. THE ELEVATOR PITCH: 1-2 sentences that would sell this book to a publisher
2. WHAT'S WORKING: The strongest elements of the current outline
3. BIGGEST WEAKNESSES: What would make you reject this manuscript? What needs the most work?
4. STRUCTURAL SUGGESTIONS: Should chapters be reordered, split, or merged? Are there pacing problems?
5. PER-CHAPTER NOTES: For each chapter (by number), suggest specific improvements — rewriting, expanding, adding foreshadowing, changing emotional beats, etc.
6. CHARACTER & WORLD NOTES: Missing characters? Underdeveloped world elements? Thematic gaps?

Remember: FULL mode is unrestricted. You may suggest any changes that improve the book.
Output ONLY your editorial brief as prose — no YAML, no code fences.""",
}

# Mode-specific chapter evolution prompts
CHAPTER_EVOLVE_PROMPTS = {
    "expand": """You are evolving Chapter {num} ("{title}") of "{book_title}" in EXPAND mode.

RULES:
- You may ONLY ADD new content to this chapter. Keep ALL existing fields unchanged.
- You may add new key_events (append to the list)
- You may add emotional_beat, location, pov_character if missing
- You may APPEND to the synopsis (keep existing text, add more detail)
- NEVER change existing text in any field
- Use double quotes for any string values containing apostrophes or colons""",

    "refine": """You are evolving Chapter {num} ("{title}") of "{book_title}" in REFINE mode.

RULES:
- You may REWRITE any field to improve quality
- Sharpen the synopsis to be more specific and compelling
- Strengthen emotional_beat for better pacing variety
- Improve key_events to be more specific and dramatic
- Ensure pov_character and location are set
- Use double quotes for any string values containing apostrophes or colons""",

    "full": """You are evolving Chapter {num} ("{title}") of "{book_title}" in FULL mode.

RULES:
- You may rewrite any field to maximize quality
- Make the synopsis specific, compelling, and evocative
- Ensure key_events are dramatic and advance the plot
- Strengthen emotional beats and character development
- Add foreshadowing connections to other chapters
- Use double quotes for any string values containing apostrophes or colons""",
}

# Mode-specific globals evolution prompts
GLOBALS_EVOLVE_PROMPTS = {
    "expand": "EXPAND mode: You may ONLY ADD new content. Keep all existing text. You may add new characters or detail.",
    "refine": "REFINE mode: Rewrite to improve quality. Do NOT add or remove characters. Same structure, better writing.",
    "full": "FULL mode: Unrestricted. Rewrite, add, or restructure anything to maximize quality.",
}

def score_config(config: dict, host: str, port: int, **kwargs) -> tuple:
    """Score a config using LLM audit. Returns (score: int, review: str)."""
    review = audit_config(config, host=host, port=port, **kwargs)
    score = parse_score(review)
    return score, review


def _yaml_quote_strings(data):
    """Recursively wrap any string containing YAML-sensitive characters in quotes."""
    if isinstance(data, dict):
        return {k: _yaml_quote_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_yaml_quote_strings(item) for item in data]
    elif isinstance(data, str):
        # Characters that can confuse YAML parsers or the LLM
        if any(c in data for c in (':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`')):
            return '"' + data.replace('\\', '\\\\').replace('"', '\\"') + '"'
        # Also quote things that look like timestamps or booleans
        if data.lower() in ('true', 'false', 'yes', 'no', 'on', 'off', 'null', '~'):
            return '"' + data + '"'
        try:
            float(data)
            return '"' + data + '"'
        except ValueError:
            pass
    return data


def _yaml_repair(text):
    """Fix common LLM YAML errors: unquoted colons, apostrophes in single quotes, etc."""
    import re as _re

    # Pass 1: Fix unquoted colons in list items (prose with colons)
    lines = text.split('\n')
    fixed = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith('- ') and ':' in stripped[2:]:
            content = stripped[2:]
            # Skip if it starts with a quote (already quoted)
            if content and content[0] in ('"', "'"):
                fixed.append(line)
                continue
            first_colon = content.index(':')
            before_colon = content[:first_colon].strip()
            if ' ' in before_colon or len(before_colon) > 25:
                content = content.replace('"', '\\"')
                fixed.append(f'{indent}- "{content}"')
                continue
        fixed.append(line)
    text = '\n'.join(fixed)

    # Pass 2: Fix apostrophes inside single-quoted YAML values
    def _fix_single_quotes(line):
        match = _re.match(r'^(\s*(?:-\s+)?(?:\w[\w\s]*:\s+)?)\'(.*)$', line)
        if not match:
            return line
        prefix = match.group(1)
        rest = match.group(2)
        if rest.endswith("'"):
            inner = rest[:-1]
            if "'" in inner:
                inner = inner.replace('"', '\\"')
                return f'{prefix}"{inner}"'
        return line

    lines = text.split('\n')
    text = '\n'.join(_fix_single_quotes(line) for line in lines)

    return text


def _fix_key_events_dicts(cfg):
    """Convert any dict entries in key_events lists back to strings."""
    if not isinstance(cfg, dict):
        return cfg
    for ch in cfg.get('chapters', []):
        if not isinstance(ch, dict):
            continue
        events = ch.get('key_events', [])
        if isinstance(events, list):
            fixed = []
            for ev in events:
                if isinstance(ev, dict):
                    for k, v in ev.items():
                        fixed.append(f"{k}: {v}")
                else:
                    fixed.append(str(ev))
            ch['key_events'] = fixed
    return cfg



def _parse_yaml_response(text, host=None, port=None, **kwargs):
    """Parse a YAML response with three-tier strategy: raw → repair → LLM retry."""
    import re

    cleaned = text.strip()
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'^```ya?ml\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    def _try_parse(t):
        try:
            parsed = yaml.safe_load(t)
            if not isinstance(parsed, dict):
                raise ValueError("YAML did not parse to a dictionary")
            return parsed, None
        except Exception as e:
            return None, e

    # Attempt 1: raw parse
    result, err = _try_parse(cleaned)
    if result:
        return result

    print(f"      ⚠ YAML parse failed, attempting repair: {err}")
    # Attempt 2: automated repair
    repaired = _yaml_repair(cleaned)
    result, err2 = _try_parse(repaired)
    if result:
        print("      🔧 YAML repaired successfully")
        return result

    # Attempt 3: LLM self-correction (if host/port provided)
    if host and port:
        print(f"      ⚠ Repair failed: {err2}")
        print("      🔄 Asking LLM to fix its YAML...")
        fix_prompt = f"""Your previous YAML output had a syntax error:

{err}

Here is the broken YAML (first 4000 chars):

{cleaned[:4000]}

Please output the CORRECTED version as valid YAML only. Pay special attention to:
- Apostrophes inside single-quoted strings (use double quotes instead)
- Colons inside list item text (quote the string)
- Proper indentation
Output ONLY the corrected YAML, nothing else."""

        retry_response = llm_generate(
            "You are a YAML syntax expert. Fix the YAML and output ONLY valid YAML.",
            fix_prompt,
            host=host, port=port,
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            temperature=0.3,
        )

        retry_cleaned = retry_response.strip()
        retry_cleaned = re.sub(r'<think>.*?</think>', '', retry_cleaned, flags=re.DOTALL)
        retry_cleaned = re.sub(r'^```ya?ml\s*\n', '', retry_cleaned)
        retry_cleaned = re.sub(r'\n```\s*$', '', retry_cleaned)
        retry_cleaned = retry_cleaned.strip()

        result, err3 = _try_parse(retry_cleaned)
        if not result:
            repaired_retry = _yaml_repair(retry_cleaned)
            result, _ = _try_parse(repaired_retry)
            if result:
                print("      🔧 LLM retry + repair succeeded")
                return result

        if result:
            print("      ✅ LLM self-corrected its YAML")
            return result
        else:
            print(f"      ❌ All parse attempts failed: {err3}")

    return None


def _merge_chapter(original: dict, evolved: dict) -> dict:
    """Merge evolved chapter with original, preserving any fields the LLM dropped."""
    merged = dict(original)  # Start with all original fields
    for key, value in evolved.items():
        if value is not None and value != "" and value != []:
            merged[key] = value
    return merged


def evolve_analysis(config: dict, mode: str, host: str, port: int, **kwargs) -> str:
    """Phase 1: Generate a strategic editorial brief (prose, not YAML)."""
    import yaml as _yaml
    safe_config = _yaml_quote_strings(config)
    config_text = _yaml.dump(safe_config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    system = ANALYSIS_PROMPTS[mode]

    user = f"""Here is the complete book configuration to analyze:

--- BOOK CONFIG ---
{config_text}
--- END CONFIG ---

Produce your editorial brief now."""

    response = llm_generate(
        system,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )

    import re
    cleaned = response.strip()
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def evolve_globals(config: dict, analysis: str, mode: str, host: str, port: int, **kwargs) -> dict:
    """Phase 2: Evolve non-chapter sections guided by the editorial brief."""
    import yaml as _yaml

    # Extract non-chapter sections
    globals_dict = {k: v for k, v in config.items() if k != "chapters"}
    safe_globals = _yaml_quote_strings(globals_dict)
    globals_text = _yaml.dump(safe_globals, default_flow_style=False, allow_unicode=True, sort_keys=False)

    mode_rules = GLOBALS_EVOLVE_PROMPTS[mode]

    system = f"""You are an expert story architect evolving the global sections of a book configuration.

{mode_rules}

IMPORTANT YAML RULES:
- Use double quotes for any string values containing apostrophes or colons
- Maintain proper 2-space indentation
- Output ONLY the YAML for these sections — no chapters, no commentary, no code fences"""

    user = f"""EDITORIAL BRIEF (use this to guide your improvements):

{analysis}

--- CURRENT GLOBAL SECTIONS ---
{globals_text}
--- END ---

Output the evolved global sections as valid YAML. Include: title, genre, setting, synopsis, pov, characters, and any other non-chapter sections. Do NOT include the chapters section."""

    response = llm_generate(
        system,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )

    result = _parse_yaml_response(response, host=host, port=port, **kwargs)
    if result is None:
        return None

    # Merge: preserve any globals the LLM dropped
    merged = dict(globals_dict)
    for key, value in result.items():
        if key != "chapters" and value is not None:
            merged[key] = value

    return merged


def evolve_chapter(config: dict, chapter: dict, analysis: str, mode: str,
                   host: str, port: int, **kwargs) -> dict:
    """Phase 3: Evolve a single chapter guided by the editorial brief."""
    import yaml as _yaml

    num = chapter.get("number", "?")
    title = chapter.get("title", "Untitled")
    book_title = config.get("title", "Untitled")

    safe_chapter = _yaml_quote_strings(chapter)
    chapter_text = _yaml.dump(safe_chapter, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Build a brief context summary (not the full config — save tokens)
    chapters = config.get("chapters", [])
    chapter_list = []
    for ch in chapters:
        if isinstance(ch, dict):
            ch_num = ch.get("number", "?")
            ch_title = ch.get("title", "")
            marker = " ◀ THIS CHAPTER" if ch_num == num else ""
            chapter_list.append(f"  Ch {ch_num}: {ch_title}{marker}")
    chapter_overview = "\n".join(chapter_list)

    system_prompt = CHAPTER_EVOLVE_PROMPTS[mode].format(num=num, title=title, book_title=book_title)

    user = f"""EDITORIAL BRIEF (use this to guide your improvements):

{analysis}

BOOK STRUCTURE:
{chapter_overview}

--- CURRENT CHAPTER {num} ---
{chapter_text}
--- END ---

Output the evolved version of ONLY Chapter {num} as valid YAML.
It must be a YAML mapping (NOT a list item). Include all fields: number, title, synopsis, pov_character, emotional_beat, location, key_events.
Do NOT include any other chapters. No commentary, no code fences."""

    response = llm_generate(
        system_prompt,
        user,
        host=host, port=port,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", DEFAULT_TOP_P),
        min_p=kwargs.get("min_p", DEFAULT_MIN_P),
        repeat_penalty=kwargs.get("repeat_penalty", 1.1),
    )

    result = _parse_yaml_response(response, host=host, port=port, **kwargs)
    if result is None:
        return None

    result = _fix_key_events_dicts({"chapters": [result]}).get("chapters", [result])[0] if "key_events" in result else result

    # Merge with original to preserve any dropped fields
    merged = _merge_chapter(chapter, result)
    merged["number"] = num  # Always preserve the chapter number

    return merged


def evolve_config(config: dict, mode: str, host: str, port: int, **kwargs) -> dict:
    """Orchestrate three-phase chapter-by-chapter evolution."""

    # ── Phase 1: Strategic Analysis ──
    print("   📋 Phase 1: Editorial analysis...")
    analysis = evolve_analysis(config, mode, host, port, **kwargs)
    if not analysis:
        print("   ❌ Failed to generate editorial analysis.")
        return None

    # Print a preview of the analysis
    lines = analysis.split("\n")
    for line in lines[:5]:
        if line.strip():
            print(f"      {line.strip()}")
    if len(lines) > 5:
        print(f"      ... ({len(lines)} lines total)")

    # ── Phase 2: Evolve Global Sections ──
    print("   🌐 Phase 2: Evolving globals (title, synopsis, characters)...")
    new_globals = evolve_globals(config, analysis, mode, host, port, **kwargs)
    if new_globals is None:
        print("   ❌ Failed to evolve global sections.")
        return None
    print("      ✅ Globals evolved")

    # ── Phase 3: Evolve Each Chapter ──
    chapters = config.get("chapters", [])
    new_chapters = []

    for i, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            new_chapters.append(chapter)
            continue

        num = chapter.get("number", i + 1)
        title = chapter.get("title", "Untitled")
        print(f"   📖 Phase 3: Evolving chapter {num}/{len(chapters)}: \"{title}\"...")

        evolved = evolve_chapter(config, chapter, analysis, mode, host, port, **kwargs)
        if evolved is None:
            print(f"      ⚠ Failed to evolve chapter {num}, keeping original")
            new_chapters.append(chapter)
        else:
            new_chapters.append(evolved)
            print(f"      ✅ Chapter {num} evolved")

    # ── Reassemble ──
    new_config = dict(new_globals)
    new_config["chapters"] = new_chapters

    # Final normalization
    new_config = _fix_key_events_dicts(new_config)
    try:
        normalized = yaml.safe_load(yaml.dump(new_config, default_flow_style=False, allow_unicode=True, sort_keys=False))
        return normalized
    except Exception as e:
        print(f"   ⚠ Failed to normalize final config: {e}")
        return new_config


def cmd_evolve(args):
    """Hill-climbing optimizer for story configs."""
    config = load_config(args.config)
    title = config.get("title", "Untitled")
    mode = args.mode
    max_rounds = args.max_rounds

    print(f"\n🧬 Evolving config: {title}")
    print(f"   Mode: {mode} | Max rounds: {max_rounds}")

    llm_kwargs = {
        "max_tokens": getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
        "temperature": getattr(args, "temperature", 0.8),
    }

    # Initial score
    print(f"\n── Round 0: Baseline Score ──")
    print(f"   Scoring current config...")
    best_score, review = score_config(config, args.host, args.port, **llm_kwargs)
    if best_score < 0:
        print("   ⚠ Could not parse baseline score from audit. Proceeding with score=0.")
        best_score = 0
    print(f"   📊 Baseline score: {best_score}/100")

    best_config = config
    import shutil

    for round_num in range(1, max_rounds + 1):
        print(f"\n── Round {round_num}/{max_rounds}: Evolving ({mode}) ──")

        # Backup current config
        version_path = next_version_path(args.config)
        shutil.copy2(args.config, version_path)
        print(f"   📦 Backed up: {version_path}")

        # Generate evolved version
        print(f"   Generating evolved config...")
        new_config = evolve_config(best_config, mode, args.host, args.port, **llm_kwargs)

        if new_config is None:
            print(f"   ❌ Evolution failed to produce valid YAML. Stopping.")
            break

        # Preserve metadata
        old_meta = best_config.get("meta", {})
        new_meta = new_config.setdefault("meta", {})
        if "origin_prompt" in old_meta:
            new_meta["origin_prompt"] = old_meta["origin_prompt"]
        from datetime import datetime, timezone
        new_meta["evolved_at"] = datetime.now(timezone.utc).isoformat()
        new_meta["evolve_mode"] = mode
        new_meta["evolve_round"] = round_num

        # Run structural fixes on new config
        num_fixes, new_config = fix_config(new_config)
        if num_fixes:
            print(f"   🔧 Applied {num_fixes} structural fix(es)")

        # Show diff
        config_diff(best_config, new_config)

        # Score the new config
        print(f"\n   Scoring evolved config...")
        new_score, new_review = score_config(new_config, args.host, args.port, **llm_kwargs)

        if new_score < 0:
            print(f"   ⚠ Could not parse score. Keeping previous best.")
            continue

        delta = new_score - best_score
        print(f"   📊 Score: {new_score}/100 (delta: {delta:+d})")

        if new_score > best_score:
            if delta < 2:
                print(f"   ⏸ Plateau detected (delta < 2). Accepting and stopping.")
                best_config = new_config
                best_score = new_score
                save_config(best_config, args.config)
                break
            else:
                print(f"   ✅ Improved! Accepting.")
                best_config = new_config
                best_score = new_score
                save_config(best_config, args.config)
        elif new_score == best_score:
            print(f"   ⏸ No improvement. Stopping.")
            # Rollback — restore the backed up version
            shutil.copy2(version_path, args.config)
            break
        else:
            print(f"   📉 Regression! Rolling back.")
            shutil.copy2(version_path, args.config)
            break

    print(f"\n🧬 Evolution complete!")
    print(f"   Best score: {best_score}/100")
    print(f"   Config: {args.config}")


def cmd_restructure(args):
    """Split oversized chapters into smaller ones."""
    import shutil
    import yaml as _yaml

    config = load_config(args.config)
    title = config.get("title", "Untitled")
    chapters = config.get("chapters", [])
    threshold = args.threshold

    print(f"\n🔪 Restructure: {title}")
    print(f"   Threshold: {threshold} key_events (chapters above this will be candidates)")

    # ── Phase 1: Analyze chapter sizes ──
    print(f"\n── Chapter Size Analysis ──")
    candidates = []
    for i, ch in enumerate(chapters):
        if not isinstance(ch, dict):
            continue
        num = ch.get("number", i + 1)
        ch_title = ch.get("title", "Untitled")
        events = ch.get("key_events", [])
        synopsis = ch.get("synopsis", "")
        event_count = len(events) if isinstance(events, list) else 0
        synopsis_len = len(synopsis) if isinstance(synopsis, str) else 0

        bar = "█" * event_count
        marker = " ← CANDIDATE" if event_count >= threshold else ""
        print(f"   Ch {num:2d}: {event_count:2d} events | {synopsis_len:4d} chars | {bar}{marker}")

        if event_count >= threshold:
            candidates.append((i, ch))

    if not candidates:
        print(f"\n   ✅ No chapters exceed the threshold of {threshold} events. Nothing to restructure.")
        return

    print(f"\n   📋 {len(candidates)} chapter(s) above threshold")

    # ── Phase 2: Ask LLM how to split each candidate ──
    llm_kwargs = {
        "max_tokens": getattr(args, "max_tokens", DEFAULT_MAX_TOKENS),
        "temperature": getattr(args, "temperature", 0.7),
    }

    # Backup before modifying
    version_path = next_version_path(args.config)
    shutil.copy2(args.config, version_path)
    print(f"   📦 Backed up: {version_path}")

    new_chapters = list(chapters)  # mutable copy
    offset = 0  # track index shifts from previous splits

    for orig_idx, chapter in candidates:
        adjusted_idx = orig_idx + offset
        num = chapter.get("number", orig_idx + 1)
        ch_title = chapter.get("title", "Untitled")
        events = chapter.get("key_events", [])
        event_count = len(events) if isinstance(events, list) else 0

        # Suggest split count: roughly ceil(events / (threshold * 0.6))
        suggested_splits = max(2, min(4, (event_count + threshold - 1) // max(1, threshold - 2)))

        print(f"\n── Splitting Chapter {num}: \"{ch_title}\" ({event_count} events → ~{suggested_splits} chapters) ──")

        safe_chapter = _yaml_quote_strings(chapter)
        chapter_text = _yaml.dump(safe_chapter, default_flow_style=False, allow_unicode=True, sort_keys=False)

        system = f"""You are an expert story architect splitting an oversized chapter into {suggested_splits} smaller chapters.

RULES:
- Split this chapter into exactly {suggested_splits} new chapters
- Each new chapter should have a clear narrative focus
- Distribute the key_events logically across the new chapters
- Each new chapter needs: title, synopsis, pov_character, emotional_beat, location, key_events
- Maintain narrative flow — the new chapters should read as a natural sequence
- Use double quotes for any string values containing apostrophes or colons
- Do NOT use chapter numbers — they will be assigned automatically
- Do NOT echo back book context, synopsis, characters, or any other metadata
- Output ONLY the new chapter list — nothing else

Output ONLY the {suggested_splits} chapters as a YAML list (starting with '- title:').
No book metadata, no commentary, no code fences."""

        # Build lightweight book context
        book_title = config.get("title", "Untitled")
        book_synopsis = config.get("synopsis", "")
        characters = config.get("characters", [])
        char_summary = ""
        if characters:
            char_names = []
            for c in characters:
                if isinstance(c, dict):
                    char_names.append(f"{c.get('name', '?')} ({c.get('role', '?')})")
                elif isinstance(c, str):
                    char_names.append(c)
            char_summary = "Characters: " + ", ".join(char_names)

        chapter_outline = "\n".join(
            f"  Ch {c.get('number', i+1)}: {c.get('title', '?')}"
            for i, c in enumerate(chapters) if isinstance(c, dict)
        )

        user = f"""BOOK CONTEXT:
Title: {book_title}
Synopsis: {book_synopsis}
{char_summary}

Chapter outline:
{chapter_outline}

--- CHAPTER {num} TO SPLIT ---
{chapter_text}
--- END ---

Split this into {suggested_splits} smaller chapters. Output as a YAML list."""

        print(f"   🤖 Asking LLM to split...")
        response = llm_generate(
            system,
            user,
            host=args.host, port=args.port,
            max_tokens=llm_kwargs.get("max_tokens", 4096),
            temperature=llm_kwargs.get("temperature", 0.7),
        )

        # Parse the response as a list of chapters
        import re
        cleaned = response.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'^```ya?ml\s*\n', '', cleaned)
        cleaned = re.sub(r'\n```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        # Try to parse - might be a list or a dict with chapters key
        try:
            parsed = yaml.safe_load(cleaned)
        except Exception as e1:
            # Try repair
            repaired = _yaml_repair(cleaned)
            try:
                parsed = yaml.safe_load(repaired)
            except Exception as e2:
                print(f"   ❌ Failed to parse split output: {e2}")
                print(f"   ⚠ Keeping original chapter {num}")
                continue

        # Normalize to list
        if isinstance(parsed, dict):
            if "chapters" in parsed:
                split_chapters = parsed["chapters"]
            else:
                split_chapters = [parsed]
        elif isinstance(parsed, list):
            split_chapters = parsed
        else:
            print(f"   ❌ Unexpected parse result type: {type(parsed)}")
            print(f"   ⚠ Keeping original chapter {num}")
            continue

        # Validate we got actual chapter dicts — strip non-chapter fields
        CHAPTER_FIELDS = {"title", "synopsis", "pov_character", "emotional_beat",
                          "location", "key_events", "number"}
        valid_splits = []
        for sc in split_chapters:
            if isinstance(sc, dict) and ("title" in sc or "synopsis" in sc):
                # Strip any echoed book-level keys (genre, characters, all, etc.)
                sc = {k: v for k, v in sc.items() if k in CHAPTER_FIELDS}
                if not sc.get("title") and not sc.get("synopsis"):
                    continue  # Skip empty after stripping
                # Ensure key_events are strings
                sc = _fix_key_events_dicts({"chapters": [sc]})["chapters"][0]
                valid_splits.append(sc)

        if len(valid_splits) < 2:
            print(f"   ❌ Got {len(valid_splits)} valid chapters (need ≥2). Keeping original.")
            continue

        print(f"   ✅ Split into {len(valid_splits)} chapters:")
        for sc in valid_splits:
            sc_title = sc.get("title", "Untitled")
            sc_events = len(sc.get("key_events", []))
            print(f"      • \"{sc_title}\" ({sc_events} events)")

        # Replace the original chapter with the splits
        new_chapters[adjusted_idx:adjusted_idx + 1] = valid_splits
        offset += len(valid_splits) - 1

    # ── Renumber all chapters ──
    print(f"\n── Renumbering {len(new_chapters)} chapters ──")
    for i, ch in enumerate(new_chapters):
        if isinstance(ch, dict):
            ch["number"] = i + 1

    config["chapters"] = new_chapters

    # Fix any structural issues
    num_fixes, config = fix_config(config)
    if num_fixes:
        print(f"   🔧 Applied {num_fixes} structural fix(es)")

    save_config(config, args.config)
    print(f"\n🔪 Restructure complete!")
    print(f"   {len(chapters)} chapters → {len(new_chapters)} chapters")
    print(f"   Config: {args.config}")

def cmd_think(args):
    """Have a character think about a situation."""
    config = load_config(args.config)

    # Find character
    char = None
    for c in config.get("characters", []):
        if c.get("name", "").lower() == args.character.lower():
            char = c
            break

    if not char:
        names = [c.get("name", "?") for c in config.get("characters", [])]
        print(f"ERROR: Character '{args.character}' not found. Available: {', '.join(names)}")
        sys.exit(1)

    print(f"\n🧠 {char['name']} thinking about: {args.situation}")
    thought = character_think(
        char, args.situation, config,
        host=args.host, port=args.port,
        max_tokens=args.max_tokens,
    )
    print(f"\n{'='*60}")
    print(f"  {char['name']} ({char.get('role', '')})")
    print(f"{'='*60}\n")
    print(thought)


def cmd_pdf(args):
    """Convert book.md to a typeset PDF."""
    import subprocess
    import shutil

    # Find input
    if args.input:
        book_md = args.input
    elif args.config:
        config = load_config(args.config)
        title = config.get("title", "untitled")
        output_dir = args.output_dir or title.lower().replace(" ", "_").replace("'", "")
        book_md = os.path.join(output_dir, "book.md")
    else:
        print("ERROR: Provide either --input book.md or --config story_config.yaml")
        sys.exit(1)

    if not os.path.exists(book_md):
        print(f"ERROR: {book_md} not found. Run compose first.", file=sys.stderr)
        sys.exit(1)

    # Output PDF path
    pdf_path = args.pdf_output or book_md.replace(".md", ".pdf")

    print(f"\n📄 PDF EXPORT")
    print(f"   Source: {book_md}")
    print(f"   Output: {pdf_path}")
    print(f"   Paper:  {args.paper}")
    print(f"   Font:   {args.font_size}pt")

    # Check pandoc
    if not shutil.which("pandoc"):
        print("ERROR: pandoc not found. Install with: sudo apt install pandoc", file=sys.stderr)
        sys.exit(1)

    # Build pandoc command with LaTeX options for book typesetting
    cmd = [
        "pandoc",
        book_md,
        "-o", pdf_path,
        "--pdf-engine=lualatex",
        f"--variable=papersize:{args.paper}",
        f"--variable=fontsize:{args.font_size}pt",
        "--variable=documentclass:book",
        "--variable=classoption:openany",
        "--variable=geometry:margin=1in",
        "--variable=mainfont:URW Bookman",
        "--variable=linestretch:1.3",
        "--variable=indent:true",
        "--variable=subparagraph:true",
        "--toc",
        "--toc-depth=1",
        "--top-level-division=chapter",
    ]

    # Add custom LaTeX header for styling
    header_file = book_md.replace(".md", "_header.tex")
    header_latex = r"""\usepackage{fancyhdr}
\usepackage{titlesec}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\small\textit{\leftmark}}
\fancyhead[RO]{\small\textit{\rightmark}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-30pt}{40pt}
"""
    with open(header_file, "w") as f:
        f.write(header_latex)
    cmd.extend(["-H", header_file])

    print(f"   Running pandoc...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            file_size = os.path.getsize(pdf_path)
            print(f"   ✅ PDF generated: {pdf_path} ({file_size // 1024} KB)")
        else:
            print(f"   ⚠ pandoc failed (exit {result.returncode})")
            if result.stderr:
                # Show last 20 lines of error for debugging
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-20:]:
                    print(f"      {line}")
    except subprocess.TimeoutExpired:
        print("   ⚠ pandoc timed out after 120s")
    except FileNotFoundError:
        print("   ⚠ pandoc not found in PATH")
    finally:
        # Clean up temp header file
        if os.path.exists(header_file):
            os.remove(header_file)


# ─── main ───────────────────────────────────────────────────────────────

def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="GhostWriter — AI-powered fiction writing system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s genesis "A cyberpunk heist in Neo-Tokyo, 2087"
  %(prog)s --base-url http://127.0.0.1:11434/v1 --model qwen2.5:32b genesis "A haunted lighthouse"
  %(prog)s compose story_config.yaml --review --character-think
  %(prog)s write "A gothic romance set in Victorian Cornwall"
  %(prog)s revise story_config.yaml                     # revise all chapters
  %(prog)s revise story_config.yaml --chapters 1 3      # revise chapters 1 and 3
  %(prog)s pdf --config story_config.yaml               # export to PDF
  %(prog)s pdf --input mybook/book.md --paper a5        # direct markdown to PDF
  %(prog)s think story_config.yaml "Elena" "She just discovered the secret room"

The LLM endpoint comes from ~/.config/ghostwriter/config.toml,
./ghostwriter.toml, or GHOSTWRITER_BASE_URL. See ghostwriter.example.toml.
        """
    )

    # Global args. Defaults come from the config file / environment.
    parser.add_argument("--host", default=settings.llm_host,
                        help="LLM host. Ignored when --base-url is set "
                             "(default: from config, else 127.0.0.1)")
    parser.add_argument("--port", type=int, default=settings.llm_port,
                        help="LLM port. Ignored when --base-url is set "
                             "(default: from config, else 8080)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API root, including /v1. "
                             "Overrides --host and --port")
    parser.add_argument("--model", default=None,
                        help="Model name sent to the endpoint "
                             "(default: from config; llama.cpp accepts any value)")
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict-openai", action="store_true",
                        help="Omit llama.cpp-only sampling fields")
    strict.add_argument("--no-strict-openai", action="store_true",
                        help="Send llama.cpp sampling fields (min_p, repeat_penalty)")
    parser.add_argument("--max-tokens", type=int, default=settings.max_tokens)
    parser.add_argument("--temperature", type=float, default=settings.temperature)

    sub = parser.add_subparsers(dest="command", required=True)

    # genesis
    p_gen = sub.add_parser("genesis", help="Generate story config from prompt")
    p_gen.add_argument("prompt", help="Creative prompt")
    p_gen.add_argument("--output", default="story_config.yaml")

    # compose
    p_comp = sub.add_parser("compose", help="Write book from config")
    p_comp.add_argument("config", help="Path to YAML config")
    p_comp.add_argument("--output", default=None, help="Output directory")
    p_comp.add_argument("--review", action="store_true", help="Run review pass on each chapter")
    p_comp.add_argument("--character-think", action="store_true",
                         help="Run character perspective simulation before each chapter")
    p_comp.add_argument("--chapter", type=int, default=None, help="Compose only this chapter number")

    # write (full pipeline)
    p_write = sub.add_parser("write", help="Full pipeline: genesis → compose")
    p_write.add_argument("prompt", help="Creative prompt")
    p_write.add_argument("--output", default=None, help="Output directory")
    p_write.add_argument("--review", action="store_true")
    p_write.add_argument("--character-think", action="store_true")

    # review
    p_rev = sub.add_parser("review", help="Review existing chapters")
    p_rev.add_argument("config", help="Path to YAML config")
    p_rev.add_argument("--output", default=None, help="Book directory")
    p_rev.add_argument("--chapter", type=int, default=None, help="Review only this chapter number")

    # revise
    p_revise = sub.add_parser("revise", help="Revise chapters based on review feedback")
    p_revise.add_argument("config", help="Path to YAML config")
    p_revise.add_argument("--output", default=None, help="Book directory")
    p_revise.add_argument("--chapters", nargs="+", help="Specific chapter numbers to revise (default: all)")
    p_revise.add_argument("--strategy", default="general",
                          choices=list(REVISION_STRATEGIES.keys()),
                          help="Revision strategy (default: general)")

    # audit
    p_audit = sub.add_parser("audit", help="Validate config and run LLM plot critique")
    p_audit.add_argument("config", help="Path to YAML config")
    p_audit.add_argument("--output", default=None, help="Output directory for audit report")
    p_audit.add_argument("--syntax-only", action="store_true", help="Skip LLM review, only run structural checks")
    p_audit.add_argument("--fix", action="store_true", help="Auto-fix structural issues (strings→dicts, summary→synopsis, add key_events)")

    # polish
    p_polish = sub.add_parser("polish", help="Automated review→revise loop with convergence")
    p_polish.add_argument("config", help="Path to YAML config")
    p_polish.add_argument("--output", default=None, help="Book directory")
    p_polish.add_argument("--chapter", type=int, default=None, help="Polish only this chapter number")
    p_polish.add_argument("--strategy", default="general",
                          choices=list(REVISION_STRATEGIES.keys()),
                          help="Revision strategy (default: general)")
    p_polish.add_argument("--min-score", type=int, default=80, help="Stop when score reaches this (default: 80)")
    p_polish.add_argument("--max-rounds", type=int, default=3, help="Max review-revise rounds (default: 3)")

    # distill
    p_distill = sub.add_parser("distill", help="Extract refined config from existing prose chapters")
    p_distill.add_argument("config", help="Path to YAML config")
    p_distill.add_argument("--output", default=None, help="Book directory containing chapter files")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Hill-climbing optimizer for story configs")
    p_evolve.add_argument("config", help="Path to YAML config")
    p_evolve.add_argument("--mode", default="full", choices=["expand", "refine", "full"],
                          help="Evolution mode (default: full)")
    p_evolve.add_argument("--max-rounds", type=int, default=3, help="Max evolution rounds (default: 3)")

    # restructure
    p_restruct = sub.add_parser("restructure", help="Split oversized chapters into smaller ones")
    p_restruct.add_argument("config", help="Path to YAML config")
    p_restruct.add_argument("--threshold", type=int, default=6,
                            help="Chapters with >= this many key_events will be split (default: 6)")

    # think
    p_think = sub.add_parser("think", help="Character perspective simulation")
    p_think.add_argument("config", help="Path to YAML config")
    p_think.add_argument("character", help="Character name")
    p_think.add_argument("situation", help="Situation to think about")

    # pdf
    p_pdf = sub.add_parser("pdf", help="Export book to PDF")
    p_pdf.add_argument("--config", default=None, help="Path to YAML config (finds book.md from output dir)")
    p_pdf.add_argument("--input", default=None, help="Direct path to book.md")
    p_pdf.add_argument("--output-dir", default=None, help="Book directory (if using --config)")
    p_pdf.add_argument("--pdf-output", default=None, help="Output PDF path (default: book.pdf next to book.md)")
    p_pdf.add_argument("--paper", default="letter", choices=["letter", "a4", "a5"], help="Paper size")
    p_pdf.add_argument("--font-size", type=int, default=11, help="Font size in pt (default: 11)")

    args = parser.parse_args()

    if args.base_url:
        settings = settings.retarget(args.base_url, os.environ)
        args.host = settings.llm_host
        args.port = settings.llm_port
    if args.model is not None:
        settings = settings.with_model(args.model)
    if args.strict_openai:
        settings = settings.with_strict(True)
    elif args.no_strict_openai:
        settings = settings.with_strict(False)
    set_settings(settings)

    if args.command != "pdf":
        model_bit = f"  model={settings.llm_model}" if settings.llm_model else ""
        where = f"  ({settings.config_path})" if settings.config_path else ""
        print(f"   LLM endpoint: {redact_url(settings.llm_base_url)}{model_bit}{where}")

    commands = {
        "genesis": cmd_genesis,
        "compose": cmd_compose,
        "write": cmd_write,
        "review": cmd_review,
        "revise": cmd_revise,
        "audit": cmd_audit,
        "polish": cmd_polish,
        "distill": cmd_distill,
        "evolve": cmd_evolve,
        "restructure": cmd_restructure,
        "think": cmd_think,
        "pdf": cmd_pdf,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
