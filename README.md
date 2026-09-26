# GhostWriter

GhostWriter is a way to write a long novel with an AI while you stay in charge of the structure. It talks to a local or hosted OpenAI-compatible model (llama.cpp, vLLM, Ollama, or a hosted API).

You chat until the outline feels right, then lock that outline in. The model drafts a chapter from it. You edit the prose yourself or with the AI. The model scores the chapter and rewrites it. You tweak the result and run that loop again until the chapter is one you want to keep.

The locked outline is a YAML config: characters, world, and a chapter-by-chapter plan with a synopsis, key events, emotional beat, point of view, and location. Every draft is written against that plan, so a book can stay consistent across 80k–150k words. Chat can propose changes to the outline. Nothing in it changes on disk until you accept.

## How a book gets written

1. **Chat the outline into shape.** Open a book and use the Chat tab. Ask for characters, turns, missing chapters, or a different ending. The model reads the outline and the chapters, then proposes edits. Accept a proposal to apply it, or reject it and keep talking.

2. **Lock the outline.** When the plan is the one you want, stop changing it. That YAML file is what Compose follows. You can still edit it by hand in the Config tab.

3. **Flesh out a chapter.** Run Compose on one chapter, or on the whole book. The model writes the prose from the locked plan.

4. **Edit.** Change the chapter in the editor, or ask Chat or Revise to change it. Manual edits and model edits share the same file, and each save keeps a numbered backup.

5. **Score, rewrite, repeat.** Review reads the chapter against the outline and scores it. Revise rewrites from that critique. Polish repeats review and revise until the score reaches the bar you set, or you stop it. Read the result, tweak the lines you care about, and run another pass when you want one.

---

## Features

- **Genesis** — Turn a creative prompt into a detailed, editable story config
- **Compose** — Generate full prose chapters with character-perspective simulation
- **Iterative quality tools** — `review`, `revise`, `polish`, `audit`, `distill`, `evolve`, `restructure`
- **Web UI** (port 8501) — Library, chapter editor, live YAML editor, streaming console, and the Chat tab where you build the outline and accept or reject each proposed change
- **Versioned backups** — Every save and pipeline step creates `_vN` snapshots
- **PDF export** — Via pandoc + weasyprint (or your own toolchain)

---

## Requirements

- Python ≥ 3.11
- An OpenAI-compatible chat-completions endpoint
  - llama.cpp (`llama-server` or `./server`) is the default assumption: `http://127.0.0.1:8080/v1`
  - Ollama, vLLM, LM Studio, and hosted APIs such as [xAI](https://docs.x.ai) work the same way once `base_url` points at their `/v1` root
- (Optional) `pandoc` + a PDF engine for the `pdf` command

---

## Quick Start

### 1. Run it

From a clone of this repo:

```bash
python3 server.py
# Listens on 0.0.0.0:8501 — open http://127.0.0.1:8501
```

If FastAPI and the other dependencies are already available, that Python is the one that runs. If they are not, the first launch creates a `.venv` in this directory, installs them, and starts the app with that environment. The install needs network access once. Later launches start directly.

```bash
python3 ghostwriter.py --help
```

uses the same launcher for the CLI.

To put `ghostwriter` and `ghostwriter-web` on your PATH:

```bash
pip install .
```

Use `pip install -e .` while you are changing the code. An environment that already has the dependencies is left as-is. GhostWriter does not create a second `.venv` in that case.

### 2. Point it at a model

Copy the example and edit the endpoint. No code changes.

```bash
mkdir -p ~/.config/ghostwriter
cp ghostwriter.example.toml ~/.config/ghostwriter/config.toml
```

llama.cpp on port 8080 is already the default, including when you skip the file entirely:

```bash
llama-server -m ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
    --host 127.0.0.1 --port 8080 --ctx-size 32768
```

Ollama:

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5:32b"
strict_openai = true
```

xAI, or any other hosted OpenAI-compatible API. Keep the key in the environment:

```toml
[llm]
base_url = "https://api.x.ai/v1"
model = "grok-4.7"
strict_openai = true
```

```bash
export XAI_API_KEY="..."            # picked up automatically for *.api.x.ai
# or, for any endpoint:
export GHOSTWRITER_API_KEY="..."
```

`strict_openai` drops llama.cpp-only sampling fields (`min_p`, `repeat_penalty`) that some servers reject. It is turned on automatically for `*.api.x.ai`.

A `ghostwriter.toml` in the directory you launch from wins over `~/.config/ghostwriter/config.toml`. `GHOSTWRITER_CONFIG` wins over both. Environment variables and CLI flags win over the file. `ghostwriter.toml` is gitignored so a local key does not get committed. The example file lists every knob.

### 3. Open the web interface

`python3 server.py` from step 1 already serves the UI. After `pip install`, the same server is on your PATH:

```bash
ghostwriter-web
# Listens on 0.0.0.0:8501 — open http://127.0.0.1:8501
```

`0.0.0.0` is the web app, so other machines on your LAN can open it. The LLM endpoint is separate and still defaults to `127.0.0.1:8080`. Change the bind address with `web.host` in the config, `GHOSTWRITER_WEB_HOST`, or `ghostwriter-web --host 127.0.0.1`.

The UI lets you:
- Create a book and chat until the outline is the one you want to lock
- Edit that outline and the chapter prose by hand
- Run Compose, Review, Revise, and Polish with streaming output
- Accept or reject each structural change before it touches the files

Host and port fields in the UI start from the config. Leave them alone unless you want one run to hit a different server.

### 4. Or use the CLI

```bash
ghostwriter --help

ghostwriter genesis "A noir detective story set in 1940s Los Angeles"
ghostwriter compose my_story.yaml
ghostwriter write "A gothic romance set in Victorian Cornwall"
ghostwriter review my_story.yaml
ghostwriter revise my_story.yaml --strategy general

# One run, different endpoint
ghostwriter --base-url http://127.0.0.1:11434/v1 --model qwen2.5:32b genesis "A haunted lighthouse"
```

---

## Project Layout

```
ghost-writer/
├── ghostwriter.py             # CLI engine (all the pipeline logic)
├── server.py                  # FastAPI web app
├── bootstrap.py               # First-run .venv setup when dependencies are missing
├── settings.py                # Config-file and environment resolution
├── ghostwriter.example.toml   # Copy this, then point it at your model
├── static/                    # Single-page web UI
├── pyproject.toml
├── setup.cfg                  # ships the web UI with pip install
├── README.md
└── books live in the library path (~/ghostwriter by default)
```

Each book is a directory containing:
- `bookname.yaml` — the source-of-truth structured config
- `chapter_01.md`, `chapter_02.md`, ...
- `review_01.md`, ... (critiques)
- Automatic `*_v1.md`, `*_v2.yaml` backups on every change

---

## Configuration & Environment

File keys live under `[llm]`, `[library]`, and `[web]` in the TOML. Environment variables override the file. CLI flags override the environment.

| Config key | Environment | Default | Description |
|---|---|---|---|
| `llm.base_url` | `GHOSTWRITER_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API root, including `/v1` |
| `llm.host` / `llm.port` | `GHOSTWRITER_HOST` / `GHOSTWRITER_PORT` | `127.0.0.1` / `8080` | Used when `base_url` is not set. Setting either rebuilds an `http://` URL |
| `llm.model` | `GHOSTWRITER_MODEL` | *(empty → `any`)* | Required by Ollama and hosted APIs. llama.cpp accepts anything |
| `llm.api_key` | `GHOSTWRITER_API_KEY` | *(empty)* | Bearer token. For `*.api.x.ai`, `XAI_API_KEY` is used when this is empty |
| `llm.strict_openai` | `GHOSTWRITER_STRICT_OPENAI` | `false` (on for `*.api.x.ai`) | Omit llama.cpp-only request fields |
| `llm.temperature` | `GHOSTWRITER_TEMPERATURE` | `0.88` | Default sampling temperature |
| `llm.max_tokens` | `GHOSTWRITER_MAX_TOKENS` | `16384` | Default max tokens |
| `library.path` | `GHOSTWRITER_LIBRARY` | `~/ghostwriter` | Where book projects are stored |
| `web.host` | `GHOSTWRITER_WEB_HOST` | `0.0.0.0` | Interface the web UI binds |
| `web.port` | `GHOSTWRITER_WEB_PORT` | `8501` | Web UI port |

`ghostwriter --base-url`, `--host`, `--port`, `--model`, `--strict-openai`, and `--no-strict-openai` cover one run. `ghostwriter-web --host` and `--port` are the bind address, not the model.

You can point several machines at the same library directory. The web UI scans it.

---

## The AI Chat (Web UI)

The Chat tab is where the outline gets built. It can read any part of the config or any chapter, search the plan, and propose changes (`add_chapter`, `update_chapter`, `add_character`, `replace_section`, and so on).

A proposal stays on screen until you accept or reject it. Accepting is what writes the outline. That is the lock you are choosing, one edit at a time.

---

## Common Workflows

1. **A new novel, one chapter at a time**
   - Web UI → New Book, or CLI `genesis`, to get a first YAML draft
   - Chat until the outline is right, and accept only the proposals you want
   - Compose the chapter you are ready to write
   - Edit it in the chapter editor, or with Chat / Revise
   - Review to score it, then Revise or Polish, and tweak until you are happy
   - Move to the next chapter with the same outline still locked

2. **Existing manuscript**
   - Drop your chapters into a folder and write a minimal `config.yaml`
   - Run `distill` to extract an outline from the prose
   - Then use the same review, edit, and polish loop

3. **A hard pass on a weak chapter**
   - `audit` checks the outline before more prose is written
   - `polish --min-score 85 --max-rounds 5` keeps scoring and rewriting
   - `restructure` splits a chapter whose plan got too large

---

## Tips for Best Results

- Finish the outline before you spend passes on prose. `key_events`, `emotional_beat`, and `synopsis` do more for a chapter than prompt tweaks.
- Keep the context window large (16k–64k+). Each draft is sent with a lot of the locked plan.
- Chat is for discovering holes and changing the plan. Compose, Review, and Polish are for the prose once that plan is locked.
- Versioned backups mean you can roll a chapter or the outline back to an earlier save.

---

## Development

```bash
pip install -e .
python3 -m unittest discover -s tests -v

ghostwriter-web                 # auto-reload on
ghostwriter-web --no-reload
```

The project uses only a handful of dependencies (see `pyproject.toml`).

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Status & Intent

GhostWriter is a personal creative tool that has been used to write multiple full-length novels. It is actively maintained.

It is deliberately opinionated about how long-form fiction should be written with AI. In particular, it strongly favors:

- Treating an explicit, machine-readable outline (the YAML config) as the source of truth rather than relying on the model's fuzzy memory
- Iterative, multi-pass refinement through critique and revision loops
- Keeping the human firmly in the decision-making loop — the AI can read state and *propose* changes via tools, but you must accept them

These are not accidental constraints. GhostWriter was built as a demanding, long-horizon creative domain in which to develop and refine patterns for **agent harnesses**: structured systems that let language models take meaningful, grounded actions against complex persistent state while staying under meaningful human control.

In other words, this agent harness for writing books was developed in part as a way to learn how to build better agent harnesses in general.

Pull requests and issues are welcome, especially around:

- New pipeline commands
- Better handling of very long contexts
- UI/UX improvements
- Export formats

