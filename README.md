# GhostWriter

**AI-powered fiction writing system** that runs against a local LLM (llama.cpp, vLLM, Ollama, or any OpenAI-compatible server).

GhostWriter helps you write long novels with strong structural control. It keeps a rich machine-readable YAML outline (characters, world, chapter-by-chapter synopses, key events, emotional beats, POV, locations) and uses that context on every generation pass so the model stays consistent across 80k–150k words.

---

## Features

- **Genesis** — Turn a creative prompt into a detailed, editable story config
- **Compose** — Generate full prose chapters with character-perspective simulation
- **Iterative quality tools** — `review`, `revise`, `polish`, `audit`, `distill`, `evolve`, `restructure`
- **Web UI** (port 8501) — Library of books, chapter editor, live YAML config editor, streaming console, and a powerful **Book AI chat** that can read your outline and *propose structural changes* via tool calling
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
- Create new books from prompts ("New Book" button)
- Browse your library
- Edit chapters and the YAML config live
- Run any pipeline command with real-time streaming output
- Chat with an AI co-author that understands your entire book and can suggest outline changes

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
| `library.path` | `GHOSTWRITER_LIBRARY` | `~/ghostwriter` | Where book projects are stored. `BOOKWRIGHT_LIBRARY` is still honored |
| `web.host` | `GHOSTWRITER_WEB_HOST` | `0.0.0.0` | Interface the web UI binds |
| `web.port` | `GHOSTWRITER_WEB_PORT` | `8501` | Web UI port |

`ghostwriter --base-url`, `--host`, `--port`, `--model`, `--strict-openai`, and `--no-strict-openai` cover one run. `ghostwriter-web --host` and `--port` are the bind address, not the model.

You can point several machines at the same library directory. The web UI scans it.

---

## The AI Chat (Web UI)

The "Chat" tab is special. It is a tool-calling agent that can:

- Read any section of your config or any chapter
- Search the outline
- Propose changes (`add_chapter`, `update_chapter`, `add_character`, `replace_section`, etc.)

When the model wants to change something, it shows you a **proposal** that you can accept or reject before it touches your files. This keeps you in control while giving the AI real agency over structure.

---

## Common Workflows

1. **New novel**
   - Web UI → ✨ New Book (or CLI `genesis`)
   - Edit the generated YAML until you're happy
   - Run `compose` (or "Compose" in the console tab)
   - Use `review` + `revise` or the `polish` loop on weak chapters

2. **Existing manuscript**
   - Drop your chapters into a folder + write a minimal `config.yaml`
   - Run `distill` to let GhostWriter extract a rich outline from the prose
   - Then iterate with `review`/`polish`/`evolve`

3. **Heavy revision pass**
   - `audit` (pre-composition sanity check)
   - `polish --min-score 85 --max-rounds 5`
   - `restructure` on chapters that grew too large

---

## Tips for Best Results

- The quality of the **outline** (especially `key_events`, `emotional_beat`, and `synopsis` per chapter) has a bigger impact than prompt engineering.
- Keep the LLM context window large (16k–64k+). GhostWriter sends a lot of grounding context.
- Use the web UI's AI chat early and often — it's excellent for discovering plot holes and suggesting new chapters.
- Versioned backups mean you can always roll back.

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

---

*GhostWriter was previously known as BookWright during early development.*