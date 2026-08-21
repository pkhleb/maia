# maia — Personal CLI Coding Assistant

A terminal-based coding assistant built around a self-hosted LLM inference server and a custom code retrieval library. Built as a daily driver for a vim/CLI-only development workflow, with a focus on intelligent context retrieval and context window management.

## Architecture Overview

```
local machine (CLI) → vLLM inference server (RunPod) → Qwen2.5-Coder-7B-Instruct
       ↑
  retlib — hybrid retrieval pipeline
  sliding window + summarization
```

maia uses [retlib](https://github.com/pkhleb/retlib) — a purpose-built code retrieval library — to decide what context to send to the model on each request. Rather than dumping files into the prompt, retlib parses the codebase into a code graph and retrieves only the symbols most relevant to the current query.

## Technical Highlights

### Self-Hosted Inference
Runs against a self-hosted [vLLM](https://github.com/vllm-project/vllm) endpoint serving `Qwen/Qwen2.5-Coder-7B-Instruct` on a RunPod RTX PRO 4500. Exposes an OpenAI-compatible API, making the inference backend swappable without changing application code.

### Hybrid Code Retrieval (via retlib)
Each user query goes through a four-stage retrieval pipeline before reaching the model:

1. **Parse** — the codebase is parsed with Python's `ast` module into a symbol graph: modules, classes, functions, methods, with typed edges (`CALLS`, `CONTAINS`, `IMPORTS`, `INHERITS`)
2. **Index** — symbols are indexed for both BM25 lexical search and semantic vector search (`all-MiniLM-L6-v2` embeddings stored in SQLite)
3. **Retrieve** — the query is classified by task intent (`debug`, `refactor`, `understand`) to select appropriate graph edge types, then hybrid search finds seed symbols and graph expansion surfaces related ones
4. **Build context** — results are ranked by a weighted combination of semantic similarity, BM25 score, graph distance, edge type, and symbol kind, then assembled within a token budget

The pipeline is fully observable — each request prints which symbols were included, their scores, and how many tokens were used.

### Context Window Management
Prompt token growth is addressed with a two-layer approach:
- **Sliding window** — only the last N conversation turns are sent as message history
- **Rolling summarization** — every N turns, older history is compressed into a natural language summary that persists across sessions in `history.json`
- **/clear command** — compresses current history into the summary before wiping turns, preserving state without carrying stale context

### Incremental Indexing
On startup, maia indexes the current project directory. Retlib uses file checksums to skip unchanged files, so subsequent startups are fast. A `/reindex` command is available for mid-session updates when files change.

## Project Structure

```
maia/
├── assistant.py        # REPL loop, startup indexing, main entry point
├── config.py           # Model config, tunable parameters
├── history_manager.py  # History/summary persistence (JSON)
├── openai_client.py    # vLLM client wrapper (singleton, loads from .env)
└── utils.py            # System prompt assembly, summarization
```

Context retrieval is handled entirely by [retlib](https://github.com/pkhleb/retlib).

## Setup

### 1. Deploy vLLM on RunPod

Create a pod with:
- **GPU:** RTX PRO 4500 (32GB VRAM)
- **Container image:** `vllm/vllm-openai:v0.6.6`
- **Start command:**
  ```
  --model Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --gpu-memory-utilization 0.95 --max-model-len 16384 --download-dir /workspace/models
  ```
- Attach a persistent network volume at `/workspace` to cache model weights (~15GB)

### 2. Create a `.env` file

```bash
RUNPOD_API_KEY=sk-your-pod-key-here
RUNPOD_BASE_URL=https://your-pod-id-8000.proxy.runpod.net/v1
```

### 3. Install and run

```bash
# install retlib
pip install -e ../retlib

# install maia dependencies
pip install openai python-dotenv

# run
python assistant.py
```

## Usage

```
> help me set up a postgres database        # baseline task
> add a users table with email and password # component task
> I'm getting a 502 error on /api/login     # debug task
> /clear                                    # compress and reset context
> /reindex                                  # reindex after editing files
```

On each request, maia prints a retrieval summary:
```
[retlib] Token budget: 2702 / 4096 used
Included (10): main, build_system_prompt, utils, ...
```

## Design Decisions

**Why self-hosted over API services?**
Full control over context window size, inference parameters, and model selection without artificial rate limits. Enables exploration of vLLM-specific features like constrained decoding.

**Why a separate retrieval library?**
Decoupling retrieval from the assistant makes both components independently testable and reusable. retlib has its own test suite (146 tests) and can be pointed at any Python codebase.

**Why hybrid BM25 + vector search?**
Neither alone is sufficient. BM25 handles exact name matches ("fix parse_file") while vector search handles semantic queries ("how does authentication work"). The weighted combination with graph expansion outperforms either in isolation.

**Why a sliding window + summary over full history?**
Sending full conversation history causes prompt tokens to grow linearly. Summarization compresses older context while preserving semantic continuity — a common pattern in production LLM applications.

## Observability

Every request surfaces retrieval decisions:
- Which symbols were retrieved and why (score breakdown per signal)
- Token budget usage (included vs excluded symbols)
- Full outgoing message array printed before each API call
- Per-request token counts (prompt / completion / total)

## Known Limitations

- **Python only** — retlib's AST parser targets Python; other languages would need additional parsers
- **String-based call resolution** — `CALLS` edges match function names as strings, so cross-module calls to external libraries don't link to actual symbols
- **Linear embedding scan** — fine for a single codebase; would need a faiss index at larger scale
- **Small model** — Qwen 2.5 Coder 7B is capable but not state-of-the-art; a larger model would improve response quality on complex tasks

## Planned Improvements

- Eval set for retrieval quality measurement (precision/recall on labeled queries)
- Code-specific embedding model (e.g. `voyage-code-2`) for better semantic retrieval
- True token count feedback from API responses to calibrate the budget
- Dev server log ingestion for in-context debugging
