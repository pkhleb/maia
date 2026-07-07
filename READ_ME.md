# maia — Personal CLI Coding Assistant

A terminal-based coding assistant built around a self-hosted LLM inference server, with a focus on intelligent context retrieval and context window management. Built as a daily driver for a vim/CLI-only development workflow.

## Architecture Overview

```
local machine (CLI) → vLLM inference server (RunPod) → Qwen2.5-Coder-7B-Instruct
       ↑
  AST-based context retrieval
  sliding window + summarization
  structured output (constrained decoding)
```

## Technical Highlights

### Self-Hosted Inference
Runs against a self-hosted [vLLM](https://github.com/vllm-project/vllm) endpoint serving `Qwen/Qwen2.5-Coder-7B-Instruct` on a RunPod RTX 4090. Exposes an OpenAI-compatible API, making the inference backend swappable without changing application code.

### AST-Based Context Retrieval
Rather than naively dumping the entire codebase into the context window on every request, maia builds a structured representation of the project at startup using Python's `ast` module:

- **File structure dict** — maps each `.py` file to its imported modules, function definitions, call graph, and function bodies (with line ranges)
- **Reverse import map** — inverts the dependency graph to find files that import a given module
- **Layered entry point detection:**
  1. Explicit function mentions in user input (string match against known function names)
  2. Explicit file mentions (string match against known filenames)
  3. LLM fallback — passes file summaries + user query to the model with constrained JSON output

Once entry points are identified, only the relevant function bodies are pulled from dependency files rather than entire files, keeping context lean.

### Context Window Management
A recurring challenge with stateful LLM sessions is prompt token growth. maia addresses this with a two-layer approach:

- **Sliding window** — only the last N conversation turns are sent as message history on each request
- **Rolling summarization** — every N turns, older history is compressed into a natural language summary via a separate model call; the summary persists across sessions in `history.json`
- **/clear command** — compresses current history into the summary before wiping turns, so context resets don't lose important state

### Structured Output via Constrained Decoding
The file relevance query uses vLLM's `guided_json` parameter to enforce a JSON schema on the model output, eliminating parsing fragility from freeform LLM responses.

```python
extra_body={"guided_json": RelevantFiles.model_json_schema()}
```

## Project Structure

```
maia/
├── assistant.py        # REPL loop, main entry point
├── config.py           # Model config, tunable parameters
├── history_manager.py  # History/summary persistence (JSON)
├── openai_client.py    # vLLM client wrapper (singleton)
└── utils.py            # AST parsing, context retrieval, summarization
```

## Setup

### 1. Deploy vLLM on RunPod

Create a pod with:
- **GPU:** RTX 4090 (24GB VRAM)
- **Container image:** `vllm/vllm-openai:v0.6.6`
- **Start command:**
  ```
  --model Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --gpu-memory-utilization 0.95 --max-model-len 16384 --download-dir /workspace/models
  ```
- Attach a persistent network volume at `/workspace` to cache model weights (~15GB)

### 2. Configure the client

```bash
export RUNPOD_API_KEY=sk-your-pod-key
```

Update `base_url` in `openai_client.py` with your pod's proxy URL.

### 3. Install and run

```bash
pip install openai pydantic
python assistant.py
```

## Usage

```
> help me set up a postgres database        # baseline task
> add a users table with email and password # component task
> I'm getting a 502 error on /api/login     # debug task
> /clear                                    # compress and reset context
```

## Design Decisions

**Why self-hosted over API services?**
Full control over context window size, inference parameters, and model selection without artificial rate limits. Enables exploration of vLLM-specific features like constrained decoding.

**Why AST over embeddings for retrieval?**
For a single-developer codebase, deterministic graph traversal is faster, cheaper, and more predictable than vector similarity. The LLM fallback handles semantic queries that static analysis can't resolve.

**Why a sliding window + summary over full history?**
Sending full conversation history causes prompt tokens to grow linearly. Summarization compresses older context while preserving semantic continuity — a common pattern in production LLM applications.

## Planned Improvements

- Embeddings-based retrieval (sentence-transformers + faiss) for semantic file search
- Per-request observability logging (token counts, retrieval decisions, latency)
- Dev server log ingestion for in-context debugging
- Automatic file structure refresh on file change detection
