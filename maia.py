#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
import argparse
from model import call_model, ModelResult
from context_ret_and_gen import self_refine, rag
from observability import Observer

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="maia — LLM coding assistant")
parser.add_argument("-d", "--dir", type=str, default=None,
                    help="working directory")
parser.add_argument("-o", "--obs", nargs="*", default=[], metavar="COMPONENT",
                    help="observability: full, minimal, rag, refine, "
                         "or individual: tokens messages critique retrieval context memory")
parser.add_argument("--refine", type=int, default=0, metavar="N",
                    help="run self-refine with N iterations (default: 0 = disabled)")
args = parser.parse_args()


working_dir = os.path.abspath(args.dir) if args.dir else None

# ── observability ─────────────────────────────────────────────────────────────
obs = Observer(args.obs)

# ── history ───────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(working_dir, ".maia_history.json") if working_dir else ".maia_history.json"

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ── messages ──────────────────────────────────────────────────────────────────
def build_messages(history: list, working_dir: str | None) -> list:
    excluded = {".env", ".venv", "__pycache__", ".git", ".gitignore", ".maia_index.db"}

    files = {}

    print(working_dir)
    if working_dir:
        for root, dirs, filenames in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in excluded]
            for filename in filenames:
                if filename in excluded:
                    continue
                path = os.path.join(root, filename)

                with open(path, "r", encoding="utf-8") as f:
                    print(path)
                    files[path] = f.read()

        DEFAULT_SETTINGS = {
            "name": "flat-512",
            "chunker": "fixed",
            "chunk_size": 512,
            "overlap": 64,
            "embedder": "all-MiniLM-L6-v2",
            "top_k":3,
        }
        system = rag(files=files, query=history[-1], settings=DEFAULT_SETTINGS, obs=obs)
    else:
        system = ""
    messages = [{"role": "system", "content": system}] + list(history)
    return list(messages)


# ── response ──────────────────────────────────────────────────────────────────
def get_response(messages: list) -> tuple[str, list[ModelResult]]:
    """
    Call the model and optionally refine the output.
    Returns (final_reply, all_results) for token accounting.
    """
    result = call_model(messages)
    results = [result]

    obs("messages", "outgoing messages", messages)

    if args.refine > 0:
        obs("critique", "self_refine — initial response", result.reply)
        refined_reply, intermediate = self_refine(result.reply, iterations=args.refine, obs=obs)
        results += intermediate
        return refined_reply, results

    return result.reply, results

# ── main ──────────────────────────────────────────────────────────────────────
history = load_history()

if history:
    print(f"[loaded {len(history)} messages from {HISTORY_FILE}]")
if working_dir:
    print(f"[working directory: {working_dir}]")
if obs:
    print(f"[observability: {', '.join(sorted(obs._active))}]")
if args.refine > 0:
    print(f"[self-refine: {args.refine} iteration(s)]")

print("maia. Ctrl+C or type 'exit' to quit.\n")

while True:
    try:
        user_input = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        save_history(history)
        break

    if not user_input:
        continue
    if user_input.lower() == "exit":
        save_history(history)
        print("Bye.")
        break

    history.append({"role": "user", "content": user_input})
    messages = build_messages(history, working_dir)
    reply, results = get_response(messages)
    history.append({"role": "assistant", "content": reply})
    save_history(history)

    prompt_tokens     = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    total_tokens      = sum(r.total_tokens for r in results)

    print(reply)
    obs("tokens", "token usage",
        f"prompt: {prompt_tokens} | completion: {completion_tokens} | total: {total_tokens}")

