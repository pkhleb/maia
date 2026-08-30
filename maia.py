#!/usr/bin/env python3
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="qwen.py — minimal LLM CLI")
parser.add_argument("-d", "--dir", type=str, default=None, help="working directory")
parser.add_argument(
    "-o", "--obs",
    nargs="*",
    default=[],
    metavar="COMPONENT",
    help="observability components or presets: full, minimal, rag, refine, "
         "or individual: tokens messages critique retrieval context memory"
)
args = parser.parse_args()

working_dir = os.path.abspath(args.dir) if args.dir else None

# ── observability ─────────────────────────────────────────────────────────────
PRESETS = {
    "full":    {"tokens", "messages", "critique", "retrieval", "context", "memory"},
    "minimal": {"tokens"},
    "rag":     {"retrieval", "context", "tokens"},
    "refine":  {"critique", "tokens"},
}

def parse_obs(values: list[str]) -> set[str]:
    components = set()
    for v in values:
        if v in PRESETS:
            components |= PRESETS[v]
        else:
            components.add(v)
    return components

_obs = parse_obs(args.obs)

def obs(flag: str, label: str, content):
    if flag not in _obs:
        return
    print(f"\n[{label}]")
    if isinstance(content, (dict, list)):
        print(json.dumps(content, indent=2))
    else:
        print(content)
    print()

# ── client ────────────────────────────────────────────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("RUNPOD_API_KEY"),
    base_url=os.environ.get("RUNPOD_BASE_URL"),
)

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# ── history ───────────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(working_dir, ".qwen_history.json") if working_dir else ".qwen_history.json"

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
    messages = history
    if "messages" in _obs:
        print(messages)
    return messages

# ── model call ────────────────────────────────────────────────────────────────
def call_model(messages: list) -> tuple[str, int, int, int]:
    obs("messages", "outgoing messages", messages)
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    obs("tokens", "token usage", f"prompt: {prompt_tokens} | completion: {completion_tokens} | total: {total_tokens}")
    return reply, prompt_tokens, completion_tokens, total_tokens

# ── main ──────────────────────────────────────────────────────────────────────
history = load_history()

if history:
    print(f"[loaded {len(history)} messages from {HISTORY_FILE}]")
if working_dir:
    print(f"[working directory: {working_dir}]")
if _obs:
    print(f"[observability: {', '.join(sorted(_obs))}]")

print("Direct Qwen CLI. Ctrl+C or type 'exit' to quit.\n")

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
    reply, prompt_tokens, completion_tokens, total_tokens = call_model(messages)
    history.append({"role": "assistant", "content": reply})
    save_history(history)

    print(reply)
    if "tokens" not in _obs:
        print(f"\n[prompt: {prompt_tokens} | completion: {completion_tokens} | total: {total_tokens}]\n")
