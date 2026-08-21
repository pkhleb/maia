import os
from dotenv import load_dotenv

load_dotenv()

from config import MODEL, N_RECENT, SUMMARY_EVERY_N, SYSTEM_PROMPT
from history_manager import load_history, save_history
from utils import build_system_prompt, maybe_summarize
from openai_client import OpenAIClient
import retlib

DB_PATH = ".retlib/index.db"


def main():
    print("Coding assistant ready. Ctrl+C or type 'exit' to quit.\n")

    # initialise and incrementally index the current project
    print("[retlib] indexing project...")
    retlib.init_db(DB_PATH)
    stats = retlib.index_directory(".", DB_PATH)
    print(f"[retlib] {stats['files']} files, {stats['symbols']} symbols, {stats['edges']} edges\n")

    history, summary = load_history()
    client = OpenAIClient.get_instance()

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            save_history(history, summary)
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            save_history(history, summary)
            print("Bye.")
            break
        if user_input.lower() == "/clear":
            summary = maybe_summarize(history, summary, client)
            history = []
            save_history(history, summary)
            print("[Context cleared. Summary preserved.]\n")
            continue
        if user_input.lower() == "/reindex":
            print("[retlib] reindexing...")
            stats = retlib.index_directory(".", DB_PATH)
            print(f"[retlib] {stats['files']} files, {stats['symbols']} symbols, {stats['edges']} edges\n")
            continue

        history.append({"role": "user", "content": user_input})
        system_prompt = build_system_prompt(summary, user_input, DB_PATH)

        response = client.chat(MODEL, [
            {"role": "system", "content": system_prompt}
        ] + history[-N_RECENT:])

        reply = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        history.append({"role": "assistant", "content": reply})
        summary = maybe_summarize(history, summary, client)

        print(reply)
        print(f"\n[prompt: {prompt_tokens} | completion: {completion_tokens} | total: {total_tokens}]\n")


if __name__ == "__main__":
    main()
