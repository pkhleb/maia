from config import MODEL, HISTORY_FILE, N_RECENT, SUMMARY_EVERY_N, SYSTEM_PROMPT
from history_manager import load_history, save_history
from utils import build_system_prompt, build_project_context, maybe_summarize
from openai_client import OpenAIClient

def main():
    print("Coding assistant ready. Ctrl+C or type 'exit' to quit.\n")
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

        history.append({"role": "user", "content": user_input})
        system_prompt = build_system_prompt(summary, user_input, client)
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

