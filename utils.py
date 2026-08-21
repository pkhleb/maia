from config import MODEL, N_RECENT, SYSTEM_PROMPT, SUMMARY_EVERY_N
from openai_client import OpenAIClient
import retlib


def build_project_context(user_input: str, db_path: str) -> str:
    """
    Retrieve relevant code context for the user's query using retlib.

    Replaces the old AST string-match + LLM fallback approach with a
    full hybrid retrieval pipeline: BM25 + embeddings → graph expansion
    → ranked symbols → token-budgeted context assembly.
    """
    results = retlib.retrieve(user_input, db_path, top_k=10, hops=2)

    if not results:
        return ""

    ctx = retlib.build_context(results, db_path)

    # print retrieval summary for observability
    print(f"\n[retlib] {retlib.summarize(ctx)}\n")

    return ctx.text


def build_system_prompt(summary: str, user_input: str, db_path: str) -> str:
    if not summary:
        summary = ""

    project_context = build_project_context(user_input, db_path)
    prompt = SYSTEM_PROMPT + "\n\n" + project_context

    if summary:
        prompt += "\n\nSummary of earlier conversation:\n" + summary

    return prompt


def maybe_summarize(history: list, current_summary: str, client: OpenAIClient) -> str:
    if len(history) % SUMMARY_EVERY_N != 0:
        return current_summary

    older = history[:-N_RECENT]
    if not older:
        return current_summary

    summary_prompt = "Summarize the following conversation history concisely, capturing key decisions, context, and progress:\n\n"
    for msg in older:
        summary_prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

    try:
        response = client.chat(MODEL, [{"role": "user", "content": summary_prompt}])
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing conversation: {e}")
        return current_summary
