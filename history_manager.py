from config import HISTORY_FILE
import os
import json

def load_history() -> tuple:
    if os.path.exists(HISTORY_FILE):
        print(f"[Loading history from {HISTORY_FILE}]")
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            # Handle old format (plain list) gracefully
            if isinstance(data, list):
                print("[Migrating old history format]")
                return data, ""
            return data.get("history", []), data.get("summary", "")
    return [], ""

def save_history(history: list, summary: str):
    with open(HISTORY_FILE, 'w') as f:
        json.dump({"history": history, "summary": summary}, f, indent=2)
    print(f"[History saved to {HISTORY_FILE}]")

