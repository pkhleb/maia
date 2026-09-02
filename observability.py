import json

class Observer:
    PRESETS = {
        "full":     {"tokens", "messages", "critique", "retrieval", "context", "memory"},
        "minimal":  {"tokens"},
        "rag": {"retrieval", "context", "tokens"},
        "refine": {"critique", "tokens"},
    }

    def __init__(self, flags: list[str]):
        self._active = set()
        for f in flags:
            self._active |= self.PRESETS.get(f, {f})

    def __call__(self, flag: str, label: str, content):
        if flag not in self._active:
            return
        print(f"\n[{label}]")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2))
        else:
            print(content)
        print()
