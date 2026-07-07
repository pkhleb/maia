import os
from openai import OpenAI
import json

class OpenAIClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("RUNPOD_API_KEY"),
            base_url = "https://oo98eur8mgzovb-8000.proxy.runpod.net/v1"
        )
    
    def chat(self, model, messages):
        print(json.dumps(messages, indent = 2))
        return self.client.chat.completions.create(model=model, messages=messages)
