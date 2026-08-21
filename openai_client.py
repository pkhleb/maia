import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    _instance = None

    def __init__(self):
        api_key = os.environ.get("RUNPOD_API_KEY")
        base_url = os.environ.get("RUNPOD_BASE_URL")

        if not api_key:
            raise ValueError("RUNPOD_API_KEY not set — check your .env file")
        if not base_url:
            raise ValueError("RUNPOD_BASE_URL not set — check your .env file")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def chat(self, model, messages):
        print(json.dumps(messages, indent=2))
        return self.client.chat.completions.create(model=model, messages=messages)

    def chat_structured(self, model, messages, schema):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"guided_json": schema},
        )
