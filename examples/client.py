import requests
from typing import List, Dict, Union
from dataclasses import dataclass
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    role: Role
    content: str

class NightwingClient:
    
    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
    
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        max_tokens: int = 50,
        use_chat_template: bool = False
    ) -> str:

        formatted_messages = [
            msg if isinstance(msg, dict) else {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "use_chat_template": use_chat_template
        }
        
        response = requests.post(
            f"{self.host}/chat",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["reply"]
    
    def embed(self, text: str) -> List[float]:
        payload = {"text": text}
        
        response = requests.post(
            f"{self.host}/embed",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["embeddings"]

if __name__ == "__main__":
    client = NightwingClient("http://localhost:8000")
    
    messages = [
        Message(role=Role.ASSISTANT, content="You are a model that responds in pirate speak only"),
        Message(role=Role.USER, content="How are you?")
    ]

    response = client.chat(messages)
    print(f"Chat response: {response}")
    
    response = client.chat([
        {"role": "assistant", "content": "You are a model that responds in pirate speak only"},
        {"role": "user", "content": "How are you?"}
    ])
    print(f"Chat response: {response}")
    
    embedding = client.embed("nightwing")
    print(f"Embedding (first 5 dimensions): {embedding[:5]}")