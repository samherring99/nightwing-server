from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    use_chat_template: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    num_particles: Optional[int] = 5
    beam_factor: Optional[int] = 1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    
    @property
    def messages(self):
        # Convert single prompt to messages format for consistency
        return [{"role": "user", "content": self.prompt}]

class CompletionResponse(BaseModel):
    reply: str