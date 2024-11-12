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

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embeddings: List[float]