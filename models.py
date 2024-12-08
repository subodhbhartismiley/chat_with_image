from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class ImageUpload(BaseModel):
    image: str  # base64 encoded image
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    images: Optional[List[str]] = None