from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
from uuid import uuid4


class MessageCreate(BaseModel):
    content: str
    role: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    timestamp: datetime = datetime.now(datetime.UTC)


class MessageUpdate(BaseModel):
    content: str
    edit_note: Optional[str] = None


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    role: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    timestamp: datetime
    version: int = 1
    edit_history: List[Dict[str, datetime | str]] = []

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
