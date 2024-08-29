from pydantic import Field

from src.core.data.structs.documents.base import Document

from typing import ClassVar


class MultimodalDocument(Document):
    """Multimodal document"""

    def __init__(self, 
            content: 
        **kwargs
        ):

        super().__init__(**kwargs)
        

        self.Type: ClassVar[str] = "MultimodalDocument"

    
    def __str__(self,cls):
        return f"MultimodalDocument(content={self.content})"

    def __repr__(self):
        return f"MultimodalDocument(content={self.content})"

    def __eq__(self, other):
        