from src.core.data.structs.documents.multimodal.base import MultimodalDocument

class MarkdownDocument(MultimodalDocument):
    """Markdown document"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.type = "markdown"

    def __str__(self):
        return f"MarkdownDocument(content={self.content})"

    def __repr__(self):
        return f"MarkdownDocument(content={self.content})"

    def __eq__(self, other):
        