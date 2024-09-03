from typing import Optional, List
from src.core.data.domain.base import DomainObject

# Remove these imports to break the circular dependency
# from src.core.tools.base import Tool
# from src.core.agents.roles.base import Role


class Message(DomainObject):
    content: Optional[str] = None
    """The contents of the message."""

    refusal: Optional[str] = None
    """The refusal message generated by the model."""

    role: str  # Change this to str instead of Role
    """The role of the author of this message."""

    tool_calls: Optional[List[dict]] = None  # Change this to List[dict] instead of List[Tool]
    """The tool calls generated by the model, such as function calls."""


__all__ = ["Message"]

