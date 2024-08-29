from enum import Enum


class Role(str, Enum):
    agent = "agent"
    user = "user"
    system = "system"
