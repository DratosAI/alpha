from enum import Enum
from typing import List, Optional
import daft
from pydantic import Field
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

from src.core.data.prompts.prompt import Prompt
from src.core.tools.base.tool import Tool
from src.core.data.domain.base.domain_object import DomainObject
from src.core.models.language.base.llm import LLM


class AgentStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    WAITING = "waiting"
    PROCESSING = "processing"


@serve.deployment
class Agent(DomainObject):
    """
    The base agent class.
    """
    def __init__(self):
        name: Optional[str] = Field(
            ...,
            description="The name of the agent", 
        ),
        status: AgentStatus = Field(
            default="Idle",
            description="The status of the agent"
        ),
        llm: Optional[LLM] = Field(
            default=LLM(),
            description="The Multimodal Model"
        ),
        tools: Optional[List[Tool]] = Field(
            default=None,
            description="The tools that the agent can use"
        )
        is_async: Optional[bool] = Field(
            default=False,
            description="Use asynchrony (i.e. for streaming)."
        )

        super().__init__()
        self.name = name
        self.llm = llm
        self.tools = tools
        self.status = status
        self.is_async = is_async

    def __call__(self, prompt: Prompt):
        response = 
        return response

    def __call__(self, prompt: Prompt, messages: Messages) -> str:
        """Chat with the model"""
        if self.is_async:
            return self.__async_call__(
                prompt=prompt, 
                messages = messages,
                is_async = True
        )
        else:
            return self.__sync_call__(
                prompt=prompt, 
                messages = messages,
                is_async = False
        )

    def __sync_call__(self, prompt: Prompt) -> str:
        """Chat with the model"""
        response = self.llm.chat(prompt=prompt, 
                            messages = messages,
                            is_async = False
        )
        return response


    async def __async_call__(self, prompt: Prompt, messages: Messages ) -> str:
        """Chat with the model"""

        await self.llm.chat(prompt=prompt, 
                            messages = messages,
                            is_async = True
        )

    def get_status(self) -> AgentStatus:
        return self.status
