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
from src.core.data.structs.messages.message import Message

class AgentStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    WAITING = "waiting"
    PROCESSING = "processing"


# @serve.deployment
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

    def __call__(self, prompt: Prompt, messages: List[Message]) -> str:
        """Chat with the model"""
        if self.is_async:
            
            action = self.infer_action(prompt,self.actions,self.tools)
            tool = self.choose_tool(self.tools)
            

            # TODO: Define Agent Prompt Template

            response = self.llm(
                prompt=prompt,
                messages=messages,
                response_model=self.tools,
                is_async=True
            )
        else:
            response = self.llm(
                prompt=prompt, 
                messages = messages,
                is_async = False
            )

        return response

    def get_status(self) -> AgentStatus:
        return self.status
    
    def infer_action(self, prompt: Prompt, tools: List[Tool]) -> str:
        """
        Infer the action to perform based on the prompt and tools.
        """
        return "python"
    
    def choose_tool(self, tools: List[Tool]) -> Tool:
        """
        Choose the tool to use based on the tools.
        """
        return tools[0]
    
__all__ = ["Agent"]
