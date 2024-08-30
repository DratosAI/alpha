from enum import Enum
from typing import List, Optional
import daft
from pydantic import Field
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

from src.core.data.prompts.prompt import Prompt
from src.core.data.prompts.prompt_settings import PromptSettings
from src.core.tools.base.tool import Tool, ToolTypes
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
    name: str = Field(..., description="The name of the agent")
    status: AgentStatus = Field(default="Idle", description="The status of the agent")
    llm: LLM = Field(default=LLM(), description="The Multimodal Model")
    tools: Optional[List[Tool]] = Field(default=None, description="The tools that the agent can use")
    is_async: bool = Field(default=False, description="Use asynchrony (i.e. for streaming).")
    prompt: Optional[Prompt] = Field(default=None, description="The prompt to use for the agent")
    actions: Optional[List[str]] = Field(default=None, description="The actions that the agent can perform")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __call__(self, prompt: Prompt, messages: List[Message]) -> str:
        """Chat with the model"""
        if self.is_async:
            
            action = self.infer_action(prompt, self.actions, self.tools)
            tool = self.choose_tool(self.tools)

            # TODO: Define Agent Prompt Template

            response = self.llm(
                prompt=prompt,
                messages=messages,
                response_model=self.tools
            )
        else:
            response = self.llm(
                prompt=prompt, 
                messages = messages
            )

        return response

    def get_status(self) -> AgentStatus:
        return self.status
    
    def infer_action(self, prompt: Prompt, actions: List[str], tools: List[Tool]) -> str:
        """
        Infer the action to perform based on the prompt and tools.
        """
        return "python"
    
    def choose_tool(self, tools: List[Tool]) -> Tool:
        """
        Choose the tool to use based on the tools.
        """
        # TODO: Implement tool selection logic
        return Tool(
            name="python",
            desc="Python tool",
            type=ToolTypes.python,
            function=lambda x: x + 1
        )
    
__all__ = ["Agent"]
