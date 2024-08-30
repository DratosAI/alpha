# @serve.deployment
from typing import List
from src.core.agents.base.base_agent import Agent
from src.core.data.prompts.prompt import Prompt
from src.core.models.language.base.llm import LLM
from src.core.tools.base.tool import Tool


class DataFrameAgent(Agent):
    """
    The base agent class.
    """
    def __init__(self, llm: LLM, tools: List[Tool]):
        super().__init__(llm, tools)

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
    
    def execute_action(self, action: str, tool: Tool, prompt: Prompt) -> str:
        """
        Execute the action based on the tool and prompt.
        """
        return "python"
    
__all__ = ["DataFrameAgent"]
