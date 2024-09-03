import os
from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import mlflow
from mlflow import deployments
from src.core.data.domain.base import DomainObject
from src.core.data.prompts import Prompt, PromptSettings
from src.core.data.structs.messages.message import Message
from api.config import config

load_dotenv()

deployments.get_deploy_client()


class LLM(DomainObject):
    """Base Multimodal Language Model that defines the interface for all language models"""

    model_name: str = Field(
        default="openai/gpt4o-mini",
        description="Name of the model in the form of a HuggingFace model name",
    )
    prompt_settings: PromptSettings = Field(
        default=None, description="Prompt settings to use for the model"
    )
    history: Optional[List[str]] = Field(default=[], description="History of messages")
    is_async: bool = Field(default=False, description="Whether to stream the output")
    api_key: str = Field(
        default=os.environ.get("OPENAI_API_KEY"),
        description="API key to use for the model",
    )

    def __init__(self, **data):
        super().__init__(**data)  # Corrected this line

    # def __init__(
    #     self,
    #     model_name: Optional[str] = Field(
    #         default="openai/gpt4o-mini",
    #         description="Name of the model in the form of a HuggingFace model name",
    #     ),
    #     prompt_settings: Optional[PromptSettings] = Field(
    #         default=None, description="Prompt settings to use for the model"
    #     ),
    #     history: Optional[List[str]] = Field(
    #         default=[], description="History of messages"
    #     ),
    #     is_async: Optional[bool] = Field(
    #         default=False, description="Whether to stream the output"
    #     ),
    # ):
    #     super().__init__()  # Corrected this line
    #     self.model_name = model_name
    #     self.prompt_settings = prompt_settings
    #     self.history = history
    #     self.is_async = is_async

    def __call__(
        self,
        prompt: Prompt,
        messages: Optional[List[Message]],
        response_model: Optional[DomainObject],
    ) -> str:
        """Chat with the model"""

        client = self.get_client()
        client.api_key = self.api_key

        mlflow.openai.autolog()
        response = client.chat.completions.create(
            response_format=response_model,
            model=self.model_name,
            messages=messages,
            max_tokens=self.prompt_settings.max_tokens,
            temperature=self.prompt_settings.temperature,
            top_p=self.prompt_settings.top_p,
            seed=self.prompt_settings.seed,
            stream=self.is_async,
        )
        return response

    def get_client(self) -> OpenAI | AsyncOpenAI:
        client = config.config.openai_proxy(
            is_async=self.is_async, api_key=self.api_key
        )

        return client
