from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import mlflow
from mlflow import deployments
from src.core.data.domain.base import DomainObject
from src.core.data.prompts import Prompt, PromptSettings
from src.core.data.structs.messages import Message
from api.config import config

deployments.get_client()
load_dotenv()


class LLM(DomainObject):
    """Base Multimodal Language Model that defines the interface for all language models"""

    def __init__(
        self,
        model_name: Optional[str] = Field(
            default="openai/gpt4o-mini",
            description="Name of the model in the form of a HuggingFace model name",
        ),
        prompt_settings: Optional[PromptSettings] = Field(
            default=None, description="Prompt settings to use for the model"
        ),
        history: Optional[List[str]] = Field(
            default=[], description="History of messages"
        ),
        is_async: Optional[bool] = Field(
            default=False, description="Whether to stream the output"
        ),
    ):
        self.super().__init__()
        self.model_name = model_name
        self.prompt_settings = prompt_settings
        self.history = history
        self.is_async = is_async

    def __call__(
        self,
        prompt: Prompt,
        messages: Optional[List[Message]],
        response_model: Optional[DomainObject],
    ) -> str:
        """Chat with the model"""

        client = self.get_client()

        mlflow.openai.autolog()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=prompt.settings.max_tokens,
            max_length=prompt.settings.max_length,
            temperature=prompt.settings.temperature,
            top_k=prompt.settings.top_k,
            top_p=prompt.settings.top_p,
            seed=prompt.settings.seed,
            stop_token=prompt.settings.stop_token,
            response_model=response_model,
            stream=self.is_async,
        )
        return response

    def get_client(self) -> OpenAI | AsyncOpenAI:
        client = config.openai_proxy(is_async=self.is_async)

        return client
