from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
import os
import asyncio

from openai import OpenAI, AsyncOpenAI
import mlflow
from mlflow import deployments
from starlette import Request, Response


from src.core.data.domain.base import DomainObject, DomainObjectError
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

    def get_provider(model_name: str) -> str:
        """Get the provider of the model"""
        return model_name.split("/")[0]

    def get_client(self) -> OpenAI | AsyncOpenAI:
        provider = self.get_provider(self.model_name)

        if provider == "openai":
            API_KEY = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "huggingface":
            API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        elif provider == "google":
            os.environ.get("GOOGLE_API_KEY")
        else:
            raise ValueError(f"Invalid provider: {provider}")

        client = config.openai(
            provider=provider, is_async=self.is_async, api_key=API_KEY
        )

        return client
