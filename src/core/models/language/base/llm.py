from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
import os
import asyncio 

from openai import OpenAI, AsyncOpenAI
import mlflow
from starlette import Request, Response
from tranformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.core.data.domain import PromptSettings, DomainObject, DomainObjectError

load_dotenv()


class LLM(DomainObject):
    """Base Multimodal Language Model that defines the interface for all language models"""

    def __init__(
        self,
        model_name: Optional[str] = Field(
            default="openai/gpt4o-mini",
            description="Name of the model in the form of a HuggingFace model name"
        ),
        prompt_settings: Optional[PromptSettings] = Field(
            default=None,
            description="Prompt settings to use for the model"
        ),
        history: Optional[List[str]] = Field(
            default=[],
            description="History of messages"
        ),
        is_async: Optional[bool] = Field(
            default=False,
            description="Whether to stream the output"
        )
    ):
        self.super().__init__()
        self.model_name = model_name
        self.prompt_settings = prompt_settings
        self.history = history
        self.is_async = is_async

        mlflow.openai.autolog()

    def __call__(self, request: Request) -> Response:
        """Chat with the model"""
        if self.stream:
            return self.__async_call__(request)
        else:
            return self.__sync_call__(request)

    def __sync_call__(self, request: Request) -> Response:
        """Chat with the model"""
        pass

    async def __async_call__(self, request: Request) -> Response:
        """Chat with the model"""
        self.prompt = request.prompt
        self.messages = request.messages
        self.model_name = request.model_name
        self.prompt.system_prompt = request.system_prompt
        self.prompt.settings = request.prompt.settings

        # TODO: Figure out how to
        # TODO: Add support for other models

        await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=self.prompt.max_tokens,
            max_length=self.prompt.max_length,
            temperature=self.prompt.temperature,
            top_k=self.prompt.top_k,
            top_p=self.prompt.top_p,
            seed=self.prompt.seed,
            stop_token=self.prompt.stop_token,
            response_model=request.response_model,
            stream=request.stream,
        )

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

        if self.stream:
            self.client = AsyncOpenAI(
                host=os.environ.get("OPENAI_API_BASE"), 
                api_key=API_KEY)
        else:
            self.client = OpenAI(
                host=os.environ.get("OPENAI_API_BASE"), 
                api_key=API_KEY)
