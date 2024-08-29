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
        stream: Optional[bool] = Field(
            default=False, 
            description="Whether to stream the output"
        )
    ):
        self.super().__init__()
        
        self.model_name = model_name
        self.prompt = prompt
        self.history = history
        self.stream = stream

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
        self.prompt = prompt

        prompt.system_prompt

        #TODO: Figure out how to
        #TODO: Add support for other models

        await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            max_tokens= prompt.max_tokens,
            max_length= prompt.max_length,
            temperature=prompt.temperature,
            top_k=prompt.top_k,
            top_p= prompt.top_p,
            seed=prompt.seed,
            stop_token=prompt.stop_token,
            response_model=request.response_model,
            stream=request.stream,
            messages=[
                {"role": "user", "content": f"Extract: `{data.query}`"},
            ],
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

        if self.stream:
            self.client = AsyncOpenAI(host=os.environ.get("OPENAI_API_BASE"), api_key=API_KEY)
        else:
            self.client = OpenAI(host=os.environ.get("OPENAI_API_BASE"), api_key=API_KEY)


