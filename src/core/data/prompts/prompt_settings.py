from typing import Optional, List, Any
from pydantic import Field
from src.core.data.domain.base import (
    DomainObject,
    DomainObjectFactory,
)
from api.config import Config


class PromptSettings(DomainObject):
    """Prompt Settings for a language model serving"""

    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens"
    )
    max_length: Optional[int] = Field(
        default=None, description="Maximum length of the prompt"
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature for text generation"
    )
    top_k: Optional[float] = Field(default=None, description="Top-k sampling parameter")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    seed: Optional[int] = Field(
        default=None, description="Random seed for text generation"
    )
    stop_token: Optional[str] = Field(
        default=None, description="Stop token for text generation"
    )

    class Meta:
        type: str = "PromptSettings"


class PromptSettingsFactory(DomainObjectFactory[PromptSettings]):
    @staticmethod
    def create_new_prompt(
        max_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop_token: Optional[str] = None,
    ) -> PromptSettings:
        return PromptSettings.create_new(
            max_tokens=max_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop_token=stop_token,
        )
