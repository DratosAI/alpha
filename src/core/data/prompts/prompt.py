from typing import Optional, List

from pydantic import Field

from api.config.config import Config
from src.core.data.domain import DomainObject, DomainObjectAccessor, DomainObjectSelector
from src.core.data.domain.base.domain_object import DomainObjectFactory
from src.core.data.prompts.prompt_settings import PromptSettings


class SystemPrompt(DomainObject):
    """System prompt for a language model"""
    name: str = Field(..., description="Name of the system prompt")
    system_prompt: str = Field(..., description="System prompt for the prompt")

    __tablename__ = "system_prompts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UserPrompt(DomainObject):
    """User prompt for a language model"""
    name: str = Field(..., description="Name of the user prompt")
    user_prompt: str = Field(..., description="User prompt for the prompt")

    __tablename__ = "user_prompts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PromptSettingsFactory(DomainObjectFactory):
    @staticmethod
    def create_new_prompt_settings(**kwargs) -> PromptSettings:
        return PromptSettings(**kwargs)


class PromptFactory(DomainObjectFactory):
    @staticmethod
    def create_new_prompt(**kwargs) -> PromptSettings:
        return PromptSettings(**kwargs)


class Prompt(DomainObject):
    """Prompt for a language model"""
    name: str = Field(..., description="Name of the prompt")
    system_prompt: SystemPrompt = Field(..., description="System prompt for the prompt")
    user_prompt: UserPrompt = Field(..., description="User prompt for the prompt")
    settings: PromptSettings = Field(..., description="Settings for the prompt")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def __init__(
    #     self,
    #     name: str,
    #     system_prompt: SystemPrompt,
    #     user_prompt: UserPrompt,
    #     settings: PromptSettings,
    #     **kwargs
    # ):

    #     super().__init__(**kwargs)
    #     self.name = name
    #     self.system_prompt = system_prompt
    #     self.user_prompt = user_prompt
    #     self.settings = settings


class PromptSettingsSelector(DomainObjectSelector):
    @staticmethod
    def by_name(name: str) -> str:
        return f"{PromptSettingsSelector.base_query('prompt_settings')} WHERE name = '{name}'"

    @staticmethod
    def by_system_prompt(system_prompt: str) -> str:
        return f"{PromptSettingsSelector.base_query('prompt_settings')} WHERE system_prompt = '{system_prompt}'"


class PromptAccessor(DomainObjectAccessor):
    def __init__(self, config: Config):
        super().__init__(config)

    async def get_prompt_by_name(self, name: str) -> Optional[PromptSettings]:
        query = PromptSettingsSelector.by_name(name)
        df = self.config.daft().sql(query)
        if df.count().collect()[0] > 0:
            prompt_data = df.collect().to_pydict()[0]
            return PromptFactory.create_from_data(prompt_data, "Prompt")
        return None

    async def get_prompts_by_system_prompt(self, system_prompt: str) -> List[PromptSettings]:
        query = PromptSettingsSelector.by_system_prompt(system_prompt)
        df = self.config.daft().sql(query)
        prompts_data = df.collect().to_pydict()
        return [PromptFactory.create_from_data(data, "Prompt") for data in prompts_data]

    async def create_prompt(self, **prompt_data) -> PromptSettings:
        new_prompt = PromptSettingsFactory.create_new_prompt(**prompt_data)
        # Here you would typically insert the new prompt into the Iceberg table
        # For this example, we'll just return the new prompt
        return new_prompt

    async def update_prompt(self, prompt: PromptSettings) -> PromptSettings:
        prompt.update_timestamp()
        # Here you would typically update the prompt in the Iceberg table
        # For this example, we'll just return the updated prompt
        return prompt

    async def delete_prompt(self, prompt_id: str) -> bool:
        # Here you would typically delete the prompt from the Iceberg table
        # For this example, we'll just return True
        return await self.delete("prompts", prompt_id, "Prompt")

__all__ = [
    "Prompt",
    "PromptAccessor",
    "UserPrompt",
    "SystemPrompt",
    "PromptFactory",
    "PromptAccessor",
    "PromptSettingsFactory",
    "PromptSettingsSelector",
]