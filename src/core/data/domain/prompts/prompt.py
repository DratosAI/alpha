from typing import Optional, List

from src.core.data.domain import DomainObject, DomainObjectAccessor, DomainSelector
from src.core.data.domain.comms.prompt import PromptSettings

class Prompt(DomainObject):
    """Prompt for a language model"""

    def __init__(self, 
            name: str,
            system_prompt: , 
            user_prompt: str, 
            **kwargs
            ):
        
        super().__init__(**kwargs)
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt



















class PromptSettingsSelector(DomainSelector):
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
