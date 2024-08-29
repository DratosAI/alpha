from pydantic import BaseModel


class CustomModel(BaseModel):
    """
    A base class for custom models in our application.
    Inherits from Pydantic's BaseModel and can be extended with additional functionality.
    """
    
    class Config:
        # Allow population by field name
        allow_population_by_field_name = True
        # Validate assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True
        # Allow arbitrary types
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """
        Override the dict method to exclude None values by default.
        """
        kwargs.setdefault('exclude_none', True)
        return super().dict(*args, **kwargs)