import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load env variables from .env if present
load_dotenv()

class Settings(BaseModel):
    """Pydantic model representing app configuration settings with automated validation."""
    port: int = Field(default=8000)
    huggingfacehub_api_token: Optional[str] = Field(default=None)

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1 or v > 65535:
            raise ValueError("port must be in the range 1-65535")
        return v

# Instantiate settings
settings = Settings(
    port=int(os.getenv("PORT", "8000")),
    huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or None
)

# Explicitly populate HF_TOKEN in the environment so underlying HF hub libraries find it
if settings.huggingfacehub_api_token:
    os.environ["HF_TOKEN"] = settings.huggingfacehub_api_token


