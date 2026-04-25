from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path.cwd() / ".env")


class Settings:
    def __init__(self) -> None:
        self.groq_api_key: str | None = os.getenv("GROQ_API_KEY")
        self.agent_api_key: str | None = os.getenv("AGENT_API_KEY")


settings = Settings()
