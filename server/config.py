from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Deepgram (STT)
    deepgram_api_key: str = ""

    # OpenAI (LLM)
    openai_api_key: str = ""

    # Cartesia (TTS)
    cartesia_api_key: str = ""

    # Google Calendar
    google_client_id: str = ""
    google_client_secret: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/callgenius"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Auth
    jwt_secret: str = "change-this-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    base_url: str = "http://localhost:8000"

    # Sentry
    sentry_dsn: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
