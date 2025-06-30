from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Travel Planner"
    flight_api_key: str | None = None          # đọc từ .env nếu có
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl: int = 86400

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # class Config:
    #     env_file = ".env"

