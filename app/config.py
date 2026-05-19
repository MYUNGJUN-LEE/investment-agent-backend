from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Investment Agent Backend"

    # If this is set, requests must include X-API-Key with the same value.
    backend_api_key: str | None = None

    # External APIs
    opendart_api_key: str | None = None

    naver_client_id: str | None = None
    naver_client_secret: str | None = None

    kis_app_key: str | None = None
    kis_app_secret: str | None = None
    kis_account_no: str | None = None
    kis_account_product_code: str | None = None
    kis_is_paper: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
