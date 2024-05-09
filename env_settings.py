from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class EnvSettings(BaseSettings):
    # third party api key
    GOOGLE_API_KEY: str = None
    # SMS_TOKEN_KEY: str = 'eland'
    #
    # # maria
    # MARIA_USER: str = "root"
    # MARIA_PASSWORD: str = "eland1234"
    # MARIA_HOST: str = "aiskm-maria"
    # MARIA_PORT: str = "3306"
    #
    # # redis
    # CELERY_BROKER: str = "redis://aiskm-redis:6379/0"
    # CELERY_BACKEND: str = "redis://aiskm-redis:6379/0"
    # TIMEZONE: str = "Asia/Taipei"
    #
    # # milvus
    # MILVUS_HOST: str = "aiskm-milvus-standalone"
    # MILVUS_PORT: str = "19530"
    # MILVUS_COLLECTION_NAME: str = "aiskm"
    #
    # # mongodb
    # MONGO_HOST: str = "aiskm-mongo"
    # MONGO_PORT: str = "27017"
    # MONGO_DB_NAME: str = "aiskm"
    # MONGO_ROOT_PASSWORD: str = "eland1234"
    #
    # # tds
    # TDS_INDEX_DB_NAME: str = "FileDatasetView"
    # TDS_HOST: str = "aiskm-tds"
    #
    # # django
    # SECRET_KEY: str = None
    # USE_SQLITE: bool = False
    # DJANGO_PORT: int = 6060
    # DJANGO_HOST: str = "0.0.0.0"
    # DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

