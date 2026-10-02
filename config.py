import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Читаем настройки из файла .env (значения из .env главнее системных переменных)
load_dotenv(BASE_DIR / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    embedding_model: str
    embedding_dimension: int
    pinecone_api_key: str
    index_name: str
    cloud: str
    region: str
    log_level: str


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise RuntimeError(f"Не заполнена переменная {name}. Проверь файл .env.")
    return value


def _parse_dimension(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise RuntimeError("EMBEDDING_DIMENSION должен быть целым числом (для text-embedding-3-small это 1536).")
    if not 1 <= value <= 20000:
        raise RuntimeError("EMBEDDING_DIMENSION должен быть от 1 до 20000.")
    return value


def load_settings() -> Settings:
    index_name = os.getenv("PINECONE_INDEX_NAME", "hr-facts").strip()
    if not re.fullmatch(r"[a-z0-9-]{1,45}", index_name):
        raise RuntimeError(
            "PINECONE_INDEX_NAME может содержать только строчные латинские буквы, цифры и дефис (до 45 символов)."
        )
    return Settings(
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_dimension=_parse_dimension(os.getenv("EMBEDDING_DIMENSION", "1536").strip()),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        index_name=index_name,
        cloud=os.getenv("PINECONE_CLOUD", "aws").strip().lower(),
        region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )


def setup_logging(level_name: str) -> None:
    import logging
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)