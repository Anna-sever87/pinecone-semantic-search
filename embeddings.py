import logging
import math
from typing import List

from openai import OpenAI, OpenAIError

from config import Settings

logger = logging.getLogger("embeddings")

BATCH_SIZE = 100  # сколько фраз отправляем в модель за один запрос


class EmbeddingError(Exception):
    """Не удалось получить эмбеддинг."""


def create_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=60.0, max_retries=2)


def embed_texts(client: OpenAI, settings: Settings, texts: List[str]) -> List[List[float]]:
    """Превращает список фраз в список векторов (эмбеддингов) с помощью Embedding API."""
    cleaned = [str(text).strip() for text in texts]
    if any(not text for text in cleaned):
        raise ValueError("Пустую фразу нельзя превратить в вектор.")
    vectors: List[List[float]] = []
    for start in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[start:start + BATCH_SIZE]
        try:
            response = client.embeddings.create(input=batch, model=settings.embedding_model)
        except OpenAIError as e:
            raise EmbeddingError(f"Ошибка запроса к Embedding API (модель {settings.embedding_model}): {e}") from e
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingError("Embedding API вернул не столько векторов, сколько было фраз.")
        vectors.extend([list(item.embedding) for item in items])
    for vector in vectors:
        if len(vector) != settings.embedding_dimension:
            raise EmbeddingError(
                f"Модель {settings.embedding_model} вернула вектор длиной {len(vector)}, "
                f"а в настройках указано {settings.embedding_dimension}. Проверьте EMBEDDING_DIMENSION в .env."
            )
    logger.info("Получено векторов: %s (размерность %s)", len(vectors), settings.embedding_dimension)
    return vectors


def embed_text(client: OpenAI, settings: Settings, text: str) -> List[float]:
    """Вектор для одной фразы."""
    return embed_texts(client, settings, [text])[0]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Косинусная близость двух векторов: чем ближе к 1, тем ближе смысл."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0