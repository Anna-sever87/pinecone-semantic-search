import logging
import time
from typing import Any, Dict, List

from pinecone import Pinecone, ServerlessSpec

from config import Settings

logger = logging.getLogger("vector_store")

UPSERT_BATCH = 100  # Pinecone рекомендует записывать векторы пачками не больше 100


class StoreError(Exception):
    """Ошибка при работе с Pinecone."""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Читает поле и у объекта, и у словаря (в разных версиях SDK ответы устроены по-разному)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, None)
    if value is None:
        try:
            value = obj[key]
        except (TypeError, KeyError, IndexError):
            value = default
    return value


class VectorStore:
    """Клиент векторной базы Pinecone: создание индекса, запись, поиск, статистика."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
        except Exception as e:
            raise StoreError(f"Не удалось создать клиент Pinecone: {e}") from e
        self.index = None

    def ensure_index(self, timeout: float = 180.0) -> None:
        """Подключается к индексу; если его нет, создаёт и ждёт, пока он будет готов."""
        name = self.settings.index_name
        try:
            if not self.pc.has_index(name):
                logger.info("Индекса «%s» нет, создаю (размерность %s, метрика cosine)", name, self.settings.embedding_dimension)
                self.pc.create_index(
                    name=name,
                    dimension=self.settings.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.settings.cloud, region=self.settings.region),
                )
            description = self._wait_until_ready(name, timeout)
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(
                f"Не удалось подготовить индекс «{name}»: {e}. Проверьте PINECONE_API_KEY, облако и регион в .env."
            ) from e

        dimension = _get(description, "dimension")
        if dimension is not None and int(dimension) != self.settings.embedding_dimension:
            raise StoreError(
                f"Индекс «{name}» создан с размерностью {dimension}, а модель даёт {self.settings.embedding_dimension}. "
                "Удалите индекс в личном кабинете Pinecone или укажите другое имя в PINECONE_INDEX_NAME."
            )
        try:
            self.index = self.pc.Index(host=_get(description, "host"))
        except Exception as e:
            raise StoreError(f"Не удалось подключиться к индексу «{name}»: {e}") from e
        logger.info("Подключено к индексу «%s»", name)

    def _wait_until_ready(self, name: str, timeout: float) -> Any:
        deadline = time.time() + timeout
        while True:
            description = self.pc.describe_index(name)
            status = _get(description, "status", {})
            if _get(status, "ready", False):
                return description
            if time.time() > deadline:
                raise StoreError(f"Индекс «{name}» не стал готов за {int(timeout)} секунд. Попробуйте запустить скрипт ещё раз.")
            logger.info("Жду готовности индекса «%s»...", name)
            time.sleep(3)

    def _require_index(self) -> Any:
        if self.index is None:
            raise StoreError("Индекс не подключён: сначала вызовите ensure_index().")
        return self.index

    def upsert(self, items: List[Dict[str, Any]]) -> int:
        """Записывает или обновляет векторы: список словарей {id, values, metadata}. Возвращает число записанных."""
        index = self._require_index()
        for item in items:
            if len(item["values"]) != self.settings.embedding_dimension:
                raise StoreError(f"Вектор {item['id']} имеет длину {len(item['values'])}, ожидалась {self.settings.embedding_dimension}.")
        written = 0
        try:
            for start in range(0, len(items), UPSERT_BATCH):
                batch = items[start:start + UPSERT_BATCH]
                index.upsert(vectors=batch)
                written += len(batch)
        except Exception as e:
            raise StoreError(f"Не удалось записать векторы в Pinecone: {e}") from e
        logger.info("Записано векторов: %s", written)
        return written

    def count(self) -> int:
        """Сколько векторов в индексе."""
        try:
            stats = self._require_index().describe_index_stats()
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"Не удалось получить статистику индекса: {e}") from e
        return int(_get(stats, "total_vector_count", 0) or 0)

    def wait_for_count(self, expected: int, timeout: float = 60.0) -> int:
        """Ждёт, пока записанные векторы станут видны в статистике (в Pinecone это занимает несколько секунд)."""
        deadline = time.time() + timeout
        current = self.count()
        while current < expected and time.time() < deadline:
            time.sleep(2)
            current = self.count()
        return current

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Ищет top_k ближайших по смыслу векторов. Возвращает список {id, score, metadata}."""
        index = self._require_index()
        if not 1 <= top_k <= 100:
            raise ValueError("top_k должен быть от 1 до 100.")
        try:
            response = index.query(vector=vector, top_k=top_k, include_metadata=True)
        except Exception as e:
            raise StoreError(f"Не удалось выполнить поиск в Pinecone: {e}") from e
        results = []
        for match in _get(response, "matches", []) or []:
            results.append({
                "id": _get(match, "id"),
                "score": float(_get(match, "score", 0.0) or 0.0),
                "metadata": dict(_get(match, "metadata", {}) or {}),
            })
        return results

    def reset(self) -> None:
        """Удаляет все векторы из индекса (сам индекс остаётся)."""
        try:
            self._require_index().delete(delete_all=True)
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"Не удалось очистить индекс: {e}") from e
        logger.info("Все векторы удалены из индекса")