import argparse
import logging
import sys
from typing import List, Optional

from config import load_settings, setup_logging
from embeddings import EmbeddingError, create_client, embed_texts
from phrases import PHRASES
from vector_store import StoreError, VectorStore

logger = logging.getLogger("load_data")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка фраз в векторную базу Pinecone.")
    parser.add_argument("--reset", action="store_true", help="сначала удалить все векторы из индекса")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"Ошибка настроек: {e}")
        return 1
    setup_logging(settings.log_level)

    try:
        client = create_client(settings)
        store = VectorStore(settings)
        store.ensure_index()
        if args.reset:
            store.reset()

        texts = [item["text"] for item in PHRASES]
        logger.info("Создаю эмбеддинги для %s фраз", len(texts))
        vectors = embed_texts(client, settings, texts)

        items = [
            {
                "id": f"phrase-{number}",  # ID - порядковый номер фразы: повторный запуск перезаписывает, а не дублирует
                "values": vector,
                "metadata": {"text": phrase["text"], "topic": phrase["topic"]},
            }
            for number, (phrase, vector) in enumerate(zip(PHRASES, vectors), start=1)
        ]
        written = store.upsert(items)
        in_index = store.wait_for_count(len(items))
    except (EmbeddingError, StoreError, ValueError) as e:
        logger.error("%s", e)
        print(f"Ошибка: {e}")
        return 1
    except Exception as e:
        logger.exception("Непредвиденная ошибка")
        print(f"Непредвиденная ошибка: {e}")
        return 1

    print(f"Загружено фраз: {written}. Векторов в индексе «{settings.index_name}»: {in_index}.")
    if in_index < len(items):
        print("Pinecone ещё обновляет статистику: подождите минуту и запустите поиск.")
    return 0


if __name__ == "__main__":
    sys.exit(main())