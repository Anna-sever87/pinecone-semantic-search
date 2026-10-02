import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import BASE_DIR, load_settings, setup_logging
from embeddings import EmbeddingError, create_client, embed_text
from vector_store import StoreError, VectorStore

logger = logging.getLogger("search")

# Запросы сформулированы другими словами, чем фразы в базе: поиск идёт по смыслу
DEFAULT_QUERIES = [
    "Сколько дней отдыха мне положено в году?",
    "Можно ли работать из дома?",
    "Кто поможет новичку освоиться в первые дни?",
    "Как получить документ о том, что я здесь работаю?",
    "Оплачивает ли компания учёбу?",
]


def format_report(results: Dict[str, List[dict]], model: str, index_name: str) -> str:
    """Собирает отчёт о поиске в формате Markdown."""
    lines = [
        "# Результаты поиска по смыслу",
        "",
        f"Индекс Pinecone: `{index_name}`, модель эмбеддингов: `{model}`.",
        "Чем выше score (косинусная близость), тем ближе смысл фразы к запросу.",
        "",
    ]
    for number, (query, matches) in enumerate(results.items(), start=1):
        lines += [f"## Запрос {number}: {query}", "", "| № | ID | score | тема | текст |", "|---|---|---|---|---|"]
        for rank, match in enumerate(matches, start=1):
            meta = match["metadata"]
            text = str(meta.get("text", "")).replace("|", "\\|")
            topic = str(meta.get("topic", "")).replace("|", "\\|")
            lines.append(f"| {rank} | {match['id']} | {match['score']:.4f} | {topic} | {text} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Поиск по смыслу в векторной базе Pinecone.")
    parser.add_argument("queries", nargs="*", help="свои запросы (если не указаны, берутся готовые)")
    parser.add_argument("--top-k", type=int, default=3, help="сколько ближайших результатов показать (по умолчанию 3)")
    parser.add_argument("--report", default="results/search_results.md", help="куда сохранить отчёт")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 20:
        print("Ошибка: --top-k должен быть от 1 до 20.")
        return 1
    queries = [q.strip() for q in args.queries if q.strip()] or DEFAULT_QUERIES

    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"Ошибка настроек: {e}")
        return 1
    setup_logging(settings.log_level)

    results: Dict[str, List[dict]] = {}
    try:
        client = create_client(settings)
        store = VectorStore(settings)
        store.ensure_index()
        if store.count() == 0:
            print("В индексе пока нет векторов. Сначала запустите: python load_data.py")
            return 1
        for query in queries:
            vector = embed_text(client, settings, query)
            matches = store.search(vector, top_k=args.top_k)
            results[query] = matches
            print(f"\nЗапрос: {query}")
            if not matches:
                print("  совпадений не найдено")
            for rank, match in enumerate(matches, start=1):
                meta = match["metadata"]
                print(f"  {rank}. {match['id']}  score {match['score']:.4f}  [{meta.get('topic', '')}] {meta.get('text', '')}")
    except (EmbeddingError, StoreError, ValueError) as e:
        logger.error("%s", e)
        print(f"Ошибка: {e}")
        return 1
    except Exception as e:
        logger.exception("Непредвиденная ошибка")
        print(f"Непредвиденная ошибка: {e}")
        return 1

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = Path(BASE_DIR) / report_path
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_report(results, settings.embedding_model, settings.index_name), encoding="utf-8")
    except OSError as e:
        print(f"Не удалось сохранить отчёт: {e}")
        return 1
    print(f"\nОтчёт сохранён: {report_path.relative_to(BASE_DIR) if report_path.is_relative_to(BASE_DIR) else report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())