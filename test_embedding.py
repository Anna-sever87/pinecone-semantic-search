import sys

from config import load_settings, setup_logging
from embeddings import EmbeddingError, cosine_similarity, create_client, embed_texts


def main() -> int:
    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"Ошибка настроек: {e}")
        return 1
    setup_logging(settings.log_level)
    client = create_client(settings)

    phrases = [
        "Пицца Пепперони",
        "Я люблю пиццу",
        "Мне нравится пицца",
        "Двигатель автомобиля сломался",
    ]
    try:
        vectors = embed_texts(client, settings, phrases)
    except (EmbeddingError, ValueError) as e:
        print(f"Ошибка: {e}")
        return 1

    first = vectors[0]
    print(f"Модель: {settings.embedding_model}")
    print(f"Фраза: «{phrases[0]}»")
    print(f"Длина вектора: {len(first)}")
    print("Первые 5 чисел:", [round(x, 4) for x in first[:5]])

    print("\nБлизость по смыслу (косинусная, чем ближе к 1, тем ближе смысл):")
    pairs = [(1, 2), (0, 1), (1, 3)]
    for a, b in pairs:
        print(f"  «{phrases[a]}» и «{phrases[b]}»: {cosine_similarity(vectors[a], vectors[b]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())