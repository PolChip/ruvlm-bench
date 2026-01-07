#!/usr/bin/env python3
"""
Проверка загруженных данных
"""
from datasets import load_from_disk
import matplotlib.pyplot as plt
import os


def main():
    print("ПРОВЕРКА ЗАГРУЖЕННЫХ ДАННЫХ")

    # Проверяем GQA-ru
    print("\n Проверка GQA-ru датасета:")
    try:
        gqa_path = "./data/gqa_ru_train_balanced_instructions_mini"
        if os.path.exists(gqa_path):
            gqa_data = load_from_disk(gqa_path)
            print(f" Загружено: {len(gqa_data)} примеров")

            # Покажем первый пример
            if len(gqa_data) > 0:
                example = gqa_data[0]
                print(f"\n Пример 1:")
                print(f"   Вопрос: {example['question']}")
                print(f"   Ответ: {example['answer']}")

                # Покажем изображение если есть
                if 'image' in example:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(example['image'])
                    plt.title(f"GQA-ru: {example['question'][:30]}...")
                    plt.axis('off')
                    plt.savefig("./results/gqa_example.png", dpi=150, bbox_inches='tight')
                    print(f"     Изображение сохранено: results/gqa_example.png")
        else:
            print(f" Путь не найден: {gqa_path}")
    except Exception as e:
        print(f" Ошибка: {e}")

    # Проверяем MMBENCH-ru
    print("\n Проверка MMBENCH-ru датасета:")
    try:
        mmbench_path = "./data/mmbench_ru_mini"
        if os.path.exists(mmbench_path):
            mmbench_data = load_from_disk(mmbench_path)
            print(f" Загружено: {len(mmbench_data)} примеров")

            # Покажем первый пример
            if len(mmbench_data) > 0:
                example = mmbench_data[0]
                print(f"\n Пример 1:")
                print(f"   Вопрос: {example['question']}")
                print(f"   Ответ: {example['answer']}")
                print(f"   Варианты: A) {example['A']}, B) {example['B']}")

                # Покажем изображение если есть
                if 'image' in example:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(example['image'])
                    plt.title(f"MMBENCH-ru: {example['question'][:30]}...")
                    plt.axis('off')
                    plt.savefig("./results/mmbench_example.png", dpi=150, bbox_inches='tight')
                    print(f"     Изображение сохранено: results/mmbench_example.png")
        else:
            print(f" Путь не найден: {mmbench_path}")
    except Exception as e:
        print(f" Ошибка: {e}")

    print(" ПРОВЕРКА ЗАВЕРШЕНА")
    print("\n Далее: python scripts/train_simple.py")


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    main()