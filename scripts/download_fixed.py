#!/usr/bin/env python3
"""
Исправленная загрузка датасетов
"""
from datasets import load_dataset
import os


def main():
    print("Загрузка датасетов (исправленная версия)")

    # Создаем папку для данных
    os.makedirs("./data", exist_ok=True)

    # 1. Загрузка GQA-ru
    print("\n1. Загрузка GQA-ru...")
    try:
        gqa_dataset = load_dataset(
            "deepvk/gqa-ru",
            "train_balanced_instructions",
            split="train[:500]"  # Уменьшаем для теста
        )

        gqa_dataset.save_to_disk("./data/gqa_ru")
        print(f"GQA-ru загружен: {len(gqa_dataset)} примеров")

        # Смотрим на структуру данных
        if len(gqa_dataset) > 0:
            example = gqa_dataset[0]
            print("\nСтруктура примера GQA-ru:")
            for key in example.keys():
                value = example[key]
                if key == 'image':
                    print(f"   {key}: {type(value)}")
                elif isinstance(value, str):
                    print(f"   {key}: {value[:50]}...")
                else:
                    print(f"   {key}: {type(value)}")

    except Exception as e:
        print(f"Ошибка GQA-ru: {e}")

    # 2. Загрузка MMBENCH-ru
    print("\n2. Загрузка MMBENCH-ru...")

    try:
        # Сначала исследуем структуру датасета
        print("Исследую структуру MMBENCH-ru...")
        mmbench_full = load_dataset("deepvk/mmbench-ru", split="dev[:2]")

        print("\nСтруктура MMBENCH-ru:")
        example = mmbench_full[0]
        for key in example.keys():
            value = example[key]
            if key == 'image':
                print(f"   {key}: {type(value)}")
            elif isinstance(value, str):
                print(f"   {key}: {value[:50]}...")
            elif isinstance(value, list):
                print(f"   {key}: список из {len(value)} элементов")
                if len(value) > 0:
                    print(f"        Первый элемент: {value[0][:30]}...")
            else:
                print(f"   {key}: {type(value)}")

        # Загружаем полную версию
        print("\nЗагружаем MMBENCH-ru...")
        mmbench_dataset = load_dataset(
            "deepvk/mmbench-ru",
            split="dev[:300]"  # 300 примеров для теста
        )

        mmbench_dataset.save_to_disk("./data/mmbench_ru")
        print(f"MMBENCH-ru загружен: {len(mmbench_dataset)} примеров")

        # Проверяем доступные поля
        if len(mmbench_dataset) > 0:
            test_example = mmbench_dataset[0]
            print("\nПроверка полей MMBENCH-ru:")
            print(f"   Доступные ключи: {list(test_example.keys())}")

            # Показываем вопрос и варианты ответа
            if 'question' in test_example:
                print(f"   Вопрос: {test_example['question']}")

            # Ищем поле с вариантами ответов
            for key in test_example.keys():
                if isinstance(test_example[key], list) and len(test_example[key]) > 0:
                    if any('вариант' in str(item).lower() or 'option' in str(item).lower()
                           for item in test_example[key][:2]):
                        print(f"   Варианты ответов (в поле '{key}'):")
                        for i, option in enumerate(test_example[key][:4]):
                            print(f"      {i + 1}. {option[:50]}...")

    except Exception as e:
        print(f"\nОшибка MMBENCH-ru: {e}")

        # Пробуем загрузить по-другому
        print("\nПробую альтернативный способ загрузки...")
        try:
            # Иногда датасет загружается как DatasetDict
            mmbench_dict = load_dataset("deepvk/mmbench-ru")
            print(f"Тип загруженных данных: {type(mmbench_dict)}")
            print(f"Ключи: {list(mmbench_dict.keys())}")

            if 'dev' in mmbench_dict:
                mmbench_dev = mmbench_dict['dev'].select(range(100))
                mmbench_dev.save_to_disk("./data/mmbench_ru_dev")
                print(f"MMBENCH-ru сохранен: {len(mmbench_dev)} примеров")

                # Смотрим на пример
                if len(mmbench_dev) > 0:
                    ex = mmbench_dev[0]
                    print("\nПример из датасета:")
                    for k, v in ex.items():
                        if k != 'image':
                            print(f"   {k}: {str(v)[:50]}...")
        except Exception as e2:
            print(f"Альтернативный способ тоже не сработал: {e2}")

    # 3. Проверяем загруженные данные
    print("\nПроверка загруженных данных")

    data_files = os.listdir("./data") if os.path.exists("./data") else []
    print("\nФайлы в папке ./data:")
    for file in data_files:
        file_path = os.path.join("./data", file)
        if os.path.isdir(file_path):
            size = len(os.listdir(file_path))
            print(f"   {file}: {size} файлов")
        else:
            print(f"   {file}")

if __name__ == "__main__":
    main()