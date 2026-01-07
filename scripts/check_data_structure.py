#!/usr/bin/env python3
"""
Проверка структуры загруженных данных
"""
import os
from datasets import load_from_disk


def main():
    print("ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")

    # Проверим папку data
    print("\nСодержимое папки ./data:")
    if os.path.exists("./data"):
        items = os.listdir("./data")
        for item in items:
            item_path = os.path.join("./data", item)
            if os.path.isdir(item_path):
                size = sum(os.path.getsize(os.path.join(item_path, f))
                           for f in os.listdir(item_path)
                           if os.path.isfile(os.path.join(item_path, f)))
                print(f"  {item}/ ({(size / 1024 / 1024):.1f} MB)")
            else:
                print(f"  {item}")
    else:
        print("  Папка ./data не существует")
        return

    # Попробуем загрузить каждый датасет
    print("\nПопытка загрузить датасеты:")

    data_dirs = [d for d in os.listdir("./data")
                 if os.path.isdir(os.path.join("./data", d))]

    for data_dir in data_dirs:
        try:
            print(f"\n  Загрузка {data_dir}...")
            dataset = load_from_disk(f"./data/{data_dir}")
            print(f"    Успешно! {len(dataset)} примеров")

            if len(dataset) > 0:
                example = dataset[0]
                print(f"    Пример 1:")
                print(f"      Ключи: {list(example.keys())}")

                # Покажем некоторые поля
                for key in ['question', 'answer', 'hint', 'A', 'B', 'image']:
                    if key in example:
                        value = example[key]
                        if isinstance(value, str):
                            print(f"      {key}: {value[:50]}...")
                        else:
                            print(f"      {key}: {type(value)}")

        except Exception as e:
            print(f"    Ошибка: {str(e)[:50]}...")

if __name__ == "__main__":
    main()