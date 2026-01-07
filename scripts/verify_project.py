#!/usr/bin/env python3
"""
Проверка готовности проекта
"""
import os
import sys
from pathlib import Path


def check_structure():
    print("Проверка структуры проекта:")

    required_dirs = ['data', 'results', 'scripts', 'src', 'notebooks']
    required_files = ['README.md', 'requirements.txt', 'main.py']

    all_ok = True

    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"Папка {dir_name}/ существует")
        else:
            print(f"Папка {dir_name}/ отсутствует")
            all_ok = False

    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"Файл {file_name} существует")
        else:
            print(f"Файл {file_name} отсутствует")
            all_ok = False

    return all_ok


def check_data():
    print("\nПроверка данных:")

    data_dirs = ['gqa_ru', 'mmbench_ru']
    all_ok = True

    for data_dir in data_dirs:
        path = f"data/{data_dir}"
        if os.path.isdir(path):
            files = os.listdir(path)
            if files:
                print(f"Датсет {data_dir}: {len(files)} файлов")
            else:
                print(f"Датсет {data_dir}: папка пуста")
                all_ok = False
        else:
            print(f"Датсет {data_dir} не найден")
            all_ok = False

    return all_ok


def check_results():
    print("\nПроверка результатов:")

    results_path = "results/final"
    if os.path.isdir(results_path):
        files = os.listdir(results_path)

        required_files = ['best_model.pt', 'final_model.pt', 'training_history.png']
        found_files = []

        for file in files:
            if file in required_files:
                found_files.append(file)

        if len(found_files) >= 2:
            print(f"Результаты обучения найдены: {', '.join(found_files)}")

            model_path = f"{results_path}/best_model.pt"
            if os.path.isfile(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"Размер модели: {size_mb:.1f} MB")

            return True
        else:
            print(f"Не хватает файлов результатов. Найдено: {found_files}")
            return False
    else:
        print("Папка results/final не найдена")
        return False


def check_requirements():
    print("\nПроверка зависимостей:")

    try:
        import torch
        import transformers
        import datasets
        import numpy
        import matplotlib

        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        print(f"Datasets: {datasets.__version__}")
        print(f"MPS доступен: {torch.backends.mps.is_available()}")

        return True
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        return False


def check_functionality():
    print("\nПроверка функциональности:")

    tests = []

    try:
        import subprocess
        result = subprocess.run(['python', 'main.py'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("main.py запускается")
            tests.append(True)
        else:
            print("main.py не запускается")
            tests.append(False)
    except:
        print("Ошибка запуска main.py")
        tests.append(False)

    scripts_to_test = ['download_fixed.py', 'train_final.py']

    for script in scripts_to_test:
        script_path = f"scripts/{script}"
        if os.path.isfile(script_path):
            print(f"Скрипт {script} существует")
            tests.append(True)
        else:
            print(f"Скрипт {script} отсутствует")
            tests.append(False)

    return all(tests)


def generate_report():
    print("\nОТЧЕТ О ГОТОВНОСТИ ПРОЕКТА")

    checks = [
        ("Структура проекта", check_structure()),
        ("Данные", check_data()),
        ("Результаты", check_results()),
        ("Зависимости", check_requirements()),
        ("Функциональность", check_functionality())
    ]

    total_passed = sum(1 for _, passed in checks if passed)
    total_checks = len(checks)

    print("\nИтоги проверки:")
    for check_name, passed in checks:
        status = "OK" if passed else "FAIL"
        print(f"{status} {check_name}")

    print(f"\nПройдено проверок: {total_passed}/{total_checks}")

    if total_passed == total_checks:
        print("\nПРОЕКТ ГОТОВ К СДАЧЕ!")
    else:
        print(f"\nНужно исправить: {total_checks - total_passed} пунктов")

    return total_passed == total_checks


if __name__ == "__main__":
    print("ПРОВЕРКА ГОТОВНОСТИ ПРОЕКТА RuVLM-Bench")

    ready = generate_report()
    sys.exit(0 if ready else 1)