#!/usr/bin/env python3
"""
Главный скрипт проекта RuVLM-Bench
"""
import sys
import os


def main():
    print("RuVLM-Bench: Русскоязычная Vision-Language модель")

    print("\nДоступные команды:")
    print("1. Загрузить данные:    python scripts/download_mini.py")
    print("2. Обучить модель:      python scripts/train_mini.py")
    print("3. Оценить модель:      python scripts/evaluate_mini.py")
    print("4. Тест системы:        python test_install.py")
    print("5. Быстрый тест:        python scripts/quick_test.py")

    print("\nДля запуска выберите команду из списка выше.")
    print("Пример: python scripts/download_mini.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())