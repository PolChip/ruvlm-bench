#!/usr/bin/env python3
"""
Быстрый тест всей системы
"""
import subprocess
import sys


def run_command(cmd):
    """Выполнить команду"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def main():
    print("Быстрый тест системы RuVLM-Bench")
    print("")

    tests = [
        ("Python версия", "python --version"),
        ("PyTorch", "python -c 'import torch; print(torch.__version__)'"),
        ("MPS доступен", "python -c 'import torch; print(torch.backends.mps.is_available())'"),
        ("HuggingFace", "python -c 'import transformers; print(transformers.__version__)'"),
        ("Папка проекта", "pwd"),
    ]

    all_passed = True

    for test_name, cmd in tests:
        passed, output = run_command(cmd)
        status = "OK" if passed else "FAIL"

        print(f"\n{status} {test_name}:")
        print(f"   Команда: {cmd}")
        print(f"   Результат: {output}")

        if not passed:
            all_passed = False

    print("\n")
    if all_passed:
        print("Все тесты пройдены! Система готова.")
        print("\nЗапустите: python scripts/download_mini.py")
    else:
        print("Есть проблемы с системой.")
        print("Проверьте установку зависимостей.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())