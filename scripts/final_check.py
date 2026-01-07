#!/usr/bin/env python3
"""
Финальная проверка проекта перед сдачей
"""
import os

print("Финальная проверка проекта")

# 1. Проверка обязательных файлов
required_files = [
    'README.md',
    'requirements.txt',
    'main.py',
    'scripts/download_fixed.py',
    'scripts/train_final.py',
    'results/final/FINAL_REPORT.md',
    'results/final/best_model.pt'
]

print("\nПроверка обязательных файлов:")
for file in required_files:
    if os.path.exists(file):
        print(file)
    else:
        print(f"{file} - не найден")

# 2. Проверка структуры
print("\nПроверка структуры проекта:")
required_dirs = ['data', 'src', 'scripts', 'results', 'notebooks']
for dir in required_dirs:
    if os.path.isdir(dir):
        print(f"{dir}/")
    else:
        print(f"{dir}/ - отсутствует")

# 3. Проверка данных
print("\nПроверка данных:")
if os.path.exists('data'):
    datasets = os.listdir('data')
    for ds in datasets:
        print(f"data/{ds}")
else:
    print("Папка data отсутствует")

print("\nПроект готов")