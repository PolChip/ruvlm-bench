"""
Конфигурации проекта
"""
import yaml
import os


def load_config(config_path="src/configs/base_config.yaml"):
    """Загрузка конфигурации из YAML файла"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Загружена конфигурация из: {config_path}")
    return config


# Экспорт конфигурации по умолчанию
BASE_CONFIG = load_config()