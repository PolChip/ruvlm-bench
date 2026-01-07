"""
Вспомогательные функции для проекта
"""
import os
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt


def ensure_dir(path):
    """Создание директории, если её нет"""
    os.makedirs(path, exist_ok=True)
    return path


def save_config(config, path):
    """Сохранение конфигурации в JSON или YAML"""
    ensure_dir(os.path.dirname(path))

    if path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"Конфигурация сохранена: {path}")


def get_timestamp():
    """Получение временной метки для именования файлов"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_training_history(history, save_path=None):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.get('train_loss', []), label='Train')
    axes[0].plot(history.get('val_loss', []), label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if 'val_accuracy' in history:
        axes[1].plot(history.get('val_accuracy', []))
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")

    plt.show()