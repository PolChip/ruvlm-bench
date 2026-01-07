#!/usr/bin/env python3
"""
Продолжение обучения с увеличенным датасетом
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from scripts.train_final import VLM_Dataset, VLM_Model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def continue_training():
    print("ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ VLM МОДЕЛИ")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Устройство: {device}")

    print("\nЗагрузка большего датасета...")
    dataset = VLM_Dataset("gqa_ru", max_samples=300)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)} примеров")
    print(f"Val: {len(val_dataset)} примеров")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("\nЗагрузка предобученной модели...")
    model = VLM_Model(num_classes=10).to(device)

    try:
        model.load_state_dict(torch.load("./results/final/best_model.pt", map_location=device))
        print("Модель загружена успешно")
    except:
        print("Не удалось загрузить модель, создаём новую")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\nПродолжение обучения...")
    additional_epochs = 3
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(additional_epochs):
        print(f"\nДополнительная эпоха {epoch + 1}/{additional_epochs}")

        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            text_ids = batch['text_ids'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(images, text_ids)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"Батч {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Средний train loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                text_ids = batch['text_ids'].to(device)
                targets = batch['target'].to(device)

                outputs = model(images, text_ids)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(accuracy)

        print(f"Средний val loss: {avg_val_loss:.4f}")
        print(f"Val accuracy: {accuracy:.2f}% ({correct}/{total})")

        if accuracy > 15.0:
            torch.save(model.state_dict(), f"./results/final/improved_model_epoch{epoch}.pt")
            print(f"Улучшенная модель сохранена!")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    torch.save(model.state_dict(), "./results/final/improved_final_model.pt")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Continued Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], marker='D', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Final Accuracy: {history["val_acc"][-1]:.1f}%')
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/final/continued_training.png", dpi=150, bbox_inches='tight')

    print("ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ ЗАВЕРШЕНО")

    print(f"\nНовая модель: ./results/final/improved_final_model.pt")
    print(f"Финальная точность: {history['val_acc'][-1]:.1f}%")

    return history


if __name__ == "__main__":
    continue_training()