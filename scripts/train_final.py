#!/usr/bin/env python3
"""
Финальный скрипт обучения VLM
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt


class VLM_Dataset(Dataset):
    """Датасет для VLM обучения"""

    def __init__(self, dataset_name="gqa_ru", max_samples=50):
        print(f"Загрузка датасета: {dataset_name}")

        # Загружаем данные
        self.data = load_from_disk(f"./data/{dataset_name}")

        # Ограничиваем количество для теста
        if len(self.data) > max_samples:
            indices = random.sample(range(len(self.data)), max_samples)
            self.data = self.data.select(indices)

        print(f"Загружено: {len(self.data)} примеров")

        # Простой словарь для текста
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.next_id = 2

        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Строим словарь из вопросов
        self._build_vocab()

        # Сохраняем информацию о датасете
        self.dataset_name = dataset_name

    def _build_vocab(self):
        """Строим простой словарь из вопросов"""
        for item in self.data:
            if 'question' in item:
                text = item['question'].lower()
                words = list(set(text.split()))
                for word in words[:10]:
                    if word not in self.vocab and len(self.vocab) < 500:
                        self.vocab[word] = self.next_id
                        self.next_id += 1

        print(f"Размер словаря: {len(self.vocab)} слов")

    def _text_to_ids(self, text):
        """Конвертируем текст в индексы"""
        if not isinstance(text, str):
            text = str(text)

        words = text.lower().split()[:8]
        ids = [self.vocab.get(word, 1) for word in words]

        while len(ids) < 8:
            ids.append(0)

        return torch.tensor(ids[:8], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Обработка изображения
        if 'image' in item:
            try:
                image = item['image']
                if not isinstance(image, torch.Tensor):
                    image = self.transform(image.convert('RGB'))
            except:
                image = torch.zeros(3, 128, 128)
        else:
            image = torch.zeros(3, 128, 128)

        # Обработка текста
        text = ""
        if 'question' in item:
            text = item['question']
        elif 'hint' in item:
            text = item['hint']

        text_ids = self._text_to_ids(text)

        if self.dataset_name == "gqa_ru":
            if 'answer' in item:
                answer = item['answer']
                target = hash(answer) % 10
            else:
                target = idx % 10
        else:
            if 'answer' in item:
                answer = item['answer']
                if answer == 'A':
                    target = 0
                elif answer == 'B':
                    target = 1
                elif answer == 'C':
                    target = 2
                elif answer == 'D':
                    target = 3
                else:
                    target = 0
            else:
                target = idx % 4

        return {
            'image': image,
            'text_ids': text_ids,
            'target': target,
            'question': str(text)[:30] + "..." if len(str(text)) > 30 else str(text),
            'dataset': self.dataset_name
        }


class VLM_Model(nn.Module):
    """Vision-Language модель"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.text_encoder = nn.Sequential(
            nn.Embedding(500, 64),
            nn.LSTM(64, 128, batch_first=True, bidirectional=True),
        )
        self.text_projection = nn.Linear(256, 256)

        self.combined = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        print(f"Модель создана:")
        print(f"Параметров: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Обучаемых: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, images, text_ids):
        img_features = self.image_encoder(images)

        text_emb = self.text_encoder[0](text_ids)
        lstm_out, (hidden, cell) = self.text_encoder[1](text_emb)

        txt_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        txt_features = self.text_projection(txt_features)

        combined = torch.cat([img_features, txt_features], dim=1)

        output = self.combined(combined)

        return output


def train():
    """Основная функция обучения"""
    print("")
    print("ОБУЧЕНИЕ VLM МОДЕЛИ НА MAC M2")
    print("")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    print(f"MPS доступен: {torch.backends.mps.is_available()}")

    os.makedirs("./results/final", exist_ok=True)
    os.makedirs("./results/final/checkpoints", exist_ok=True)

    print("\nПодготовка данных...")

    dataset = VLM_Dataset("gqa_ru", max_samples=100)

    if len(dataset) == 0:
        print("Нет данных для обучения")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)} примеров")
    print(f"Val: {len(val_dataset)} примеров")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("\nСоздание модели...")
    model = VLM_Model(num_classes=10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()

    print("\nНачало обучения...")
    num_epochs = 5
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'learning_rate': []
    }

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")

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

            if batch_idx % 5 == 0 and batch_idx > 0:
                print(f"Батч {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

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

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': accuracy,
            'history': history
        }

        checkpoint_path = f"./results/final/checkpoints/epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "./results/final/best_model.pt")
            print(f"Новая лучшая модель! Точность: {accuracy:.2f}%")

        print(f"Чекпоинт сохранён: {checkpoint_path}")

        scheduler.step()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\nСохранение финальной модели...")
    torch.save(model.state_dict(), "./results/final/final_model.pt")

    print("\nВизуализация результатов обучения...")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', marker='o', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], marker='D', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Validation Accuracy\nBest: {best_acc:.1f}%')
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'], marker='^', color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/final/training_history.png", dpi=150, bbox_inches='tight')

    print("\nТестовые предсказания на валидационных данных:")

    model.eval()
    test_examples = []

    with torch.no_grad():
        val_batch = next(iter(val_loader))

        for i in range(min(3, len(val_batch['image']))):
            image = val_batch['image'][i:i + 1].to(device)
            text_ids = val_batch['text_ids'][i:i + 1].to(device)
            target = val_batch['target'][i:i + 1].to(device)
            question = val_batch['question'][i]

            output = model(image, text_ids)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            conf = probabilities[0][predicted].item()

            test_examples.append({
                'question': question,
                'predicted': predicted.item(),
                'true': target.item(),
                'confidence': conf,
                'correct': predicted.item() == target.item()
            })

    for i, example in enumerate(test_examples):
        print(f"\nПример {i + 1}:")
        print(f"Вопрос: {example['question']}")
        print(f"Предсказанный класс: {example['predicted']}")
        print(f"Истинный класс: {example['true']}")
        print(f"Уверенность: {example['confidence']:.2%}")
        print(f"Результат: {'ПРАВИЛЬНО' if example['correct'] else 'НЕПРАВИЛЬНО'}")

    print("\nОценка на MMBENCH-ru датасете:")

    try:
        mmbench_dataset = VLM_Dataset("mmbench_ru", max_samples=30)
        mmbench_loader = DataLoader(mmbench_dataset, batch_size=2, shuffle=False)

        model.eval()
        mmbench_correct = 0
        mmbench_total = 0

        with torch.no_grad():
            for batch in mmbench_loader:
                images = batch['image'].to(device)
                text_ids = batch['text_ids'].to(device)
                targets = batch['target'].to(device)

                outputs = model(images, text_ids)
                _, predicted = torch.max(outputs, 1)

                mmbench_total += targets.size(0)
                mmbench_correct += (predicted == targets).sum().item()

        mmbench_acc = 100 * mmbench_correct / mmbench_total if mmbench_total > 0 else 0
        print(f"MMBENCH-ru точность: {mmbench_acc:.1f}% ({mmbench_correct}/{mmbench_total})")

    except Exception as e:
        print(f"Ошибка оценки на MMBENCH-ru: {e}")

    print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")

    print(f"\nРЕЗУЛЬТАТЫ СОХРАНЕНЫ В ./results/final/")
    print("Модели:")
    print("final_model.pt - финальная модель")
    print("best_model.pt - лучшая модель по точности")
    print("checkpoints/ - чекпоинты каждой эпохи")
    print("\nГрафики:")
    print("training_history.png - график обучения")
    print(f"\nЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    print(f"Лучшая точность: {best_acc:.1f}%")
    print(f"Финальная точность: {history['val_acc'][-1]:.1f}%")

    return history


def main():
    """Главная функция"""
    try:
        history = train()
        return 0
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("\nРекомендации по устранению:")
        print("1. Уменьшите batch_size в строках 195, 196")
        print("2. Уменьшите max_samples в строке 146")
        print("3. Уменьшите размер модели в классе VLM_Model")
        return 1


if __name__ == "__main__":
    sys.exit(main())