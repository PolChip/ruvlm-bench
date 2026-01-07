"""
Простая VLM модель для тестирования на Mac M2
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from src.utils.device import get_device

class SimpleVLM(nn.Module):
    """Упрощённая Vision-Language модель для Mac"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = get_device(verbose=False)

        print("🔧 Инициализация SimpleVLM...")

        # Vision encoder (CLIP)
        print(f"   Загрузка vision encoder: {config['model']['vision_encoder']}")
        self.vision_model = AutoModel.from_pretrained(
            config['model']['vision_encoder'],
            torch_dtype=torch.float16 if config['device']['mixed_precision'] else torch.float32
        )

        # Language model (Russian GPT)
        print(f"   Загрузка language model: {config['model']['language_model']}")
        self.language_model = AutoModel.from_pretrained(
            config['model']['language_model'],
            torch_dtype=torch.float16 if config['device']['mixed_precision'] else torch.float32
        )

        # Получаем размерности
        # Для CLIP: hidden_size находится в vision_config
        vision_dim = self.vision_model.config.vision_config.hidden_size
        language_dim = self.language_model.config.hidden_size

        print(f"   Vision dimension: {vision_dim}")
        print(f"   Language dimension: {language_dim}")

        # Проекционный слой
        self.projection = nn.Linear(vision_dim, language_dim)

        # Заморозить часть слоёв для экономии памяти
        self._freeze_layers()

        # Перенести на устройство
        self.to(self.device)

        print(f" Модель инициализирована на устройстве: {self.device}")
        print(f"   Параметров: {sum(p.numel() for p in self.parameters()):,}")

    def _freeze_layers(self):
        """Заморозка слоёв для экономии памяти"""
        # Заморозить vision encoder (кроме последних слоёв)
        for name, param in list(self.vision_model.named_parameters()):
            if 'vision_model.encoder.layers.11' not in name:  # Не замораживать последний слой
                param.requires_grad = False

        # Заморозить language model (кроме последних слоёв)
        for name, param in list(self.language_model.named_parameters()):
            if 'h.11' not in name:  # Не замораживать последний слой GPT
                param.requires_grad = False

        print("   Заморожены слои для экономии памяти")
        print(f"   Обучаемых параметров: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, images, input_ids, attention_mask):
        """Прямой проход"""
        # Визуальные фичи
        vision_outputs = self.vision_model(pixel_values=images)

        # Для CLIP используем pooler_output или last_hidden_state
        if hasattr(vision_outputs, 'pooler_output'):
            image_features = vision_outputs.pooler_output
        else:
            image_features = vision_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Проекция
        projected_features = self.projection(image_features)

        # Текстовые фичи
        text_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Берем последний hidden state
        text_features = text_outputs.last_hidden_state

        # Простая конкатенация (визуальные фичи + текстовые)
        # Расширяем визуальные фичи до размерности текстовых
        projected_features = projected_features.unsqueeze(1)  # [batch, 1, dim]

        # Конкатенируем
        combined_features = torch.cat([projected_features, text_features], dim=1)

        return combined_features

    def predict(self, image, question, tokenizer, processor, max_length=50):
        """Предсказание для одного примера"""
        self.eval()

        # Подготовка изображения
        image_inputs = processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Подготовка текста
        text_inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Прямой проход
        with torch.no_grad():
            features = self.forward(
                images=image_inputs['pixel_values'],
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )

        # Простая логика: вернуть размер фичей
        return f"Вопрос: {question}\nОтвет: [модель готова, фичи: {features.shape}]"