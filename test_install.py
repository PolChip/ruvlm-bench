import torch
import transformers
import datasets

print("Проверка установки на Mac M2")

# Проверка PyTorch и MPS
print(f"PyTorch версия: {torch.__version__}")
print(f"MPS доступен: {torch.backends.mps.is_available()}")
print(f"MPS построен: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(3, 3).to(device)
    print(f"MPS тест: {x.device}")
else:
    print("MPS не доступен! Установите PyTorch с поддержкой MPS")

# Проверка других библиотек
print(f"\nTransformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")

print("\nВсе библиотеки установлены!")