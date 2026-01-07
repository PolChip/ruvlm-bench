
import torch
import platform


def get_device(verbose=True):
    """
    Определяет лучшее доступное устройство для вычислений

    Returns:
        torch.device: Лучшее доступное устройство
    """
    device = None

    # 1. Проверка MPS (Metal Performance Shaders) - для Mac M1/M2
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f" Используется MPS (Apple Silicon ускорение)")

    # 2. Проверка CUDA (NVIDIA GPU) - на Mac обычно нет
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f" Используется CUDA: {torch.cuda.get_device_name(0)}")

    # 3. Fallback на CPU
    else:
        device = torch.device("cpu")
        if verbose:
            print("  Используется CPU. Для лучшей производительности:")
            print("   - Убедитесь, что установлен PyTorch с поддержкой MPS")
            print("   - Обновитесь до последней версии macOS")

    # Дополнительная информация
    if verbose:
        system_info = platform.platform()
        print(f"   Система: {system_info}")
        print(f"   PyTorch версия: {torch.__version__}")

    return device


def optimize_for_mps():
    """
    Оптимизация настроек для MPS на Mac
    """
    if torch.backends.mps.is_available():
        # Установить лимит памяти для MPS (процент от доступной)
        torch.mps.set_per_process_memory_fraction(0.7)

        # Включить autocast для mixed precision
        torch.autocast("mps", dtype=torch.float16)

        print(" MPS оптимизирован:")
        print(f"   Лимит памяти: 70%")
        print(f"   Mixed precision: включен")

        return True
    return False


def clear_mps_cache():
    """
    Очистка кэша MPS (решает проблему с памятью)
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print(" Кэш MPS очищен")
        return True
    return False


def print_memory_info():
    """
    Вывод информации об использовании памяти
    """
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024 ** 2
        reserved = torch.mps.driver_allocated_memory() / 1024 ** 2

        print(f" Память MPS:")
        print(f"   Выделено: {allocated:.1f} MB")
        print(f"   Зарезервировано: {reserved:.1f} MB")

        return allocated, reserved

    print("  MPS не доступен для мониторинга памяти")
    return None, None


# Тест устройства при импорте
if __name__ == "__main__":
    device = get_device()
    print(f"\nИспользуемое устройство: {device}")