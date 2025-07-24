import os
import sys


def check_io(args, input_keys=None, output_keys=None):
    input_keys = input_keys or []
    output_keys = output_keys or []

    # Перевірка вхідних файлів
    for key in input_keys:
        path = getattr(args, key, None)
        if path and not os.path.isfile(path):
            print(f"❌ Помилка: вхідний файл не існує: {path}", file=sys.stderr)
            sys.exit(1)

    # Створення директорій для вихідних файлів
    for key in output_keys:
        path = getattr(args, key, None)
        if path:
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} мс"
    elif seconds < 60:
        return f"{seconds:.2f} сек"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins} хв {secs:.1f} сек"
