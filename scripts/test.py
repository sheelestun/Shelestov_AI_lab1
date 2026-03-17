import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_simpsons_data
from src.model import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {DEVICE}")

(_, _), (_, _), (X_test, y_test), label_dict = load_simpsons_data(
    'dataset/simpsons_dataset',
    img_size=(64, 64)
)

# Загрузка модели
model = create_model((3, 64, 64), len(label_dict)).to(DEVICE)

if not os.path.exists('artifacts/model.pth'):
    print("Модель не найдена! Сначала запустите train.py")
    exit(1)

model.load_state_dict(torch.load('artifacts/model.pth', weights_only=True))
model.eval()

# Оценка
criterion = nn.CrossEntropyLoss()
test_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for X_batch, y_batch in DataLoader(TensorDataset(X_test, y_test), batch_size=64):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_acc = correct / total
test_loss /= len(DataLoader(TensorDataset(X_test, y_test), batch_size=64))

# Загрузка метрик валидации для сравнения
with open('artifacts/metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
print("=" * 60)
print(f"Количество классов: {metrics.get('num_classes', 0)}")
print(f"Точность на тесте: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Потери на тесте: {test_loss:.4f}")
print(f"---")
print(f"Для сравнения (Val во время обучения): {metrics.get('val_accuracy', 0):.4f}")
print("=" * 60)

# Сохранение финальных результатов
results = {
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'val_accuracy': metrics.get('val_accuracy', 0),
    'num_classes': metrics.get('num_classes', 0)
}

with open('artifacts/test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Результаты сохранены в artifacts/test_results.json")