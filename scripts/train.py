import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

os.makedirs('artifacts', exist_ok=True)

from src.data import load_simpsons_data
from src.model import create_model
from src.utils import plot_training_history, get_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

(X_train, y_train), (X_val, y_val), (X_test, y_test), label_dict = load_simpsons_data(
    'dataset/simpsons_dataset',
    img_size=(64, 64)
)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
print(f"Количество классов: {len(label_dict)}")

train_transform = get_transforms()
val_transform = get_transforms(training=False)

def apply_transforms(X_batch, transform, is_train=True):
    if is_train and transform.transforms:
        transformed = []
        for img in X_batch:
            img_pil = transforms.ToPILImage()(img)
            img_aug = transform(img_pil)
            transformed.append(img_aug)
        return torch.stack(transformed)
    else:
        return X_batch

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

model = create_model((3, 64, 64), len(label_dict)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
patience_counter = 0
max_patience = 8

print("\nНачало обучения...")
for epoch in range(50):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch = apply_transforms(X_batch, train_transform, is_train=True)
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss /= len(train_loader)
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Early Stopping & Checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'artifacts/model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\nEarlyStopping на эпохе {epoch + 1}")
            break

plot_training_history(history)
metrics = {
    'val_accuracy': float(max(history['val_acc'])),
    'num_classes': len(label_dict),
    'label_map': label_dict
}

with open('artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\nОбучение завершено!")
print(f"Лучшая точность на валидации: {best_val_acc:.4f}")
print(f"Модель сохранена в 'artifacts/model.pth'")
print(f"Для проверки на тесте запустите: python scripts/test.py")