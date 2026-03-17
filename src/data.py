import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def load_simpsons_data(data_path, img_size=(64, 64)):
    images = []
    labels = []
    label_dict = {}

    characters = sorted([d for d in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, d))])

    print(f"Найдено {len(characters)} персонажей")
    for label_idx, character in enumerate(characters):
        char_path = os.path.join(data_path, character)
        label_dict[label_idx] = character
        char_count = 0

        for img_name in os.listdir(char_path):
            img_path = os.path.join(char_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue

            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label_idx)  # Целое число, не One-Hot!
            char_count += 1

        print(f"{character:30s} | Загружено: {char_count:4d} изображений")

    X = np.array(images)
    y = np.array(labels)

    print(f"\nВсего загружено: {len(X)} изображений {len(characters)} персонажей")

    # Разделение
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # X_train = torch.tensor(X_train).unsqueeze(1)
    # X_val = torch.tensor(X_val).unsqueeze(1)
    # X_test = torch.tensor(X_test).unsqueeze(1)

    X_train = torch.tensor(X_train).permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_dict