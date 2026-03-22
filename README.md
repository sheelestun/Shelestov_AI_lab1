## Задача 1

- Тип задачи: Классификация
- Датасет: The Simpsons Characters Data
- Метрика качества: Accuracy

Краткое описание задачи
Задача заключается в разработке модели машинного обучения для классификации персонажей мультсериала «Симпсоны» на основе их визуальных характеристик из набора данных The Simpsons Characters Data. 

Целью является автоматическое определение имени персонажа по изображению с максимизацией метрики Accuracy.

## Стркутура директории

Classification/

├── data/

│   └── simpsons_dataset/

├── scripts/

│   ├── train.py

│   └── test.py

├── src/

│   ├── data.py

|   ├── model.py

│   └── utils.py


├── artifacts/

|   ├── metrics.json

│   ├── model.pth

|   ├── test_results.json

│   └── training_history.png


## Подготовка датасета

- Скачать zip-архив по ссылке: https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset
- Расархировать в папку data.
- Удалить из папки simpsons_dataset папку simpsons_dataset


## Параметры обучения

- Batch size: 64
- Learning rate: 0.001
- Weght decay: 0.001
- Epochs: 50
- Early stopping: patience = 8
- Train/val/test: 70/15/15

## Характеристики модели

- Кол-во параметров: ~423000
- Recaptive field: 75pix


## Как воспроизвести

Команда обучения:
```
python scripts/train.py
```
Команда валидации:
```
python scripts/test.py
```
## Development

- Создание и активация виртуального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
```

## **Результаты**

- Точность на тесте: 0.9210 (92.10%)
- Потери на тесте: 0.3816
- Для сравнения (Val во время обучения): 0.9217

## *График*

<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/a0353f8e-41c8-4169-95bf-ed4a14c59aa6" />







