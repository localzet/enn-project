# Архитектура ENN Project

## Модульная структура

Проект организован как модульная система с ядром и специализированными модулями.

## Ядро (core/)

### `enn_core.py`
Основная реализация ENN с PyTorch:
- Динамическая топология
- Энергобюджет
- Обучение и предсказание

### `topology_evolution.py`
Формализованная эволюция топологии:
- Правила создания нейронов
- Правила удаления нейронов
- Математическая формализация

### `energy_management.py`
Управление энергобюджетом:
- Потребление энергии
- Генерация энергии
- Отслеживание истории

### `stability_analysis.py`
Теоретический анализ стабильности:
- Анализ сходимости
- Стабильность энергии
- Стабильность топологии

### `causality.py`
Формальная причинно-следственная логика:
- Granger Causality
- Structural Causal Models
- Do-calculus

## Модули (modules/)

### `autonomous_agents.py`
Для автономных агентов:
- Онлайн обучение
- Адаптация к среде
- Защита от забывания

### `continual_learning.py`
Для continual learning:
- Защита от катастрофического забывания
- Память для задач
- Передача знаний

### `resource_constrained.py`
Для ресурсно-ограниченных систем:
- Строгое управление энергией
- Ограничение размера
- Энергоэффективный режим

## Бенчмарки (benchmarks/)

### `compare_models.py`
Сравнение с базовыми моделями:
- MLP
- RNN
- Метрики сравнения

### `robustness_tests.py`
Тесты робастности:
- Устойчивость к шуму
- Катастрофическое забывание
- Изменение распределения
- Выбросы

## Использование

### Базовое использование

```python
from core import EmergentNeuralNetwork

enn = EmergentNeuralNetwork(input_size=10, output_size=1)
enn.learn(x, y, reward=1.0)
predictions = enn.forward(x)
```

### Специализированные модули

```python
from modules import AutonomousAgentENN

agent = AutonomousAgentENN(input_size=8, output_size=2)
agent.learn_online(observation, action, reward)
```

### Бенчмарки

```python
from benchmarks import compare_with_baselines

results = compare_with_baselines("dataset", train_data, test_data)
```

## Расширение

Для добавления нового модуля:

1. Создайте файл в `modules/`
2. Наследуйтесь от `EmergentNeuralNetwork`
3. Добавьте специализированные методы
4. Обновите `modules/__init__.py`

## Зависимости

- PyTorch - для нейронных сетей
- NumPy - для вычислений
- SciPy - для статистики
- scikit-learn - для базовых моделей

