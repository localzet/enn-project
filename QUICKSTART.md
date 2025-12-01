# Быстрый старт ENN

## Установка

```bash
cd enn-project
pip install -r requirements.txt
```

## Базовое использование

### 1. Простое обучение

```python
import torch
from core.enn_core import EmergentNeuralNetwork

# Создание сети
enn = EmergentNeuralNetwork(
    input_size=10,
    output_size=1,
    initial_hidden=5,
    max_neurons=50
)

# Данные
x = torch.randn(100, 10)
y = torch.sum(x, dim=1, keepdim=True) / 10

# Обучение
for i in range(100):
    metrics = enn.learn(x[i:i+1], y[i:i+1], reward=1.0)
    if (i + 1) % 20 == 0:
        print(f"Эпоха {i+1}: ошибка={metrics['error']:.6f}")

# Предсказание
predictions = enn.forward(x)
```

### 2. Автономный агент

```python
from modules.autonomous_agents import AutonomousAgentENN

agent = AutonomousAgentENN(input_size=8, output_size=2)

# Онлайн обучение
observation = torch.randn(8)
action = torch.randn(2)
reward = 1.0

metrics = agent.learn_online(observation, action, reward)
print(f"Ошибка: {metrics['error']:.6f}")
```

### 3. Continual Learning

```python
from modules.continual_learning import ContinualLearningENN

cl_enn = ContinualLearningENN(input_size=10, output_size=1)

# Задача 1
x1 = torch.randn(50, 10)
y1 = torch.randn(50, 1)
for i in range(50):
    cl_enn.learn_task(0, x1[i:i+1], y1[i:i+1], importance=1.0)

# Задача 2
x2 = torch.randn(50, 10)
y2 = torch.randn(50, 1)
for i in range(50):
    cl_enn.learn_task(1, x2[i:i+1], y2[i:i+1], importance=1.0)

# Оценка забывания
stats = cl_enn.get_forgetting_statistics()
print(f"Забывание: {stats['average_forgetting']:.6f}")
```

### 4. Ресурсно-ограниченная система

```python
from modules.resource_constrained import ResourceConstrainedENN

rc_enn = ResourceConstrainedENN(
    input_size=5,
    output_size=1,
    max_energy_per_step=1.0,
    max_neurons=30
)

# Адаптация к ресурсам
rc_enn.adapt_to_resources(available_energy=10.0, available_memory=20)

# Обучение
x = torch.randn(50, 5)
y = torch.randn(50, 1)
for i in range(50):
    metrics = rc_enn.learn(x[i:i+1], y[i:i+1], reward=1.0)
    print(f"Энергия использована: {metrics.get('energy_used', 0):.2f}")
```

## Запуск примеров

```bash
python examples/basic_usage.py
```

## Бенчмарки

```python
from benchmarks.compare_models import compare_with_baselines

# Подготовьте данные
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
X_test = torch.randn(20, 10)
y_test = torch.randn(20, 1)

# Сравнение
results = compare_with_baselines(
    "synthetic",
    (X_train, y_train),
    (X_test, y_test),
    epochs=50
)

from benchmarks.compare_models import print_comparison_results
print_comparison_results(results)
```

## Тесты робастности

```python
from benchmarks.robustness_tests import RobustnessTester
from core.enn_core import EmergentNeuralNetwork

# Обучение модели
enn = EmergentNeuralNetwork(input_size=10, output_size=1)
# ... обучение ...

# Тесты
tester = RobustnessTester()

# Устойчивость к шуму
tester.test_noise_robustness(enn, X_test, y_test)

# Катастрофическое забывание
tester.test_catastrophic_forgetting(enn, task1_data, task2_data)

# Отчет
print(tester.generate_report())
```

## Дополнительная информация

- `README.md` - Полная документация
- `ARCHITECTURE.md` - Архитектура проекта
- `SUMMARY.md` - Резюме реализации
- `BREAKTHROUGH_PAPER.md` - Научная статья

