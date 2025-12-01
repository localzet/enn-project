# Emergent Neural Network (ENN)

## Алгоритм нейронной сети с динамической топологией

**Emergent Neural Network (ENN)** - это научно обоснованный алгоритм нейронной сети, объединяющий динамическую топологию, энергобюджет и формальные методы анализа.

## 🎯 Ключевые особенности

### Научная новизна:

1. **Динамическая топология** - сеть сама определяет оптимальную структуру
2. **Энергобюджет** - формализованное управление ресурсами
3. **Теоретический анализ** - условия сходимости и стабильности
4. **Формальная каузальность** - Granger Causality, SCM, do-calculus
5. **Модульная архитектура** - специализированные модули для разных применений

## 📁 Структура проекта

```
enn-project/
├── core/                    # Ядро алгоритма
│   ├── enn_core.py         # Основная реализация ENN
│   ├── topology_evolution.py # Эволюция топологии
│   ├── energy_management.py  # Управление энергией
│   ├── stability_analysis.py # Анализ стабильности
│   └── causality.py         # Причинно-следственная логика
│
├── modules/                 # Специализированные модули
│   ├── autonomous_agents.py # Для автономных агентов
│   ├── continual_learning.py # Для continual learning
│   └── resource_constrained.py # Для ресурсно-ограниченных систем
│
├── benchmarks/             # Бенчмарки и сравнения
│   ├── compare_models.py   # Сравнение с базовыми моделями
│   └── robustness_tests.py # Тесты робастности
│
└── experiments/            # Эксперименты
```

## 🚀 Быстрый старт

### Базовое использование

```python
from core import EmergentNeuralNetwork
import torch

# Создание сети
enn = EmergentNeuralNetwork(
    input_size=10,
    output_size=1,
    initial_hidden=5,
    max_neurons=100
)

# Обучение
x = torch.randn(100, 10)
y = torch.randn(100, 1)

for i in range(100):
    enn.learn(x[i:i+1], y[i:i+1], reward=1.0)

# Использование
predictions = enn.forward(x)
```

### Специализированные модули

#### Автономные агенты

```python
from modules import AutonomousAgentENN

agent = AutonomousAgentENN(input_size=8, output_size=2)
agent.learn_online(observation, action, reward)
```

#### Continual Learning

```python
from modules import ContinualLearningENN

cl_enn = ContinualLearningENN(input_size=10, output_size=1)
cl_enn.learn_task(task_id=0, x, y, importance=1.0)
cl_enn.learn_task(task_id=1, x2, y2, importance=1.0)
```

#### Ресурсно-ограниченные системы

```python
from modules import ResourceConstrainedENN

rc_enn = ResourceConstrainedENN(
    input_size=5,
    output_size=1,
    max_energy_per_step=1.0,
    max_neurons=50
)
rc_enn.adapt_to_resources(available_energy=10.0, available_memory=30)
```

## 🔬 Научные компоненты

### Теоретический анализ

```python
from core import StabilityAnalyzer

analyzer = StabilityAnalyzer()
report = analyzer.get_stability_report(weights, energy_history, neuron_counts, max_neurons)
```

### Причинно-следственная логика

```python
from core import CausalityAnalyzer

causality = CausalityAnalyzer()
causal_graph = causality.build_causal_graph(activations_history)
```

## 📊 Бенчмарки

### Сравнение с базовыми моделями

```python
from benchmarks import compare_with_baselines

results = compare_with_baselines(
    dataset_name="synthetic",
    train_data=(X_train, y_train),
    test_data=(X_test, y_test),
    epochs=100
)
```

### Тесты робастности

```python
from benchmarks import RobustnessTester

tester = RobustnessTester()
tester.test_noise_robustness(model, x, y)
tester.test_catastrophic_forgetting(model, task1_data, task2_data)
report = tester.generate_report()
```

## 📚 Документация

- `BREAKTHROUGH_PAPER.md` - Полная научная статья
- `core/` - Документация ядра
- `modules/` - Документация модулей
- `benchmarks/` - Документация бенчмарков

## 🎯 Области применения

1. **Автономные агенты** - адаптация к новым условиям
2. **Continual Learning** - обучение на последовательности задач
3. **Ресурсно-ограниченные системы** - мобильные устройства, IoT
4. **Научные исследования** - изучение эмерджентных свойств

## 📝 Установка

```bash
pip install -r requirements.txt
```