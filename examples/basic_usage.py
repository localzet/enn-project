"""
Базовые примеры использования ENN
"""

import torch
import numpy as np
import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import EmergentNeuralNetwork, StabilityAnalyzer, CausalityAnalyzer


def example_basic_training():
    """Базовое обучение ENN"""
    print("="*70)
    print("Пример 1: Базовое обучение ENN")
    print("="*70)
    
    # Создание сети
    enn = EmergentNeuralNetwork(
        input_size=10,
        output_size=1,
        initial_hidden=5,
        max_neurons=50
    )
    
    # Генерация данных
    x = torch.randn(100, 10)
    y = torch.sum(x, dim=1, keepdim=True) / 10  # Простая функция
    
    # Обучение
    print("\nОбучение...")
    for i in range(100):
        metrics = enn.learn(x[i:i+1], y[i:i+1], reward=1.0)
        if (i + 1) % 20 == 0:
            print(f"  Эпоха {i+1}: ошибка={metrics['error']:.6f}, "
                  f"нейронов={metrics['num_neurons']}, "
                  f"энергия={metrics['energy_budget']:.2f}")
    
    # Тестирование
    with torch.no_grad():
        predictions = enn.forward(x[:10])
        test_error = torch.nn.functional.mse_loss(predictions, y[:10])
        print(f"\nОшибка на тесте: {test_error.item():.6f}")
    
    # Статистика
    stats = enn.get_statistics()
    print(f"\nФинальная статистика:")
    print(f"  Нейронов: {stats['num_neurons']}")
    print(f"  Связей: {stats['num_connections']}")
    print(f"  Энергобюджет: {stats['energy_budget']:.2f}")


def example_stability_analysis():
    """Анализ стабильности"""
    print("\n" + "="*70)
    print("Пример 2: Анализ стабильности")
    print("="*70)
    
    enn = EmergentNeuralNetwork(input_size=5, output_size=1)
    
    # Обучение с отслеживанием истории
    energy_history = []
    neuron_counts = []
    
    x = torch.randn(50, 5)
    y = torch.randn(50, 1)
    
    for i in range(50):
        enn.learn(x[i:i+1], y[i:i+1], reward=1.0)
        stats = enn.get_statistics()
        energy_history.append(stats['energy_budget'])
        neuron_counts.append(stats['num_neurons'])
    
    # Анализ стабильности
    analyzer = StabilityAnalyzer()
    
    # Анализ энергии
    energy_stability = analyzer.analyze_energy_stability(energy_history)
    print(f"\nСтабильность энергии:")
    print(f"  Стабильна: {energy_stability['is_stable']}")
    print(f"  Минимальная энергия: {energy_stability['min_energy']:.2f}")
    print(f"  Средняя энергия: {energy_stability['mean_energy']:.2f}")
    
    # Анализ топологии
    topology_stability = analyzer.analyze_topology_stability(neuron_counts, enn.max_neurons)
    print(f"\nСтабильность топологии:")
    print(f"  Стабильна: {topology_stability['is_stable']}")
    print(f"  Максимум нейронов: {topology_stability['max_count']}")
    print(f"  Среднее нейронов: {topology_stability['mean_count']:.1f}")


def example_causality_analysis():
    """Анализ причинно-следственных связей"""
    print("\n" + "="*70)
    print("Пример 3: Анализ причинно-следственных связей")
    print("="*70)
    
    enn = EmergentNeuralNetwork(input_size=5, output_size=1)
    
    # Обучение с сохранением истории активаций
    activations_history = {nid: [] for nid in enn.neurons.keys()}
    
    x = torch.randn(100, 5)
    y = torch.sum(x[:, :2], dim=1, keepdim=True)  # Только первые 2 входа влияют
    
    for i in range(100):
        # Прямой проход
        with torch.no_grad():
            output = enn.forward(x[i:i+1])
            # Сохраняем активации (упрощенно)
            for nid in enn.neurons.keys():
                if nid in enn.input_ids:
                    idx = enn.input_ids.index(nid)
                    activations_history[nid].append(x[i, idx].item())
                else:
                    activations_history[nid].append(output[0, 0].item() if nid in enn.output_ids else 0.5)
        
        # Обучение
        enn.learn(x[i:i+1], y[i:i+1], reward=1.0)
    
    # Анализ причинности
    causality = CausalityAnalyzer()
    causal_graph = causality.build_causal_graph(activations_history)
    
    print(f"\nОбнаружено причинно-следственных связей: {len(causal_graph)}")
    
    # Показываем топ-5 связей
    sorted_connections = sorted(causal_graph.items(), key=lambda x: x[1], reverse=True)
    print("\nТоп-5 причинно-следственных связей:")
    for (src, tgt), strength in sorted_connections[:5]:
        print(f"  {src} → {tgt}: {strength:.4f}")


if __name__ == "__main__":
    example_basic_training()
    example_stability_analysis()
    example_causality_analysis()
    
    print("\n" + "="*70)
    print("Все примеры выполнены успешно!")
    print("="*70)

