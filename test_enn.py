"""
Тестирование Emergent Neural Network
"""
import numpy as np
from emergent_neural_network import EmergentNeuralNetwork


def test_basic_functionality():
    """Базовый тест функциональности"""
    print("🧪 Тест 1: Базовая функциональность")
    
    # Создаем сеть
    enn = EmergentNeuralNetwork(input_size=5, initial_neurons=3, max_neurons=50)
    
    # Тестовые данные
    inputs = np.random.randn(5)
    targets = np.array([0.5, 0.3])
    
    # Прямой проход
    outputs = enn.forward(inputs)
    print(f"  ✅ Прямой проход: вход {inputs.shape} → выход {outputs.shape}")
    
    # Обучение
    for i in range(100):
        inputs = np.random.randn(5)
        targets = np.array([np.sin(np.sum(inputs)), np.cos(np.sum(inputs))])
        reward = 1.0 if np.mean(np.abs(targets - enn.forward(inputs))) < 0.5 else 0.0
        enn.learn(inputs, targets, reward)
    
    stats = enn.get_statistics()
    print(f"  ✅ Обучение завершено:")
    print(f"     Нейронов: {stats['neurons']}")
    print(f"     Связей: {stats['connections']}")
    print(f"     Энергия потреблена: {stats['total_energy_consumed']:.2f}")
    print()


def test_dynamic_topology():
    """Тест динамической топологии"""
    print("🧪 Тест 2: Динамическая топология")
    
    enn = EmergentNeuralNetwork(input_size=10, initial_neurons=5, max_neurons=100)
    
    initial_stats = enn.get_statistics()
    print(f"  Начальное состояние: {initial_stats['neurons']} нейронов, {initial_stats['connections']} связей")
    
    # Обучение на сложной задаче
    for i in range(500):
        inputs = np.random.randn(10)
        # Сложная нелинейная функция
        targets = np.array([
            np.tanh(np.sum(inputs[:5])),
            np.sin(np.sum(inputs[5:])),
            np.cos(np.sum(inputs))
        ])
        reward = 1.0
        enn.learn(inputs, targets, reward)
        
        if (i + 1) % 100 == 0:
            stats = enn.get_statistics()
            print(f"  Цикл {i+1}: {stats['neurons']} нейронов, {stats['connections']} связей")
    
    final_stats = enn.get_statistics()
    print(f"  ✅ Финальное состояние: {final_stats['neurons']} нейронов, {final_stats['connections']} связей")
    print(f"     Создано новых нейронов: {final_stats['neurons'] - initial_stats['neurons']}")
    print()


def test_thermodynamics():
    """Тест термодинамики информации"""
    print("🧪 Тест 3: Термодинамика информации")
    
    enn = EmergentNeuralNetwork(input_size=8, temperature=1.0, energy_budget=100.0)
    
    # Обучение
    for i in range(200):
        inputs = np.random.randn(8)
        targets = np.array([np.sum(inputs) / 8])
        reward = 1.0
        enn.learn(inputs, targets, reward)
    
    stats = enn.get_statistics()
    print(f"  ✅ Термодинамические параметры:")
    print(f"     Средняя энтропия: {stats['average_entropy']:.4f}")
    print(f"     Энергия потреблена: {stats['total_energy_consumed']:.2f}")
    print(f"     Энергетический бюджет: {stats['energy_budget']:.2f}")
    print(f"     Температура: {stats['temperature']:.2f}")
    print()


def test_causality():
    """Тест причинно-следственной логики"""
    print("🧪 Тест 4: Причинно-следственная логика")
    
    enn = EmergentNeuralNetwork(input_size=5, initial_neurons=5)
    
    # Создаем данные с явными причинно-следственными связями
    # inputs[0] и inputs[1] влияют на targets[0]
    # inputs[2] и inputs[3] влияют на targets[1]
    
    for i in range(300):
        inputs = np.random.randn(5)
        targets = np.array([
            np.tanh(inputs[0] + inputs[1]),  # Причина: inputs[0], inputs[1]
            np.sin(inputs[2] + inputs[3])     # Причина: inputs[2], inputs[3]
        ])
        reward = 1.0
        enn.learn(inputs, targets, reward)
    
    # Проверяем причинно-следственные связи
    important_neurons = enn._find_important_neurons()
    print(f"  ✅ Важные нейроны (причинно-следственная важность): {important_neurons[:5]}")
    
    # Проверяем связи с высокой причинно-следственной силой
    causal_connections = [
        (cid, c) for cid, c in enn.connections.items()
        if c.causal_strength > 0.5
    ]
    print(f"  ✅ Связи с высокой причинно-следственной силой: {len(causal_connections)}")
    print()


def test_emergence():
    """Тест эмерджентности"""
    print("🧪 Тест 5: Эмерджентность")
    
    enn = EmergentNeuralNetwork(input_size=6, initial_neurons=3, max_neurons=50)
    
    initial_neurons = len(enn.neurons)
    
    # Обучение на задаче, требующей новых нейронов
    for i in range(400):
        inputs = np.random.randn(6)
        # Сложная функция, требующая новых нейронов
        targets = np.array([
            np.tanh(np.sum(inputs**2)),
            np.sin(np.prod(inputs[:3])),
            np.cos(np.prod(inputs[3:]))
        ])
        reward = 1.0
        enn.learn(inputs, targets, reward)
    
    final_neurons = len(enn.neurons)
    new_neurons = final_neurons - initial_neurons
    
    print(f"  ✅ Эмерджентное создание нейронов:")
    print(f"     Начало: {initial_neurons} нейронов")
    print(f"     Конец: {final_neurons} нейронов")
    print(f"     Создано новых: {new_neurons}")
    
    # Проверяем уровень эмерджентности
    emergence_levels = [n.emergence_level for n in enn.neurons.values() if n.emergence_level > 0]
    if emergence_levels:
        print(f"     Средний уровень эмерджентности: {np.mean(emergence_levels):.3f}")
    print()


def test_save_load():
    """Тест сохранения и загрузки"""
    print("🧪 Тест 6: Сохранение и загрузка")
    
    # Создаем и обучаем сеть
    enn1 = EmergentNeuralNetwork(input_size=5, initial_neurons=5)
    for i in range(100):
        inputs = np.random.randn(5)
        targets = np.array([np.sum(inputs)])
        enn1.learn(inputs, targets, 1.0)
    
    stats1 = enn1.get_statistics()
    print(f"  Исходная сеть: {stats1['neurons']} нейронов, {stats1['connections']} связей")
    
    # Сохраняем
    enn1.save("test_enn.json")
    print("  ✅ Сеть сохранена")
    
    # Загружаем
    enn2 = EmergentNeuralNetwork(input_size=5)
    enn2.load("test_enn.json")
    stats2 = enn2.get_statistics()
    print(f"  Загруженная сеть: {stats2['neurons']} нейронов, {stats2['connections']} связей")
    
    # Проверяем, что структура совпадает
    assert stats1['neurons'] == stats2['neurons'], "Количество нейронов не совпадает"
    assert stats1['connections'] == stats2['connections'], "Количество связей не совпадает"
    print("  ✅ Структура сети совпадает")
    print()


def run_all_tests():
    """Запустить все тесты"""
    print("="*70)
    print("ТЕСТИРОВАНИЕ EMERGENT NEURAL NETWORK (ENN)")
    print("="*70)
    print()
    
    try:
        test_basic_functionality()
        test_dynamic_topology()
        test_thermodynamics()
        test_causality()
        test_emergence()
        test_save_load()
        
        print("="*70)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

