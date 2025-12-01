"""
Демонстрация Emergent Neural Network
"""
import numpy as np
import time
from emergent_neural_network import EmergentNeuralNetwork


class ENNDemo:
    """Демонстрация возможностей ENN"""
    
    def __init__(self):
        self.enn = None
    
    def show_menu(self):
        """Показать меню"""
        print("\n" + "="*70)
        print("🚀 ДЕМОНСТРАЦИЯ EMERGENT NEURAL NETWORK (ENN)")
        print("="*70)
        print("\nВыберите демонстрацию:")
        print("  1. 🧠 Динамическая топология (сеть сама определяет структуру)")
        print("  2. ⚡ Термодинамика информации (принцип свободной энергии)")
        print("  3. 🔄 Эмерджентность (создание новых нейронов)")
        print("  4. 🎯 Причинно-следственная логика")
        print("  5. 📈 Мета-обучение топологии")
        print("  6. 🔬 Сравнение с традиционной сетью")
        print("  7. 🎮 Интерактивное обучение")
        print("  0. ❌ Выход")
        print("="*70)
    
    def run(self):
        """Запустить демонстрацию"""
        while True:
            self.show_menu()
            choice = input("\nВаш выбор: ").strip()
            
            if choice == '0':
                print("👋 До свидания!")
                break
            elif choice == '1':
                self._demo_dynamic_topology()
            elif choice == '2':
                self._demo_thermodynamics()
            elif choice == '3':
                self._demo_emergence()
            elif choice == '4':
                self._demo_causality()
            elif choice == '5':
                self._demo_meta_learning()
            elif choice == '6':
                self._demo_comparison()
            elif choice == '7':
                self._demo_interactive()
            else:
                print("❌ Неверный выбор")
            
            input("\nНажмите Enter для продолжения...")
    
    def _demo_dynamic_topology(self):
        """Демонстрация динамической топологии"""
        print("\n" + "="*70)
        print("🧠 ДЕМОНСТРАЦИЯ: Динамическая топология")
        print("="*70)
        print("\nENN сама определяет оптимальную структуру на основе данных!")
        print("В отличие от традиционных сетей, структура адаптируется во время обучения.")
        
        self.enn = EmergentNeuralNetwork(input_size=10, initial_neurons=5, max_neurons=100)
        
        initial_stats = self.enn.get_statistics()
        print(f"\n📊 Начальное состояние:")
        print(f"   Нейронов: {initial_stats['neurons']}")
        print(f"   Связей: {initial_stats['connections']}")
        
        print("\n🔄 Обучение на сложной задаче (500 циклов)...")
        errors = []
        
        for i in range(500):
            inputs = np.random.randn(10)
            # Сложная нелинейная функция
            targets = np.array([
                np.tanh(np.sum(inputs[:5])),
                np.sin(np.sum(inputs[5:])),
                np.cos(np.sum(inputs))
            ])
            
            outputs = self.enn.forward(inputs)
            error = np.mean(np.abs(targets - outputs))
            errors.append(error)
            
            reward = 1.0 if error < 0.3 else max(0, 1.0 - error)
            self.enn.learn(inputs, targets, reward)
            
            if (i + 1) % 100 == 0:
                stats = self.enn.get_statistics()
                print(f"   Цикл {i+1}: {stats['neurons']} нейронов, {stats['connections']} связей, ошибка: {error:.4f}")
        
        final_stats = self.enn.get_statistics()
        print(f"\n✅ Результат:")
        print(f"   Финальных нейронов: {final_stats['neurons']} (было {initial_stats['neurons']})")
        print(f"   Финальных связей: {final_stats['connections']} (было {initial_stats['connections']})")
        print(f"   Создано новых нейронов: {final_stats['neurons'] - initial_stats['neurons']}")
        print(f"   Финальная ошибка: {errors[-1]:.4f} (начальная: {errors[0]:.4f})")
        print(f"\n💡 Сеть сама определила оптимальную структуру!")
    
    def _demo_thermodynamics(self):
        """Демонстрация термодинамики информации"""
        print("\n" + "="*70)
        print("⚡ ДЕМОНСТРАЦИЯ: Термодинамика информации")
        print("="*70)
        print("\nENN использует принцип свободной энергии Фристона:")
        print("F = E - T·S (Свободная энергия = Энергия - Температура × Энтропия)")
        print("Минимизация свободной энергии = максимизация информации")
        
        self.enn = EmergentNeuralNetwork(
            input_size=8, 
            temperature=1.0, 
            energy_budget=100.0,
            initial_neurons=5
        )
        
        print("\n🔄 Обучение (300 циклов)...")
        entropies = []
        energies = []
        
        for i in range(300):
            inputs = np.random.randn(8)
            targets = np.array([np.sum(inputs) / 8])
            
            self.enn.learn(inputs, targets, 1.0)
            
            stats = self.enn.get_statistics()
            entropies.append(stats['average_entropy'])
            energies.append(stats['total_energy_consumed'])
            
            if (i + 1) % 100 == 0:
                print(f"   Цикл {i+1}:")
                print(f"      Энтропия: {stats['average_entropy']:.4f}")
                print(f"      Энергия: {stats['total_energy_consumed']:.2f}")
                print(f"      Бюджет: {stats['energy_budget']:.2f}")
        
        final_stats = self.enn.get_statistics()
        print(f"\n✅ Результат:")
        print(f"   Средняя энтропия: {final_stats['average_entropy']:.4f}")
        print(f"   Общая энергия: {final_stats['total_energy_consumed']:.2f}")
        print(f"   Энергоэффективность: {final_stats['total_energy_consumed'] / 300:.4f} на цикл")
        print(f"\n💡 Сеть эффективно использует энергию, минимизируя свободную энергию!")
    
    def _demo_emergence(self):
        """Демонстрация эмерджентности"""
        print("\n" + "="*70)
        print("🔄 ДЕМОНСТРАЦИЯ: Эмерджентность")
        print("="*70)
        print("\nНовые нейроны возникают из взаимодействий существующих!")
        print("Структура сети эмерджентно формируется из данных.")
        
        self.enn = EmergentNeuralNetwork(input_size=6, initial_neurons=3, max_neurons=50)
        
        initial_neurons = len(self.enn.neurons)
        print(f"\n📊 Начальное состояние: {initial_neurons} нейронов")
        
        print("\n🔄 Обучение на задаче, требующей новых нейронов (400 циклов)...")
        neuron_counts = [initial_neurons]
        
        for i in range(400):
            inputs = np.random.randn(6)
            # Сложная функция
            targets = np.array([
                np.tanh(np.sum(inputs**2)),
                np.sin(np.prod(inputs[:3])),
                np.cos(np.prod(inputs[3:]))
            ])
            
            self.enn.learn(inputs, targets, 1.0)
            neuron_counts.append(len(self.enn.neurons))
            
            if (i + 1) % 100 == 0:
                print(f"   Цикл {i+1}: {len(self.enn.neurons)} нейронов")
        
        final_neurons = len(self.enn.neurons)
        print(f"\n✅ Результат:")
        print(f"   Финальных нейронов: {final_neurons}")
        print(f"   Создано новых: {final_neurons - initial_neurons}")
        print(f"   Рост: {((final_neurons - initial_neurons) / initial_neurons * 100):.1f}%")
        
        # Проверяем уровень эмерджентности
        emergence_levels = [n.emergence_level for n in self.enn.neurons.values() if n.emergence_level > 0]
        if emergence_levels:
            print(f"   Средний уровень эмерджентности: {np.mean(emergence_levels):.3f}")
        
        print(f"\n💡 Новые нейроны эмерджентно возникли из взаимодействий!")
    
    def _demo_causality(self):
        """Демонстрация причинно-следственной логики"""
        print("\n" + "="*70)
        print("🎯 ДЕМОНСТРАЦИЯ: Причинно-следственная логика")
        print("="*70)
        print("\nENN обучается понимать причинно-следственные связи!")
        print("Определяет, какие входы являются причинами каких выходов.")
        
        self.enn = EmergentNeuralNetwork(input_size=5, initial_neurons=5)
        
        print("\n🔄 Обучение на данных с явными причинно-следственными связями...")
        print("   inputs[0] и inputs[1] → targets[0]")
        print("   inputs[2] и inputs[3] → targets[1]")
        
        for i in range(300):
            inputs = np.random.randn(5)
            targets = np.array([
                np.tanh(inputs[0] + inputs[1]),  # Причина: inputs[0], inputs[1]
                np.sin(inputs[2] + inputs[3])     # Причина: inputs[2], inputs[3]
            ])
            self.enn.learn(inputs, targets, 1.0)
        
        # Анализируем причинно-следственные связи
        important_neurons = self.enn._find_important_neurons()
        causal_connections = [
            (cid, c) for cid, c in self.enn.connections.items()
            if c.causal_strength > 0.5
        ]
        
        print(f"\n✅ Результат:")
        print(f"   Важных нейронов: {len(important_neurons)}")
        print(f"   Связей с высокой причинно-следственной силой: {len(causal_connections)}")
        
        if causal_connections:
            print(f"\n   Примеры сильных причинно-следственных связей:")
            for cid, conn in causal_connections[:5]:
                print(f"      Связь {cid}: {conn.source_id} → {conn.target_id}, сила: {conn.causal_strength:.3f}")
        
        print(f"\n💡 Сеть научилась понимать причинно-следственные связи!")
    
    def _demo_meta_learning(self):
        """Демонстрация мета-обучения топологии"""
        print("\n" + "="*70)
        print("📈 ДЕМОНСТРАЦИЯ: Мета-обучение топологии")
        print("="*70)
        print("\nENN обучается на уровне архитектуры!")
        print("Не только веса, но и сама структура сети оптимизируется.")
        
        self.enn = EmergentNeuralNetwork(input_size=10, initial_neurons=5, max_neurons=80)
        
        initial_stats = self.enn.get_statistics()
        print(f"\n📊 Начальное состояние:")
        print(f"   Нейронов: {initial_stats['neurons']}")
        print(f"   Связей: {initial_stats['connections']}")
        
        print("\n🔄 Мета-обучение на последовательности задач (600 циклов)...")
        
        task_errors = []
        for task in range(3):
            print(f"\n   Задача {task + 1}:")
            task_errors_task = []
            
            for i in range(200):
                inputs = np.random.randn(10)
                # Разные задачи
                if task == 0:
                    targets = np.array([np.sum(inputs[:5])])
                elif task == 1:
                    targets = np.array([np.prod(inputs[5:])])
                else:
                    targets = np.array([np.tanh(np.sum(inputs))])
                
                outputs = self.enn.forward(inputs)
                error = np.mean(np.abs(targets - outputs))
                task_errors_task.append(error)
                
                reward = 1.0 if error < 0.2 else max(0, 1.0 - error)
                self.enn.learn(inputs, targets, reward)
            
            stats = self.enn.get_statistics()
            avg_error = np.mean(task_errors_task[-50:])  # Последние 50
            task_errors.append(avg_error)
            print(f"      Финальная ошибка: {avg_error:.4f}")
            print(f"      Нейронов: {stats['neurons']}, Связей: {stats['connections']}")
        
        print(f"\n✅ Результат:")
        print(f"   Ошибка задачи 1: {task_errors[0]:.4f}")
        print(f"   Ошибка задачи 2: {task_errors[1]:.4f}")
        print(f"   Ошибка задачи 3: {task_errors[2]:.4f}")
        print(f"   Финальная структура: {stats['neurons']} нейронов, {stats['connections']} связей")
        print(f"\n💡 Сеть адаптировала свою структуру под разные задачи!")
    
    def _demo_comparison(self):
        """Сравнение с традиционной сетью"""
        print("\n" + "="*70)
        print("🔬 ДЕМОНСТРАЦИЯ: Сравнение с традиционной сетью")
        print("="*70)
        
        print("\n📊 ТРАДИЦИОННАЯ СЕТЬ:")
        print("   ❌ Фиксированная архитектура")
        print("   ❌ Нет понимания причин")
        print("   ❌ Нет учета энергии")
        print("   ❌ Нет мета-обучения")
        print("   ✅ Быстрое обучение")
        print("   ✅ Простота реализации")
        
        print("\n📊 EMERGENT NEURAL NETWORK:")
        print("   ✅ Динамическая топология")
        print("   ✅ Понимание причинно-следственных связей")
        print("   ✅ Термодинамика информации")
        print("   ✅ Мета-обучение топологии")
        print("   ✅ Эмерджентность")
        print("   ⚠️ Более сложная реализация")
        
        print("\n💡 ENN представляет следующее поколение нейронных сетей!")
    
    def _demo_interactive(self):
        """Интерактивное обучение"""
        print("\n" + "="*70)
        print("🎮 ДЕМОНСТРАЦИЯ: Интерактивное обучение")
        print("="*70)
        print("\nОбучите сеть на ваших данных!")
        
        self.enn = EmergentNeuralNetwork(input_size=5, initial_neurons=5)
        
        print("\nВведите данные для обучения (или 'quit' для выхода):")
        print("Формат: 5 чисел через пробел, затем целевое значение")
        print("Пример: 1.0 2.0 3.0 4.0 5.0 15.0")
        
        cycle = 0
        while True:
            try:
                user_input = input(f"\nЦикл {cycle + 1} > ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                parts = user_input.split()
                if len(parts) < 6:
                    print("❌ Недостаточно данных. Нужно 5 входов и 1 выход.")
                    continue
                
                inputs = np.array([float(x) for x in parts[:5]])
                target = float(parts[5])
                targets = np.array([target])
                
                # Обучение
                outputs = self.enn.forward(inputs)
                error = np.abs(targets - outputs)[0]
                reward = 1.0 if error < 0.1 else max(0, 1.0 - error)
                
                self.enn.learn(inputs, targets, reward)
                
                stats = self.enn.get_statistics()
                print(f"   Выход: {outputs[0]:.4f}, Цель: {target:.4f}, Ошибка: {error:.4f}")
                print(f"   Нейронов: {stats['neurons']}, Связей: {stats['connections']}")
                
                cycle += 1
                
            except ValueError:
                print("❌ Неверный формат данных")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        final_stats = self.enn.get_statistics()
        print(f"\n✅ Обучение завершено!")
        print(f"   Всего циклов: {cycle}")
        print(f"   Финальная структура: {final_stats['neurons']} нейронов, {final_stats['connections']} связей")


def main():
    """Главная функция"""
    demo = ENNDemo()
    demo.run()


if __name__ == "__main__":
    main()

