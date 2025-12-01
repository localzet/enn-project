"""
Emergent Neural Network (ENN) - Алгоритм нейронной сети
Объединяет: динамическую топологию, термодинамику информации, 
эмерджентную архитектуру, причинно-следственную логику, мета-обучение

Научная новизна:
1. Динамическая топология - сеть сама определяет оптимальную структуру
2. Термодинамика информации - использование принципов энтропии и свободной энергии
3. Эмерджентная архитектура - структура возникает из взаимодействий
4. Причинно-следственная логика - понимание причин и следствий
5. Мета-обучение топологии - обучение на уровне архитектуры
6. Биологическая пластичность - адаптация связей как в мозге
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
from scipy.special import expit


@dataclass
class Connection:
    """Связь между нейронами с динамической силой"""
    source_id: int
    target_id: int
    weight: float
    strength: float = 1.0  # Сила связи (может изменяться)
    age: int = 0  # Возраст связи
    plasticity: float = 0.1  # Пластичность (способность к изменению)
    causal_strength: float = 0.0  # Сила причинно-следственной связи
    last_activation: float = 0.0  # Последняя активация
    entropy: float = 0.0  # Энтропия связи


@dataclass
class Neuron:
    """Нейрон с динамическими свойствами"""
    neuron_id: int
    activation: float = 0.0
    threshold: float = 0.5
    energy: float = 1.0  # Внутренняя энергия
    entropy: float = 0.0  # Энтропия нейрона
    free_energy: float = 0.0  # Свободная энергия (термодинамика)
    connections_in: List[int] = field(default_factory=list)  # ID входящих связей
    connections_out: List[int] = field(default_factory=list)  # ID исходящих связей
    activation_function: str = "adaptive"  # Адаптивная функция активации
    homeostasis: float = 0.5  # Гомеостаз (поддержание баланса)
    causal_importance: float = 0.0  # Важность в причинно-следственных цепочках
    emergence_level: float = 0.0  # Уровень эмерджентности


class EmergentNeuralNetwork:
    """
    Emergent Neural Network
    
    Ключевые инновации:
    1. Динамическая топология - структура сети определяется данными
    2. Термодинамика информации - использование свободной энергии
    3. Эмерджентная архитектура - структура возникает из взаимодействий
    4. Причинно-следственная логика - понимание причин
    5. Мета-обучение топологии - обучение на уровне архитектуры
    """
    
    def __init__(self, 
                 input_size: int = 10,
                 initial_neurons: int = 5,
                 max_neurons: int = 1000,
                 temperature: float = 1.0,  # Температура системы (термодинамика)
                 energy_budget: float = 100.0,  # Бюджет энергии
                 plasticity_rate: float = 0.1):
        """
        Инициализация сети
        
        Args:
            input_size: Размер входного вектора
            initial_neurons: Начальное количество нейронов
            max_neurons: Максимальное количество нейронов
            temperature: Температура системы (влияет на стохастичность)
            energy_budget: Бюджет энергии для создания новых нейронов
            plasticity_rate: Скорость пластичности
        """
        self.input_size = input_size
        self.max_neurons = max_neurons
        self.temperature = temperature
        self.energy_budget = energy_budget
        self.plasticity_rate = plasticity_rate
        
        # Нейроны и связи
        self.neurons: Dict[int, Neuron] = {}
        self.connections: Dict[int, Connection] = {}
        self.connection_counter = 0
        
        # Входные и выходные нейроны
        self.input_neurons: List[int] = []
        self.output_neurons: List[int] = []
        
        # Инициализация
        self._initialize_network(initial_neurons)
        
        # Статистика
        self.generation = 0
        self.total_energy_consumed = 0.0
        self.total_entropy = 0.0
        self.learning_history = []
        
        # Параметры эволюции
        self.emergence_threshold = 0.3  # Порог для создания новых нейронов
        self.causal_learning_rate = 0.01
        self.topology_learning_rate = 0.001
    
    def _initialize_network(self, initial_neurons: int):
        """Инициализация начальной сети"""
        # Создаем входные нейроны
        for i in range(self.input_size):
            neuron_id = i
            neuron = Neuron(
                neuron_id=neuron_id,
                threshold=0.5,
                energy=1.0,
                activation_function="linear"
            )
            self.neurons[neuron_id] = neuron
            self.input_neurons.append(neuron_id)
        
        # Создаем начальные скрытые нейроны
        for i in range(initial_neurons):
            neuron_id = self.input_size + i
            neuron = Neuron(
                neuron_id=neuron_id,
                threshold=np.random.uniform(0.3, 0.7),
                energy=1.0,
                activation_function="adaptive"
            )
            self.neurons[neuron_id] = neuron
        
        # Создаем выходные нейроны (начинаем с одного, будет расти)
        output_id = self.input_size + initial_neurons
        neuron = Neuron(
            neuron_id=output_id,
            threshold=0.5,
            energy=1.0,
            activation_function="adaptive"
        )
        self.neurons[output_id] = neuron
        self.output_neurons.append(output_id)
        
        # Создаем начальные связи (случайная топология)
        self._create_initial_connections()
    
    def _create_initial_connections(self):
        """Создание начальных связей"""
        neuron_ids = list(self.neurons.keys())
        
        # Связываем входы со скрытыми нейронами
        for input_id in self.input_neurons:
            for hidden_id in neuron_ids:
                if hidden_id not in self.input_neurons:
                    if np.random.random() < 0.3:  # 30% вероятность связи
                        self._add_connection(input_id, hidden_id)
        
        # Связываем скрытые нейроны с выходом
        for hidden_id in neuron_ids:
            if hidden_id not in self.input_neurons and hidden_id not in self.output_neurons:
                for output_id in self.output_neurons:
                    if np.random.random() < 0.5:
                        self._add_connection(hidden_id, output_id)
    
    def _add_connection(self, source_id: int, target_id: int, weight: Optional[float] = None) -> int:
        """Добавить связь между нейронами"""
        if weight is None:
            weight = np.random.randn() * 0.5
        
        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            strength=1.0,
            plasticity=self.plasticity_rate
        )
        
        connection_id = self.connection_counter
        self.connections[connection_id] = connection
        self.connection_counter += 1
        
        # Обновляем списки связей в нейронах
        if connection_id not in self.neurons[source_id].connections_out:
            self.neurons[source_id].connections_out.append(connection_id)
        if connection_id not in self.neurons[target_id].connections_in:
            self.neurons[target_id].connections_in.append(connection_id)
        
        return connection_id
    
    def _remove_connection(self, connection_id: int):
        """Удалить связь"""
        if connection_id in self.connections:
            conn = self.connections[connection_id]
            # Удаляем из списков нейронов
            if connection_id in self.neurons[conn.source_id].connections_out:
                self.neurons[conn.source_id].connections_out.remove(connection_id)
            if connection_id in self.neurons[conn.target_id].connections_in:
                self.neurons[conn.target_id].connections_in.remove(connection_id)
            del self.connections[connection_id]
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Прямой проход через сеть с динамической топологией
        
        Использует термодинамику информации для управления активацией
        """
        if len(inputs) != self.input_size:
            inputs = np.pad(inputs, (0, max(0, self.input_size - len(inputs))))
            inputs = inputs[:self.input_size]
        
        # Инициализация входов
        for i, input_id in enumerate(self.input_neurons):
            self.neurons[input_id].activation = inputs[i]
            self.neurons[input_id].energy = 1.0
        
        # Распространение активации (итеративный процесс)
        max_iterations = 50
        convergence_threshold = 0.001
        
        for iteration in range(max_iterations):
            prev_activations = {nid: n.activation for nid, n in self.neurons.items()}
            
            # Обновляем все нейроны (кроме входных)
            for neuron_id, neuron in self.neurons.items():
                if neuron_id in self.input_neurons:
                    continue
                
                # Собираем входные сигналы
                input_sum = 0.0
                total_strength = 0.0
                
                for conn_id in neuron.connections_in:
                    conn = self.connections[conn_id]
                    source_neuron = self.neurons[conn.source_id]
                    
                    # Взвешенный сигнал с учетом силы связи
                    signal = source_neuron.activation * conn.weight * conn.strength
                    input_sum += signal
                    total_strength += abs(conn.strength)
                
                # Нормализация
                if total_strength > 0:
                    input_sum /= total_strength
                
                # Вычисляем свободную энергию (термодинамика)
                free_energy = self._calculate_free_energy(neuron, input_sum)
                neuron.free_energy = free_energy
                
                # Активация с учетом термодинамики
                activation = self._activate_with_thermodynamics(neuron, input_sum, free_energy)
                
                # Обновляем состояние нейрона
                neuron.activation = activation
                neuron.entropy = self._calculate_entropy(neuron)
                
                # Обновляем энергию (потребление энергии)
                energy_consumption = abs(activation) * 0.01
                neuron.energy = max(0, neuron.energy - energy_consumption)
                self.total_energy_consumed += energy_consumption
            
            # Проверка сходимости
            max_change = max(
                abs(self.neurons[nid].activation - prev_activations[nid])
                for nid in self.neurons.keys()
                if nid not in self.input_neurons
            )
            
            if max_change < convergence_threshold:
                break
        
        # Собираем выходы
        outputs = np.array([
            self.neurons[output_id].activation
            for output_id in self.output_neurons
        ])
        
        return outputs
    
    def _calculate_free_energy(self, neuron: Neuron, input_sum: float) -> float:
        """
        Вычисление свободной энергии (термодинамика информации)
        
        Свободная энергия = Энергия - Температура * Энтропия
        Минимизация свободной энергии = максимизация информации
        """
        # Энергия системы
        energy = abs(input_sum) * neuron.energy
        
        # Энтропия (мера неопределенности)
        entropy = neuron.entropy if neuron.entropy > 0 else 0.001
        
        # Свободная энергия (принцип свободной энергии Фристона)
        free_energy = energy - self.temperature * entropy
        
        return free_energy
    
    def _activate_with_thermodynamics(self, neuron: Neuron, input_sum: float, free_energy: float) -> float:
        """
        Активация с учетом термодинамики
        
        Использует принцип минимизации свободной энергии
        """
        if neuron.activation_function == "adaptive":
            # Адаптивная активация на основе свободной энергии
            # Минимизируем свободную энергию
            if free_energy < 0:
                # Низкая свободная энергия = высокая активация
                activation = expit(-free_energy / self.temperature)
            else:
                # Высокая свободная энергия = низкая активация
                activation = expit(input_sum - neuron.threshold)
        elif neuron.activation_function == "linear":
            activation = np.tanh(input_sum)
        else:
            activation = expit(input_sum - neuron.threshold)
        
        # Гомеостаз - поддержание баланса
        activation = activation * (1 - neuron.homeostasis) + neuron.homeostasis * 0.5
        
        return np.clip(activation, -1.0, 1.0)
    
    def _calculate_entropy(self, neuron: Neuron) -> float:
        """Вычисление энтропии нейрона (мера неопределенности)"""
        # Энтропия на основе распределения активаций
        p = abs(neuron.activation)
        p = np.clip(p, 0.001, 0.999)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return entropy
    
    def learn(self, inputs: np.ndarray, targets: np.ndarray, reward: float = 0.0):
        """
        Обучение сети с динамической топологией
        
        Включает:
        1. Обучение весов
        2. Обучение топологии (создание/удаление связей)
        3. Обучение причинно-следственных связей
        4. Мета-обучение архитектуры
        """
        # Прямой проход
        outputs = self.forward(inputs)
        
        # Вычисляем ошибку
        if len(outputs) != len(targets):
            # Адаптируем количество выходов
            self._adapt_output_size(len(targets))
            outputs = self.forward(inputs)
        
        error = targets - outputs
        
        # 1. Обучение весов (backpropagation-like)
        self._update_weights(inputs, error, reward)
        
        # 2. Обучение топологии (создание/удаление связей)
        self._evolve_topology(inputs, error, reward)
        
        # 3. Обучение причинно-следственных связей
        self._learn_causality(inputs, outputs, error)
        
        # 4. Мета-обучение (создание новых нейронов при необходимости)
        if self._should_create_neuron(error, reward):
            self._create_emergent_neuron(inputs, error)
        
        # 5. Удаление неиспользуемых связей
        self._prune_connections()
        
        # Сохраняем историю
        self.learning_history.append({
            'error': np.mean(np.abs(error)),
            'reward': reward,
            'neurons': len(self.neurons),
            'connections': len(self.connections),
            'entropy': self._calculate_network_entropy()
        })
        
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
    
    def _update_weights(self, inputs: np.ndarray, error: np.ndarray, reward: float):
        """Обновление весов связей"""
        # Обратное распространение ошибки
        error_gradients = {}
        
        # Начинаем с выходных нейронов
        for i, output_id in enumerate(self.output_neurons):
            error_gradients[output_id] = error[i] * reward
        
        # Распространяем ошибку назад
        visited = set(self.output_neurons)
        queue = list(self.output_neurons)
        
        while queue:
            neuron_id = queue.pop(0)
            if neuron_id in self.input_neurons:
                continue
            
            neuron = self.neurons[neuron_id]
            gradient = error_gradients.get(neuron_id, 0.0)
            
            # Обновляем входящие связи
            for conn_id in neuron.connections_in:
                conn = self.connections[conn_id]
                source_neuron = self.neurons[conn.source_id]
                
                # Градиент для веса
                weight_gradient = gradient * source_neuron.activation * conn.strength
                
                # Обновляем вес с учетом пластичности
                learning_rate = self.plasticity_rate * conn.plasticity
                conn.weight += learning_rate * weight_gradient * reward
                
                # Обновляем силу связи
                conn.strength = np.clip(conn.strength + learning_rate * abs(weight_gradient), 0.0, 2.0)
                
                # Обновляем причинно-следственную силу
                if abs(weight_gradient) > 0.01:
                    conn.causal_strength = min(1.0, conn.causal_strength + self.causal_learning_rate)
                
                # Распространяем ошибку дальше
                if conn.source_id not in visited:
                    visited.add(conn.source_id)
                    queue.append(conn.source_id)
                    
                    if conn.source_id not in error_gradients:
                        error_gradients[conn.source_id] = 0.0
                    error_gradients[conn.source_id] += gradient * conn.weight * conn.strength
    
    def _evolve_topology(self, inputs: np.ndarray, error: np.ndarray, reward: float):
        """
        Эволюция топологии - создание и удаление связей
        
        Это ключевая инновация - сеть сама определяет оптимальную структуру
        """
        # Если ошибка большая, создаем новые связи
        avg_error = np.mean(np.abs(error))
        
        if avg_error > 0.3 and reward > 0:
            # Создаем связи между важными нейронами
            important_neurons = self._find_important_neurons()
            
            for source_id in important_neurons[:5]:
                for target_id in important_neurons[:5]:
                    if source_id != target_id:
                        # Проверяем, нет ли уже связи
                        existing = False
                        for conn in self.connections.values():
                            if conn.source_id == source_id and conn.target_id == target_id:
                                existing = True
                                break
                        
                        if not existing and np.random.random() < 0.1:
                            self._add_connection(source_id, target_id)
        
        # Ослабляем слабые связи
        for conn_id, conn in list(self.connections.items()):
            if abs(conn.weight) < 0.01 and conn.strength < 0.1:
                # Связь очень слабая, удаляем с вероятностью
                if np.random.random() < 0.05:
                    self._remove_connection(conn_id)
    
    def _learn_causality(self, inputs: np.ndarray, outputs: np.ndarray, error: np.ndarray):
        """
        Обучение причинно-следственных связей
        
        Определяет, какие нейроны являются причинами каких эффектов
        """
        # Анализируем активации и обновляем причинно-следственные связи
        for conn_id, conn in self.connections.items():
            source_neuron = self.neurons[conn.source_id]
            target_neuron = self.neurons[conn.target_id]
            
            # Если активация источника коррелирует с изменением цели
            if abs(source_neuron.activation) > 0.5:
                # Увеличиваем причинно-следственную силу
                correlation = abs(source_neuron.activation * target_neuron.activation)
                conn.causal_strength = min(1.0, conn.causal_strength + 
                                         self.causal_learning_rate * correlation)
                
                # Обновляем важность нейронов
                source_neuron.causal_importance = max(source_neuron.causal_importance,
                                                      conn.causal_strength)
                target_neuron.causal_importance = max(target_neuron.causal_importance,
                                                      conn.causal_strength)
    
    def _should_create_neuron(self, error: np.ndarray, reward: float) -> bool:
        """Определяет, нужно ли создать новый нейрон"""
        avg_error = np.mean(np.abs(error))
        
        # Создаем нейрон, если:
        # 1. Ошибка большая
        # 2. Есть энергия
        # 3. Есть место
        # 4. Награда положительная
        
        return (avg_error > 0.2 and 
                self.energy_budget > 10.0 and
                len(self.neurons) < self.max_neurons and
                reward > 0 and
                np.random.random() < 0.1)
    
    def _create_emergent_neuron(self, inputs: np.ndarray, error: np.ndarray):
        """
        Создание нового нейрона через эмерджентность
        
        Новый нейрон возникает из взаимодействий существующих
        """
        if len(self.neurons) >= self.max_neurons:
            return
        
        # Находим области с высокой активностью
        active_neurons = [
            (nid, n) for nid, n in self.neurons.items()
            if abs(n.activation) > 0.5 and nid not in self.input_neurons
        ]
        
        if len(active_neurons) < 2:
            return
        
        # Создаем новый нейрон
        new_id = max(self.neurons.keys()) + 1
        new_neuron = Neuron(
            neuron_id=new_id,
            threshold=np.random.uniform(0.3, 0.7),
            energy=1.0,
            activation_function="adaptive",
            emergence_level=1.0
        )
        
        self.neurons[new_id] = new_neuron
        
        # Связываем с активными нейронами
        for nid, _ in active_neurons[:3]:
            self._add_connection(nid, new_id)
        
        # Связываем с выходными нейронами
        for output_id in self.output_neurons:
            if np.random.random() < 0.5:
                self._add_connection(new_id, output_id)
        
        # Потребляем энергию
        self.energy_budget -= 5.0
    
    def _find_important_neurons(self) -> List[int]:
        """Находит важные нейроны (высокая причинно-следственная важность)"""
        neurons_with_importance = [
            (nid, n.causal_importance + abs(n.activation))
            for nid, n in self.neurons.items()
        ]
        neurons_with_importance.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in neurons_with_importance]
    
    def _prune_connections(self):
        """Удаление неиспользуемых связей"""
        for conn_id, conn in list(self.connections.items()):
            # Удаляем очень слабые связи
            if abs(conn.weight) < 0.001 and conn.strength < 0.05:
                if np.random.random() < 0.1:
                    self._remove_connection(conn_id)
    
    def _adapt_output_size(self, target_size: int):
        """Адаптирует количество выходных нейронов"""
        current_size = len(self.output_neurons)
        
        if target_size > current_size:
            # Добавляем выходные нейроны
            for i in range(target_size - current_size):
                new_id = max(self.neurons.keys()) + 1
                neuron = Neuron(
                    neuron_id=new_id,
                    threshold=0.5,
                    energy=1.0,
                    activation_function="adaptive"
                )
                self.neurons[new_id] = neuron
                self.output_neurons.append(new_id)
                
                # Связываем со скрытыми нейронами
                for hidden_id in self.neurons.keys():
                    if hidden_id not in self.input_neurons and hidden_id not in self.output_neurons:
                        if np.random.random() < 0.3:
                            self._add_connection(hidden_id, new_id)
    
    def _calculate_network_entropy(self) -> float:
        """Вычисляет общую энтропию сети"""
        total_entropy = 0.0
        for neuron in self.neurons.values():
            total_entropy += neuron.entropy
        return total_entropy / len(self.neurons) if self.neurons else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику сети"""
        return {
            'neurons': len(self.neurons),
            'connections': len(self.connections),
            'input_neurons': len(self.input_neurons),
            'output_neurons': len(self.output_neurons),
            'generation': self.generation,
            'total_energy_consumed': self.total_energy_consumed,
            'average_entropy': self._calculate_network_entropy(),
            'energy_budget': self.energy_budget,
            'temperature': self.temperature,
            'learning_history_size': len(self.learning_history)
        }
    
    def save(self, filepath: str):
        """Сохранить сеть"""
        data = {
            'input_size': self.input_size,
            'max_neurons': self.max_neurons,
            'temperature': self.temperature,
            'energy_budget': self.energy_budget,
            'generation': self.generation,
            'neurons': {
                str(nid): {
                    'neuron_id': n.neuron_id,
                    'threshold': n.threshold,
                    'energy': n.energy,
                    'activation_function': n.activation_function,
                    'homeostasis': n.homeostasis,
                    'causal_importance': n.causal_importance
                }
                for nid, n in self.neurons.items()
            },
            'connections': {
                str(cid): {
                    'source_id': c.source_id,
                    'target_id': c.target_id,
                    'weight': c.weight,
                    'strength': c.strength,
                    'causal_strength': c.causal_strength
                }
                for cid, c in self.connections.items()
            },
            'input_neurons': self.input_neurons,
            'output_neurons': self.output_neurons
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Загрузить сеть"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.input_size = data['input_size']
        self.max_neurons = data['max_neurons']
        self.temperature = data['temperature']
        self.energy_budget = data['energy_budget']
        self.generation = data['generation']
        
        # Восстанавливаем нейроны
        self.neurons = {}
        for nid_str, n_data in data['neurons'].items():
            neuron = Neuron(
                neuron_id=n_data['neuron_id'],
                threshold=n_data['threshold'],
                energy=n_data['energy'],
                activation_function=n_data['activation_function'],
                homeostasis=n_data['homeostasis'],
                causal_importance=n_data['causal_importance']
            )
            self.neurons[neuron.neuron_id] = neuron
        
        # Восстанавливаем связи
        self.connections = {}
        self.connection_counter = 0
        for cid_str, c_data in data['connections'].items():
            conn = Connection(
                source_id=c_data['source_id'],
                target_id=c_data['target_id'],
                weight=c_data['weight'],
                strength=c_data['strength'],
                causal_strength=c_data['causal_strength']
            )
            self.connections[self.connection_counter] = conn
            self.neurons[conn.source_id].connections_out.append(self.connection_counter)
            self.neurons[conn.target_id].connections_in.append(self.connection_counter)
            self.connection_counter += 1
        
        self.input_neurons = data['input_neurons']
        self.output_neurons = data['output_neurons']

