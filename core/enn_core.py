"""
Ядро Emergent Neural Network - упрощенная и формализованная версия

Ключевые идеи:
1. Динамическая топология - сеть сама определяет структуру
2. Энергобюджет - ограничение на создание новых нейронов
3. Формальные правила эволюции топологии

Математическая формализация:
- Правила создания нейронов основаны на ошибке и энергобюджете
- Правила удаления основаны на активности и важности
- Стабильность гарантируется через энергетические ограничения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math
from .topology_evolution import TopologyEvolution
from .energy_management import EnergyManager


@dataclass
class Neuron:
    """Нейрон с формализованными свойствами"""
    neuron_id: int
    activation: float = 0.0
    threshold: float = 0.5
    energy: float = 1.0
    importance: float = 0.0  # Важность нейрона
    age: int = 0
    activation_history: List[float] = field(default_factory=list)


@dataclass
class Connection:
    """Связь между нейронами"""
    source_id: int
    target_id: int
    weight: torch.Tensor
    strength: float = 1.0
    age: int = 0
    importance: float = 0.0


# TopologyEvolution и EnergyManager импортируются из отдельных модулей


class EmergentNeuralNetwork(nn.Module):
    """
    Ядро Emergent Neural Network
    
    Упрощенная версия с фокусом на:
    1. Динамическая топология (формализованная)
    2. Энергобюджет (управление ресурсами)
    3. Стабильность (гарантии сходимости)
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 initial_hidden: int = 5,
                 max_neurons: int = 1000,
                 learning_rate: float = 0.01,
                 energy_budget: float = 100.0):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.max_neurons = max_neurons
        self.learning_rate = learning_rate
        
        # Компоненты
        self.energy_manager = EnergyManager(initial_budget=energy_budget)
        self.topology_evolution = TopologyEvolution()
        
        # Нейроны и связи
        self.neurons: Dict[int, Neuron] = {}
        self.connections: Dict[Tuple[int, int], Connection] = {}
        self.neuron_counter = input_size + output_size
        
        # Входные и выходные нейроны
        self.input_ids = list(range(input_size))
        self.output_ids = list(range(input_size, input_size + output_size))
        
        # Инициализация
        self._initialize_network(initial_hidden)
        
        # Статистика
        self.generation = 0
        self.total_energy_consumed = 0.0
        
        # PyTorch параметры
        self._register_parameters()
    
    def _initialize_network(self, initial_hidden: int):
        """Инициализация начальной сети"""
        # Входные нейроны
        for i in self.input_ids:
            self.neurons[i] = Neuron(neuron_id=i, threshold=0.0)
        
        # Выходные нейроны
        for i in self.output_ids:
            self.neurons[i] = Neuron(neuron_id=i, threshold=0.5)
        
        # Начальные скрытые нейроны
        start_id = len(self.input_ids) + len(self.output_ids)
        for i in range(initial_hidden):
            neuron_id = start_id + i
            self.neurons[neuron_id] = Neuron(
                neuron_id=neuron_id,
                threshold=np.random.uniform(0.3, 0.7)
            )
        
        # Начальные связи
        self._create_initial_connections()
    
    def _create_initial_connections(self):
        """Создание начальных связей"""
        hidden_ids = [nid for nid in self.neurons.keys() 
                     if nid not in self.input_ids and nid not in self.output_ids]
        
        # Связи вход -> скрытые
        for input_id in self.input_ids:
            for hidden_id in hidden_ids[:3]:  # Связываем с первыми 3 скрытыми
                weight = torch.randn(1) * 0.5
                self.connections[(input_id, hidden_id)] = Connection(
                    source_id=input_id,
                    target_id=hidden_id,
                    weight=weight
                )
        
        # Связи скрытые -> выход
        for hidden_id in hidden_ids:
            for output_id in self.output_ids:
                weight = torch.randn(1) * 0.5
                self.connections[(hidden_id, output_id)] = Connection(
                    source_id=hidden_id,
                    target_id=output_id,
                    weight=weight
                )
    
    def _register_parameters(self):
        """Регистрация параметров PyTorch"""
        # Регистрируем веса связей как параметры
        for (src, tgt), conn in self.connections.items():
            param_name = f"weight_{src}_{tgt}"
            self.register_parameter(param_name, nn.Parameter(conn.weight))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входной тензор [batch_size, input_size]
        
        Returns:
            Выходной тензор [batch_size, output_size]
        """
        batch_size = x.shape[0]
        
        # Инициализация активаций
        activations = {}
        for nid in self.neurons.keys():
            if nid in self.input_ids:
                idx = self.input_ids.index(nid)
                activations[nid] = x[:, idx]
            else:
                activations[nid] = torch.zeros(batch_size)
        
        # Итеративное распространение (до сходимости или максимум итераций)
        max_iterations = 20
        convergence_threshold = 0.001
        
        for iteration in range(max_iterations):
            prev_activations = {k: v.clone() for k, v in activations.items()}
            
            # Обновляем все нейроны (кроме входных)
            for neuron_id, neuron in self.neurons.items():
                if neuron_id in self.input_ids:
                    continue
                
                # Собираем входные сигналы
                input_sum = torch.zeros(batch_size)
                
                for (src, tgt), conn in self.connections.items():
                    if tgt == neuron_id:
                        input_sum += activations[src] * conn.weight.squeeze()
                
                # Активация
                activation = torch.tanh(input_sum - neuron.threshold)
                activations[neuron_id] = activation
            
            # Проверка сходимости
            max_change = max(
                (activations[nid] - prev_activations[nid]).abs().max().item()
                for nid in activations.keys()
                if nid not in self.input_ids
            )
            
            if max_change < convergence_threshold:
                break
        
        # Собираем выходы
        outputs = torch.stack([activations[oid] for oid in self.output_ids], dim=1)
        
        # Потребление энергии
        all_activations = [a.mean().item() for a in activations.values()]
        self.energy_manager.consume(all_activations)
        
        return outputs
    
    def learn(self, 
              x: torch.Tensor,
              y: torch.Tensor,
              reward: float = 1.0) -> Dict[str, float]:
        """
        Обучение сети
        
        Args:
            x: Входные данные
            y: Целевые значения
            reward: Награда за правильное предсказание
        
        Returns:
            Словарь с метриками обучения
        """
        # Прямой проход
        outputs = self.forward(x)
        
        # Вычисление ошибки
        error = F.mse_loss(outputs, y)
        error_value = error.item()
        
        # Обратное распространение
        error.backward()
        
        # Обновление весов
        with torch.no_grad():
            for (src, tgt), conn in self.connections.items():
                if conn.weight.grad is not None:
                    conn.weight -= self.learning_rate * conn.weight.grad
                    conn.weight.grad.zero_()
        
        # Обновление важности нейронов
        self._update_importance(outputs, y)
        
        # Генерация энергии
        self.energy_manager.generate(reward)
        
        # Эволюция топологии
        metrics = self._evolve_topology(error_value, reward)
        
        # Старение нейронов
        for neuron in self.neurons.values():
            neuron.age += 1
        
        self.generation += 1
        
        metrics.update({
            'error': error_value,
            'energy_budget': self.energy_manager.budget,
            'num_neurons': len(self.neurons),
            'num_connections': len(self.connections)
        })
        
        return metrics
    
    def _update_importance(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Обновление важности нейронов на основе вклада в ошибку"""
        # Упрощенная версия: важность = корреляция с ошибкой
        error = (outputs - targets).abs().mean(dim=0)
        
        for neuron_id, neuron in self.neurons.items():
            if neuron_id in self.output_ids:
                idx = self.output_ids.index(neuron_id)
                neuron.importance = error[idx].item()
            else:
                # Для скрытых: важность = средняя важность связанных выходов
                importance_sum = 0.0
                count = 0
                for (src, tgt), conn in self.connections.items():
                    if src == neuron_id and tgt in self.output_ids:
                        idx = self.output_ids.index(tgt)
                        importance_sum += error[idx].item()
                        count += 1
                if count > 0:
                    neuron.importance = importance_sum / count
    
    def _evolve_topology(self, error: float, reward: float) -> Dict[str, float]:
        """
        Эволюция топологии сети
        
        Returns:
            Метрики эволюции
        """
        metrics = {
            'neurons_created': 0,
            'neurons_removed': 0,
            'connections_added': 0,
            'connections_removed': 0
        }
        
        # Создание новых нейронов
        if (len(self.neurons) < self.max_neurons and
            self.topology_evolution.should_create_neuron(
                error, self.energy_manager.budget, len(self.neurons)
            )):
            cost = self.topology_evolution.calculate_creation_cost(len(self.neurons))
            if self.energy_manager.spend(cost):
                self._create_neuron()
                metrics['neurons_created'] = 1
        
        # Удаление неважных нейронов
        neurons_to_remove = []
        for neuron_id, neuron in self.neurons.items():
            if (neuron_id not in self.input_ids and 
                neuron_id not in self.output_ids and
                self.topology_evolution.should_remove_neuron(neuron, 0.1)):
                neurons_to_remove.append(neuron_id)
        
        for neuron_id in neurons_to_remove:
            energy_return = self.topology_evolution.calculate_energy_return(
                self.neurons[neuron_id]
            )
            self.energy_manager.budget += energy_return
            self._remove_neuron(neuron_id)
            metrics['neurons_removed'] += 1
        
        return metrics
    
    def _create_neuron(self):
        """Создание нового нейрона"""
        new_id = self.neuron_counter
        self.neuron_counter += 1
        
        neuron = Neuron(
            neuron_id=new_id,
            threshold=np.random.uniform(0.3, 0.7)
        )
        self.neurons[new_id] = neuron
        
        # Связываем с активными нейронами
        important_neurons = sorted(
            [(nid, n.importance) for nid, n in self.neurons.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for src_id, _ in important_neurons:
            if src_id != new_id:
                weight = torch.randn(1) * 0.5
                self.connections[(src_id, new_id)] = Connection(
                    source_id=src_id,
                    target_id=new_id,
                    weight=weight
                )
        
        # Связываем с выходами
        for output_id in self.output_ids:
            weight = torch.randn(1) * 0.5
            self.connections[(new_id, output_id)] = Connection(
                source_id=new_id,
                target_id=output_id,
                weight=weight
            )
    
    def _remove_neuron(self, neuron_id: int):
        """Удаление нейрона"""
        # Удаляем все связи с этим нейроном
        connections_to_remove = [
            (src, tgt) for (src, tgt) in self.connections.keys()
            if src == neuron_id or tgt == neuron_id
        ]
        for conn_key in connections_to_remove:
            del self.connections[conn_key]
        
        # Удаляем нейрон
        del self.neurons[neuron_id]
    
    def get_statistics(self) -> Dict[str, float]:
        """Получить статистику сети"""
        return {
            'num_neurons': len(self.neurons),
            'num_connections': len(self.connections),
            'energy_budget': self.energy_manager.budget,
            'generation': self.generation,
            'total_energy_consumed': self.total_energy_consumed
        }

