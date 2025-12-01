"""
ENN для ресурсно-ограниченных систем

Специализация для систем с жесткими ограничениями:
- Мобильные устройства
- IoT устройства
- Встраиваемые системы
"""

import torch
from typing import Dict, Optional
import numpy as np

import torch
from typing import Dict, Optional
import numpy as np
import sys
import os

# Добавляем путь к корню проекта для импорта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.enn_core import EmergentNeuralNetwork
from core.energy_management import EnergyManager


class ResourceConstrainedENN(EmergentNeuralNetwork):
    """
    ENN для ресурсно-ограниченных систем
    
    Особенности:
    1. Строгое управление энергопотреблением
    2. Ограничение размера сети
    3. Приоритизация вычислений
    4. Адаптация к доступным ресурсам
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 max_energy_per_step: float = 1.0,
                 max_neurons: int = 50,  # Меньше для ограниченных систем
                 energy_efficiency_mode: bool = True,
                 **kwargs):
        """
        Args:
            input_size: Размер входа
            output_size: Размер выхода
            max_energy_per_step: Максимальная энергия на шаг
            max_neurons: Максимальное количество нейронов
            energy_efficiency_mode: Режим энергоэффективности
            **kwargs: Дополнительные параметры
        """
        super().__init__(input_size, output_size, max_neurons=max_neurons, **kwargs)
        
        self.max_energy_per_step = max_energy_per_step
        self.energy_efficiency_mode = energy_efficiency_mode
        
        # Статистика ресурсов
        self.energy_usage_history = []
        self.computation_time_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход с учетом энергетических ограничений
        
        В режиме энергоэффективности:
        - Пропускаем менее важные нейроны
        - Используем упрощенные вычисления
        """
        if self.energy_efficiency_mode:
            return self._energy_efficient_forward(x)
        else:
            return super().forward(x)
    
    def _energy_efficient_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Энергоэффективный прямой проход"""
        # Используем только важные нейроны
        important_neurons = {
            nid: n for nid, n in self.neurons.items()
            if n.importance > 0.1 or nid in self.input_ids or nid in self.output_ids
        }
        
        # Упрощенный проход только через важные нейроны
        batch_size = x.shape[0]
        activations = {}
        
        # Инициализация входов
        for nid in self.input_ids:
            idx = self.input_ids.index(nid)
            activations[nid] = x[:, idx]
        
        # Ограниченное распространение
        for neuron_id, neuron in important_neurons.items():
            if neuron_id in self.input_ids:
                continue
            
            input_sum = torch.zeros(batch_size)
            for (src, tgt), conn in self.connections.items():
                if tgt == neuron_id and src in activations:
                    input_sum += activations[src] * conn.weight.squeeze()
            
            activation = torch.tanh(input_sum - neuron.threshold)
            activations[neuron_id] = activation
        
        # Собираем выходы
        outputs = torch.stack([activations[oid] for oid in self.output_ids], dim=1)
        
        return outputs
    
    def learn(self, x: torch.Tensor, y: torch.Tensor, reward: float = 1.0) -> Dict[str, float]:
        """
        Обучение с учетом энергетических ограничений
        """
        # Проверяем доступную энергию
        if not self.energy_manager.can_afford(self.max_energy_per_step):
            # Недостаточно энергии - пропускаем обучение
            return {
                'error': float('inf'),
                'energy_budget': self.energy_manager.budget,
                'skipped': True
            }
        
        # Обычное обучение
        metrics = super().learn(x, y, reward)
        
        # Отслеживание энергопотребления
        energy_used = self.max_energy_per_step
        self.energy_usage_history.append(energy_used)
        
        metrics['energy_used'] = energy_used
        metrics['energy_efficiency'] = 1.0 / (energy_used + 1e-6)
        
        return metrics
    
    def adapt_to_resources(self, available_energy: float, available_memory: int):
        """
        Адаптация к доступным ресурсам
        
        Args:
            available_energy: Доступная энергия
            available_memory: Доступная память (в количестве нейронов)
        """
        # Обновляем энергобюджет
        self.energy_manager.budget = min(available_energy, self.energy_manager.budget)
        
        # Ограничиваем размер сети
        if len(self.neurons) > available_memory:
            # Удаляем наименее важные нейроны
            neurons_to_remove = len(self.neurons) - available_memory
            sorted_neurons = sorted(
                [(nid, n.importance) for nid, n in self.neurons.items()
                 if nid not in self.input_ids and nid not in self.output_ids],
                key=lambda x: x[1]
            )
            
            for nid, _ in sorted_neurons[:neurons_to_remove]:
                self._remove_neuron(nid)
        
        # Адаптируем энергоэффективность
        if available_energy < 10.0:
            self.energy_efficiency_mode = True
        else:
            self.energy_efficiency_mode = False
    
    def get_resource_statistics(self) -> Dict[str, float]:
        """Статистика использования ресурсов"""
        return {
            'current_energy': self.energy_manager.budget,
            'num_neurons': len(self.neurons),
            'num_connections': len(self.connections),
            'average_energy_per_step': np.mean(self.energy_usage_history) if self.energy_usage_history else 0.0,
            'energy_efficiency_mode': self.energy_efficiency_mode
        }

