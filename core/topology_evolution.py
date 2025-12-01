"""
Формализованная эволюция топологии сети

Математическая формализация правил создания и удаления нейронов
"""

from dataclasses import dataclass
from typing import Dict, List
from .enn_core import Neuron


@dataclass
class TopologyEvolution:
    """
    Формализованная эволюция топологии
    
    Правила создания нейрона:
    - Условие: E(t) > E_threshold AND error(t) > error_threshold
    - Стоимость: C_new = α * (1 + complexity_penalty)
    - Где: E(t) - текущий энергобюджет, error(t) - текущая ошибка
    
    Правила удаления нейрона:
    - Условие: importance < importance_threshold AND age > age_min
    - Возврат энергии: E_return = β * energy_invested
    """
    
    def __init__(self, 
                 energy_threshold: float = 10.0,
                 error_threshold: float = 0.2,
                 importance_threshold: float = 0.1,
                 age_min: int = 100,
                 creation_cost: float = 5.0,
                 complexity_penalty: float = 0.1):
        self.energy_threshold = energy_threshold
        self.error_threshold = error_threshold
        self.importance_threshold = importance_threshold
        self.age_min = age_min
        self.creation_cost = creation_cost
        self.complexity_penalty = complexity_penalty
    
    def should_create_neuron(self, 
                             current_error: float,
                             energy_budget: float,
                             num_neurons: int) -> bool:
        """
        Формальное правило создания нейрона
        
        Returns:
            True если выполняются все условия создания
        """
        if energy_budget < self.energy_threshold:
            return False
        
        if current_error < self.error_threshold:
            return False
        
        # Штраф за сложность
        complexity = num_neurons * self.complexity_penalty
        effective_cost = self.creation_cost * (1 + complexity)
        
        return energy_budget >= effective_cost
    
    def calculate_creation_cost(self, num_neurons: int) -> float:
        """Вычислить стоимость создания нового нейрона"""
        complexity = num_neurons * self.complexity_penalty
        return self.creation_cost * (1 + complexity)
    
    def should_remove_neuron(self,
                            neuron: Neuron,
                            min_importance: float) -> bool:
        """
        Формальное правило удаления нейрона
        
        Returns:
            True если нейрон можно удалить
        """
        return (neuron.importance < self.importance_threshold and
                neuron.age > self.age_min)
    
    def calculate_energy_return(self, neuron: Neuron) -> float:
        """Вычислить возврат энергии при удалении нейрона"""
        return self.creation_cost * 0.5  # Возвращаем 50%

