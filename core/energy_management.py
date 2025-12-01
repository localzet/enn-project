"""
Управление энергобюджетом сети

Формализация:
- E(t+1) = E(t) - consumption(t) + generation(t) - creation_cost(t)
- consumption(t) = Σ |activation_i| * energy_per_activation
- generation(t) = base_generation + reward_bonus
"""

from typing import List


class EnergyManager:
    """
    Управление энергобюджетом
    
    Формализация:
    - E(t+1) = E(t) - consumption(t) + generation(t) - creation_cost(t)
    - consumption(t) = Σ |activation_i| * energy_per_activation
    - generation(t) = base_generation + reward_bonus
    """
    
    def __init__(self,
                 initial_budget: float = 100.0,
                 energy_per_activation: float = 0.01,
                 base_generation: float = 0.1,
                 reward_multiplier: float = 0.5):
        self.budget = initial_budget
        self.energy_per_activation = energy_per_activation
        self.base_generation = base_generation
        self.reward_multiplier = reward_multiplier
        self.history = []
    
    def consume(self, activations: List[float]) -> float:
        """Потребление энергии на основе активаций"""
        consumption = sum(abs(a) for a in activations) * self.energy_per_activation
        self.budget = max(0, self.budget - consumption)
        self.history.append(('consume', consumption))
        return consumption
    
    def generate(self, reward: float = 0.0) -> float:
        """Генерация энергии"""
        generation = self.base_generation + reward * self.reward_multiplier
        self.budget += generation
        self.history.append(('generate', generation))
        return generation
    
    def can_afford(self, cost: float) -> bool:
        """Проверить, достаточно ли энергии"""
        return self.budget >= cost
    
    def spend(self, cost: float) -> bool:
        """Потратить энергию"""
        if self.can_afford(cost):
            self.budget -= cost
            self.history.append(('spend', cost))
            return True
        return False
    
    def get_statistics(self) -> dict:
        """Получить статистику энергопотребления"""
        if not self.history:
            return {'current_budget': self.budget}
        
        total_consumed = sum(v for t, v in self.history if t == 'consume')
        total_generated = sum(v for t, v in self.history if t == 'generate')
        total_spent = sum(v for t, v in self.history if t == 'spend')
        
        return {
            'current_budget': self.budget,
            'total_consumed': total_consumed,
            'total_generated': total_generated,
            'total_spent': total_spent,
            'net_energy': total_generated - total_consumed - total_spent
        }

