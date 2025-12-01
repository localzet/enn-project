"""
ENN для автономных агентов

Специализация для агентов, адаптирующихся к новым условиям:
- Онлайн обучение в реальном времени
- Адаптация к изменяющейся среде
- Управление ресурсами в полевых условиях
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Добавляем путь к корню проекта для импорта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.enn_core import EmergentNeuralNetwork
from core.energy_management import EnergyManager


class AutonomousAgentENN(EmergentNeuralNetwork):
    """
    ENN для автономных агентов
    
    Особенности:
    1. Быстрая адаптация к новым условиям
    2. Управление энергобюджетом в реальном времени
    3. Приоритизация важных паттернов
    4. Защита от катастрофического забывания
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 adaptation_rate: float = 0.1,
                 memory_budget: float = 50.0,
                 **kwargs):
        """
        Args:
            input_size: Размер входного вектора
            output_size: Размер выходного вектора
            adaptation_rate: Скорость адаптации к новым условиям
            memory_budget: Бюджет памяти для хранения важных паттернов
            **kwargs: Дополнительные параметры для базового ENN
        """
        super().__init__(input_size, output_size, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.memory_budget = memory_budget
        
        # Память важных паттернов
        self.important_patterns: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.pattern_importance: List[float] = []
        
        # История адаптации
        self.adaptation_history = []
        self.environment_changes = 0
    
    def learn_online(self,
                    observation: torch.Tensor,
                    action: torch.Tensor,
                    reward: float,
                    next_observation: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Онлайн обучение на одном наблюдении
        
        Args:
            observation: Текущее наблюдение среды
            action: Выполненное действие
            reward: Полученная награда
            next_observation: Следующее наблюдение (опционально)
        
        Returns:
            Метрики обучения
        """
        # Предсказываем действие на основе наблюдения
        predicted_action = self.forward(observation.unsqueeze(0))
        
        # Вычисляем ошибку
        target_action = action.unsqueeze(0) if len(action.shape) == 1 else action
        error = torch.nn.functional.mse_loss(predicted_action, target_action)
        
        # Адаптивная награда (учитывает изменение среды)
        adaptive_reward = reward * (1.0 + self.adaptation_rate)
        
        # Обучение
        metrics = self.learn(observation.unsqueeze(0), target_action, adaptive_reward)
        
        # Сохранение важных паттернов
        if reward > 0.5:  # Важный паттерн
            self._store_important_pattern(observation, action, reward)
        
        # Обнаружение изменения среды
        if self._detect_environment_change(observation, error.item()):
            self.environment_changes += 1
            self._adapt_to_new_environment()
        
        metrics.update({
            'environment_changes': self.environment_changes,
            'important_patterns': len(self.important_patterns)
        })
        
        return metrics
    
    def _store_important_pattern(self,
                                observation: torch.Tensor,
                                action: torch.Tensor,
                                importance: float):
        """Сохранение важного паттерна"""
        # Проверяем бюджет памяти
        if len(self.important_patterns) * self.memory_budget / 100 > self.memory_budget:
            # Удаляем наименее важный паттерн
            min_idx = np.argmin(self.pattern_importance)
            self.important_patterns.pop(min_idx)
            self.pattern_importance.pop(min_idx)
        
        self.important_patterns.append((observation.clone(), action.clone()))
        self.pattern_importance.append(importance)
    
    def _detect_environment_change(self,
                                  observation: torch.Tensor,
                                  error: float) -> bool:
        """
        Обнаружение изменения среды
        
        Признаки изменения:
        - Резкое увеличение ошибки
        - Необычные паттерны в наблюдениях
        """
        if len(self.adaptation_history) < 10:
            self.adaptation_history.append(error)
            return False
        
        recent_errors = self.adaptation_history[-10:]
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # Изменение среды: ошибка значительно выше среднего
        threshold = mean_error + 2 * std_error
        is_change = error > threshold
        
        self.adaptation_history.append(error)
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
        
        return is_change
    
    def _adapt_to_new_environment(self):
        """Адаптация к новой среде"""
        # Увеличиваем скорость обучения
        self.learning_rate *= 1.1
        
        # Переобучаемся на важных паттернах
        if self.important_patterns:
            for obs, act in self.important_patterns[:5]:  # Топ-5 паттернов
                self.learn(obs.unsqueeze(0), act.unsqueeze(0), reward=0.5)
    
    def replay_important_patterns(self, num_patterns: int = 5):
        """Воспроизведение важных паттернов для предотвращения забывания"""
        if not self.important_patterns:
            return
        
        # Выбираем наиболее важные паттерны
        top_indices = np.argsort(self.pattern_importance)[-num_patterns:]
        
        for idx in top_indices:
            obs, act = self.important_patterns[idx]
            self.learn(obs.unsqueeze(0), act.unsqueeze(0), reward=0.3)
    
    def get_adaptation_statistics(self) -> Dict[str, float]:
        """Получить статистику адаптации"""
        return {
            'environment_changes': self.environment_changes,
            'important_patterns': len(self.important_patterns),
            'adaptation_rate': self.adaptation_rate,
            'recent_error': np.mean(self.adaptation_history[-10:]) if self.adaptation_history else 0.0
        }

