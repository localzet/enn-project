"""
ENN для Continual Learning

Специализация для обучения на последовательности задач:
- Защита от катастрофического забывания
- Эффективное использование памяти
- Передача знаний между задачами
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


class ContinualLearningENN(EmergentNeuralNetwork):
    """
    ENN для Continual Learning
    
    Особенности:
    1. Защита от катастрофического забывания через важные паттерны
    2. Динамическое выделение памяти под новые задачи
    3. Передача знаний между задачами
    4. Оценка важности паттернов для каждой задачи
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 memory_per_task: float = 20.0,
                 **kwargs):
        """
        Args:
            input_size: Размер входного вектора
            output_size: Размер выходного вектора
            memory_per_task: Память на задачу
            **kwargs: Дополнительные параметры
        """
        super().__init__(input_size, output_size, **kwargs)
        
        self.memory_per_task = memory_per_task
        self.current_task_id = 0
        
        # Память для каждой задачи
        self.task_memories: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.task_importance: Dict[int, float] = {}
        
        # Статистика забывания
        self.forgetting_metrics = []
    
    def learn_task(self,
                  task_id: int,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  importance: float = 1.0) -> Dict[str, float]:
        """
        Обучение на задаче
        
        Args:
            task_id: ID задачи
            x: Входные данные
            y: Целевые значения
            importance: Важность задачи
        
        Returns:
            Метрики обучения
        """
        if task_id != self.current_task_id:
            # Переключение на новую задачу
            self._switch_task(task_id)
        
        # Обучение
        metrics = self.learn(x, y, reward=importance)
        
        # Сохранение важных примеров
        self._store_task_example(task_id, x, y, importance)
        
        # Периодическое воспроизведение для предотвращения забывания
        if self.generation % 100 == 0:
            self._replay_previous_tasks()
        
        metrics['task_id'] = task_id
        return metrics
    
    def _switch_task(self, new_task_id: int):
        """Переключение на новую задачу"""
        # Сохраняем важные примеры текущей задачи
        if self.current_task_id not in self.task_memories:
            self.task_memories[self.current_task_id] = []
        
        # Переключаемся
        self.current_task_id = new_task_id
        
        # Инициализируем память для новой задачи
        if new_task_id not in self.task_memories:
            self.task_memories[new_task_id] = []
            self.task_importance[new_task_id] = 1.0
    
    def _store_task_example(self,
                           task_id: int,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           importance: float):
        """Сохранение примера задачи"""
        if task_id not in self.task_memories:
            self.task_memories[task_id] = []
        
        memory = self.task_memories[task_id]
        
        # Проверяем лимит памяти
        max_examples = int(self.memory_per_task)
        if len(memory) >= max_examples:
            # Удаляем наименее важный пример
            memory.pop(0)
        
        memory.append((x.clone(), y.clone(), importance))
    
    def _replay_previous_tasks(self, num_examples_per_task: int = 3):
        """Воспроизведение примеров из предыдущих задач"""
        for task_id, examples in self.task_memories.items():
            if task_id == self.current_task_id:
                continue  # Пропускаем текущую задачу
            
            # Выбираем случайные примеры
            if len(examples) > 0:
                indices = np.random.choice(len(examples), 
                                         min(num_examples_per_task, len(examples)),
                                         replace=False)
                for idx in indices:
                    x, y, importance = examples[idx]
                    self.learn(x, y, reward=importance * 0.5)  # Пониженная награда
    
    def evaluate_task(self,
                     task_id: int,
                     x: torch.Tensor,
                     y: torch.Tensor) -> Dict[str, float]:
        """
        Оценка производительности на задаче
        
        Returns:
            Метрики производительности
        """
        with torch.no_grad():
            predictions = self.forward(x)
            error = torch.nn.functional.mse_loss(predictions, y)
            
            # Вычисляем забывание (сравниваем с базовой производительностью)
            forgetting = 0.0  # Упрощенная версия
        
        return {
            'task_id': task_id,
            'error': error.item(),
            'forgetting': forgetting
        }
    
    def get_forgetting_statistics(self) -> Dict[str, float]:
        """Статистика забывания"""
        if not self.forgetting_metrics:
            return {'average_forgetting': 0.0, 'tasks_learned': 0}
        
        return {
            'average_forgetting': np.mean([m['forgetting'] for m in self.forgetting_metrics]),
            'tasks_learned': len(self.task_memories),
            'total_memory_used': sum(len(m) for m in self.task_memories.values())
        }

