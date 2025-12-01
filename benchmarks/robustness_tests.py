"""
Тесты робастности ENN

Проверяем:
- Устойчивость к шуму
- Катастрофическое забывание
- Адаптация к изменению распределения
- Устойчивость к выбросам
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import EmergentNeuralNetwork


class RobustnessTester:
    """Тестер робастности"""
    
    def __init__(self):
        self.results = {}
    
    def test_noise_robustness(self,
                              model: EmergentNeuralNetwork,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5, 1.0]) -> Dict[str, List[float]]:
        """
        Тест устойчивости к шуму
        
        Args:
            model: Обученная модель
            x: Входные данные
            y: Целевые значения
            noise_levels: Уровни шума
        
        Returns:
            Результаты для каждого уровня шума
        """
        results = {
            'noise_levels': noise_levels,
            'errors': [],
            'energy_consumption': []
        }
        
        for noise_level in noise_levels:
            # Добавляем шум
            x_noisy = x + torch.randn_like(x) * noise_level
            
            # Предсказание
            with torch.no_grad():
                predictions = model.forward(x_noisy)
                error = torch.nn.functional.mse_loss(predictions, y).item()
            
            results['errors'].append(error)
            
            # Энергопотребление
            stats = model.get_statistics()
            results['energy_consumption'].append(stats.get('total_energy_consumed', 0.0))
        
        self.results['noise_robustness'] = results
        return results
    
    def test_catastrophic_forgetting(self,
                                    model: EmergentNeuralNetwork,
                                    task1_data: Tuple[torch.Tensor, torch.Tensor],
                                    task2_data: Tuple[torch.Tensor, torch.Tensor],
                                    num_task2_epochs: int = 100) -> Dict[str, float]:
        """
        Тест катастрофического забывания
        
        Args:
            model: Модель, обученная на задаче 1
            task1_data: Данные задачи 1
            task2_data: Данные задачи 2
            num_task2_epochs: Количество эпох обучения на задаче 2
        
        Returns:
            Метрики забывания
        """
        x1, y1 = task1_data
        x2, y2 = task2_data
        
        # Оценка производительности на задаче 1 до обучения на задаче 2
        with torch.no_grad():
            pred1_before = model.forward(x1)
            error1_before = torch.nn.functional.mse_loss(pred1_before, y1).item()
        
        # Обучение на задаче 2
        for epoch in range(num_task2_epochs):
            for i in range(len(x2)):
                model.learn(x2[i:i+1], y2[i:i+1], reward=1.0)
        
        # Оценка производительности на задаче 1 после обучения на задаче 2
        with torch.no_grad():
            pred1_after = model.forward(x1)
            error1_after = torch.nn.functional.mse_loss(pred1_after, y1).item()
            
            pred2_after = model.forward(x2)
            error2_after = torch.nn.functional.mse_loss(pred2_after, y2).item()
        
        # Метрика забывания
        forgetting = error1_after - error1_before
        retention = 1.0 - (forgetting / (error1_before + 1e-6))
        
        results = {
            'error_task1_before': error1_before,
            'error_task1_after': error1_after,
            'error_task2_after': error2_after,
            'forgetting': forgetting,
            'retention': retention
        }
        
        self.results['catastrophic_forgetting'] = results
        return results
    
    def test_distribution_shift(self,
                               model: EmergentNeuralNetwork,
                               train_data: Tuple[torch.Tensor, torch.Tensor],
                               test_data_shifted: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Тест адаптации к изменению распределения
        
        Args:
            model: Модель, обученная на train_data
            train_data: Обучающие данные
            test_data_shifted: Тестовые данные со сдвигом распределения
        
        Returns:
            Метрики адаптации
        """
        x_train, y_train = train_data
        x_test, y_test = test_data_shifted
        
        # Оценка до адаптации
        with torch.no_grad():
            pred_before = model.forward(x_test)
            error_before = torch.nn.functional.mse_loss(pred_before, y_test).item()
        
        # Адаптация на новых данных
        adaptation_errors = []
        for i in range(min(50, len(x_test))):
            metrics = model.learn(x_test[i:i+1], y_test[i:i+1], reward=1.0)
            adaptation_errors.append(metrics['error'])
        
        # Оценка после адаптации
        with torch.no_grad():
            pred_after = model.forward(x_test)
            error_after = torch.nn.functional.mse_loss(pred_after, y_test).item()
        
        adaptation_improvement = error_before - error_after
        
        results = {
            'error_before_adaptation': error_before,
            'error_after_adaptation': error_after,
            'adaptation_improvement': adaptation_improvement,
            'adaptation_rate': np.mean(adaptation_errors[-10:]) if adaptation_errors else 0.0
        }
        
        self.results['distribution_shift'] = results
        return results
    
    def test_outlier_robustness(self,
                               model: EmergentNeuralNetwork,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               outlier_ratio: float = 0.1) -> Dict[str, float]:
        """
        Тест устойчивости к выбросам
        
        Args:
            model: Модель
            x: Входные данные
            y: Целевые значения
            outlier_ratio: Доля выбросов
        
        Returns:
            Метрики устойчивости
        """
        n_outliers = int(len(x) * outlier_ratio)
        
        # Создаем выбросы
        outlier_indices = np.random.choice(len(x), n_outliers, replace=False)
        x_with_outliers = x.clone()
        y_with_outliers = y.clone()
        
        for idx in outlier_indices:
            x_with_outliers[idx] = torch.randn_like(x[idx]) * 5.0  # Большой шум
            y_with_outliers[idx] = torch.randn_like(y[idx]) * 5.0
        
        # Обучение с выбросами
        errors_with_outliers = []
        for i in range(len(x_with_outliers)):
            metrics = model.learn(x_with_outliers[i:i+1], y_with_outliers[i:i+1], reward=1.0)
            errors_with_outliers.append(metrics['error'])
        
        # Оценка на чистых данных
        with torch.no_grad():
            pred_clean = model.forward(x)
            error_clean = torch.nn.functional.mse_loss(pred_clean, y).item()
        
        results = {
            'error_with_outliers': np.mean(errors_with_outliers),
            'error_on_clean_data': error_clean,
            'outlier_ratio': outlier_ratio,
            'robustness_score': 1.0 / (error_clean + 1e-6)
        }
        
        self.results['outlier_robustness'] = results
        return results
    
    def generate_report(self) -> str:
        """Генерация отчета о робастности"""
        report = "\n" + "="*70 + "\n"
        report += "ОТЧЕТ О РОБАСТНОСТИ ENN\n"
        report += "="*70 + "\n"
        
        for test_name, results in self.results.items():
            report += f"\n{test_name.upper()}:\n"
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    report += f"  {key}: {value:.6f}\n"
                elif isinstance(value, list):
                    report += f"  {key}: {value}\n"
        
        report += "\n" + "="*70 + "\n"
        return report

