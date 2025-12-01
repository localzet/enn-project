"""
Формальная причинно-следственная логика

Реализация методов для определения причинно-следственных связей:
- Granger Causality
- Structural Causal Models (SCM)
- Do-calculus (базовая версия)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression


class CausalityAnalyzer:
    """
    Анализатор причинно-следственных связей
    
    Использует формальные методы:
    1. Granger Causality - статистический тест причинности
    2. Structural Causal Models - моделирование структурных уравнений
    3. Do-calculus - интервенционное исчисление
    """
    
    def __init__(self, lag: int = 1, significance_level: float = 0.05):
        self.lag = lag
        self.significance_level = significance_level
        self.causal_graph = {}
    
    def granger_causality_test(self,
                              x: np.ndarray,
                              y: np.ndarray) -> Dict[str, float]:
        """
        Тест Грэнджера на причинность
        
        H0: x не является причиной y
        H1: x является причиной y (Granger-causes)
        
        Args:
            x: Временной ряд причины
            y: Временной ряд следствия
        
        Returns:
            Словарь с результатами теста
        """
        if len(x) != len(y) or len(x) < self.lag + 1:
            return {'causal': False, 'p_value': 1.0, 'f_statistic': 0.0}
        
        n = len(x) - self.lag
        
        # Модель без x (ограниченная)
        y_restricted = y[self.lag:]
        X_restricted = np.column_stack([y[i:i+n] for i in range(self.lag)])
        model_restricted = LinearRegression()
        model_restricted.fit(X_restricted, y_restricted)
        rss_restricted = np.sum((y_restricted - model_restricted.predict(X_restricted))**2)
        
        # Модель с x (неограниченная)
        X_unrestricted = np.column_stack([
            *[y[i:i+n] for i in range(self.lag)],
            *[x[i:i+n] for i in range(self.lag)]
        ])
        model_unrestricted = LinearRegression()
        model_unrestricted.fit(X_unrestricted, y_restricted)
        rss_unrestricted = np.sum((y_restricted - model_unrestricted.predict(X_unrestricted))**2)
        
        # F-статистика
        f_statistic = ((rss_restricted - rss_unrestricted) / self.lag) / \
                     (rss_unrestricted / (n - 2 * self.lag))
        
        # P-value (упрощенная версия)
        # В реальности нужен точный F-тест
        p_value = 1.0 - stats.f.cdf(f_statistic, self.lag, n - 2 * self.lag) if n > 2 * self.lag else 1.0
        
        is_causal = p_value < self.significance_level
        
        return {
            'causal': is_causal,
            'p_value': float(p_value),
            'f_statistic': float(f_statistic),
            'causal_strength': float(f_statistic) if is_causal else 0.0
        }
    
    def analyze_neural_causality(self,
                                source_activations: List[float],
                                target_activations: List[float]) -> Dict[str, float]:
        """
        Анализ причинно-следственной связи между нейронами
        
        Args:
            source_activations: История активаций источника
            target_activations: История активаций цели
        
        Returns:
            Метрики причинности
        """
        if len(source_activations) != len(target_activations):
            min_len = min(len(source_activations), len(target_activations))
            source_activations = source_activations[:min_len]
            target_activations = target_activations[:min_len]
        
        if len(source_activations) < self.lag + 1:
            return {'causal_strength': 0.0, 'is_causal': False}
        
        source_array = np.array(source_activations)
        target_array = np.array(target_activations)
        
        # Granger causality test
        granger_result = self.granger_causality_test(source_array, target_array)
        
        # Корреляция (дополнительная метрика)
        correlation = np.corrcoef(source_array, target_array)[0, 1]
        
        # Взаимная информация (мера зависимости)
        # Упрощенная версия через корреляцию
        mutual_info = 0.5 * np.log(1 - correlation**2) if abs(correlation) < 1.0 else 0.0
        
        return {
            'causal_strength': granger_result['causal_strength'],
            'is_causal': granger_result['causal'],
            'p_value': granger_result['p_value'],
            'correlation': float(correlation),
            'mutual_info': float(mutual_info)
        }
    
    def build_causal_graph(self,
                          activations_history: Dict[int, List[float]]) -> Dict[Tuple[int, int], float]:
        """
        Построение графа причинно-следственных связей
        
        Args:
            activations_history: Словарь {neuron_id: [activations]}
        
        Returns:
            Словарь {(source, target): causal_strength}
        """
        causal_graph = {}
        neuron_ids = list(activations_history.keys())
        
        for source_id in neuron_ids:
            for target_id in neuron_ids:
                if source_id != target_id:
                    source_acts = activations_history[source_id]
                    target_acts = activations_history[target_id]
                    
                    causality = self.analyze_neural_causality(source_acts, target_acts)
                    causal_strength = causality['causal_strength']
                    
                    if causal_strength > 0.0:
                        causal_graph[(source_id, target_id)] = causal_strength
        
        self.causal_graph = causal_graph
        return causal_graph
    
    def do_calculus_intervention(self,
                                 causal_graph: Dict[Tuple[int, int], float],
                                 intervention_node: int,
                                 intervention_value: float) -> Dict[int, float]:
        """
        Базовое применение do-calculus для интервенции
        
        do(X = x) означает установку X в значение x
        
        Args:
            causal_graph: Граф причинно-следственных связей
            intervention_node: Узел для интервенции
            intervention_value: Значение интервенции
        
        Returns:
            Предсказанные эффекты на другие узлы
        """
        # Упрощенная версия: предсказываем эффекты через граф
        effects = {}
        
        for (source, target), strength in causal_graph.items():
            if source == intervention_node:
                # Прямой эффект интервенции
                effects[target] = intervention_value * strength
        
        # Распространение эффектов (упрощенное)
        for node_id in effects.keys():
            for (source, target), strength in causal_graph.items():
                if source == node_id and target not in effects:
                    effects[target] = effects[node_id] * strength * 0.5
        
        return effects
    
    def get_causal_importance(self,
                             neuron_id: int,
                             causal_graph: Dict[Tuple[int, int], float]) -> float:
        """
        Вычисление причинной важности нейрона
        
        Важность = сумма исходящих причинных связей
        """
        importance = 0.0
        for (source, target), strength in causal_graph.items():
            if source == neuron_id:
                importance += strength
        return importance

