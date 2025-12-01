"""
Теоретический анализ стабильности ENN

Формальные условия сходимости и стабильности динамики сети
"""

import numpy as np
from typing import Dict, List, Tuple
import torch


class StabilityAnalyzer:
    """
    Анализ стабильности динамики сети
    
    Теоретические результаты:
    1. Условие сходимости: |λ_max(W)| < 1, где W - матрица весов
    2. Стабильность энергобюджета: E(t) > 0 для всех t
    3. Ограниченность роста: |neurons(t)| < max_neurons
    """
    
    def __init__(self):
        self.stability_history = []
    
    def analyze_convergence(self, weights: Dict[Tuple[int, int], torch.Tensor]) -> Dict[str, float]:
        """
        Анализ сходимости на основе спектрального радиуса матрицы весов
        
        Условие сходимости: ρ(W) < 1, где ρ - спектральный радиус
        
        Returns:
            Словарь с метриками сходимости
        """
        # Строим матрицу весов
        neuron_ids = set()
        for (src, tgt) in weights.keys():
            neuron_ids.add(src)
            neuron_ids.add(tgt)
        
        neuron_ids = sorted(neuron_ids)
        n = len(neuron_ids)
        
        if n == 0:
            return {'spectral_radius': 0.0, 'is_stable': True}
        
        W = np.zeros((n, n))
        id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}
        
        for (src, tgt), weight in weights.items():
            if src in id_to_idx and tgt in id_to_idx:
                src_idx = id_to_idx[src]
                tgt_idx = id_to_idx[tgt]
                W[src_idx, tgt_idx] = weight.item() if isinstance(weight, torch.Tensor) else weight
        
        # Вычисляем собственные значения
        eigenvalues = np.linalg.eigvals(W)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        # Условие стабильности
        is_stable = spectral_radius < 1.0
        
        result = {
            'spectral_radius': float(spectral_radius),
            'is_stable': is_stable,
            'max_eigenvalue': float(np.max(np.real(eigenvalues))),
            'min_eigenvalue': float(np.min(np.real(eigenvalues)))
        }
        
        self.stability_history.append(result)
        return result
    
    def analyze_energy_stability(self, energy_history: List[float]) -> Dict[str, float]:
        """
        Анализ стабильности энергобюджета
        
        Условие: E(t) > 0 для всех t
        
        Returns:
            Метрики стабильности энергии
        """
        if not energy_history:
            return {'is_stable': True, 'min_energy': 0.0}
        
        energy_array = np.array(energy_history)
        min_energy = float(np.min(energy_array))
        mean_energy = float(np.mean(energy_array))
        std_energy = float(np.std(energy_array))
        
        # Стабильность: энергия не уходит в отрицательную область
        is_stable = min_energy >= 0.0
        
        # Проверка на взрыв (быстрый рост)
        if len(energy_history) > 1:
            growth_rate = np.diff(energy_array)
            max_growth = float(np.max(np.abs(growth_rate)))
        else:
            max_growth = 0.0
        
        return {
            'is_stable': is_stable,
            'min_energy': min_energy,
            'mean_energy': mean_energy,
            'std_energy': std_energy,
            'max_growth_rate': max_growth
        }
    
    def analyze_topology_stability(self, 
                                  neuron_counts: List[int],
                                  max_neurons: int) -> Dict[str, float]:
        """
        Анализ стабильности топологии
        
        Условие: |neurons(t)| < max_neurons для всех t
        
        Returns:
            Метрики стабильности топологии
        """
        if not neuron_counts:
            return {'is_stable': True, 'max_count': 0}
        
        counts_array = np.array(neuron_counts)
        max_count = int(np.max(counts_array))
        mean_count = float(np.mean(counts_array))
        
        # Стабильность: не превышаем максимум
        is_stable = max_count < max_neurons
        
        # Скорость роста
        if len(neuron_counts) > 1:
            growth_rate = np.diff(counts_array)
            mean_growth = float(np.mean(growth_rate))
        else:
            mean_growth = 0.0
        
        return {
            'is_stable': is_stable,
            'max_count': max_count,
            'mean_count': mean_count,
            'mean_growth_rate': mean_growth
        }
    
    def get_stability_report(self,
                           weights: Dict[Tuple[int, int], torch.Tensor],
                           energy_history: List[float],
                           neuron_counts: List[int],
                           max_neurons: int) -> Dict[str, any]:
        """
        Полный отчет о стабильности
        
        Returns:
            Комплексный отчет со всеми метриками
        """
        convergence = self.analyze_convergence(weights)
        energy = self.analyze_energy_stability(energy_history)
        topology = self.analyze_topology_stability(neuron_counts, max_neurons)
        
        overall_stable = (convergence['is_stable'] and 
                          energy['is_stable'] and 
                          topology['is_stable'])
        
        return {
            'overall_stable': overall_stable,
            'convergence': convergence,
            'energy': energy,
            'topology': topology
        }

