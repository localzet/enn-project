"""
ENN для DevOps и Infrastructure

Применение ENN для:
- Мониторинг и адаптивная оптимизация инфраструктуры
- Предсказание сбоев и автоматическое масштабирование
- Адаптивное управление ресурсами
- Continual learning на метриках инфраструктуры
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.enn_core import EmergentNeuralNetwork
from core.stability_analysis import StabilityAnalyzer


class InfrastructureMonitorENN(EmergentNeuralNetwork):
    """
    ENN для мониторинга инфраструктуры
    
    Использование:
    - Предсказание нагрузки на основе метрик
    - Адаптивное масштабирование
    - Обнаружение аномалий
    - Оптимизация ресурсов
    """
    
    def __init__(self,
                 metric_count: int = 10,  # CPU, Memory, Network, Disk, etc.
                 prediction_horizon: int = 1,  # Шагов вперед для предсказания
                 **kwargs):
        """
        Args:
            metric_count: Количество метрик для мониторинга
            prediction_horizon: Горизонт предсказания
            **kwargs: Параметры базового ENN
        """
        super().__init__(
            input_size=metric_count,
            output_size=metric_count * prediction_horizon,
            **kwargs
        )
        
        self.metric_count = metric_count
        self.prediction_horizon = prediction_horizon
        
        # История метрик
        self.metrics_history: List[Dict[str, float]] = []
        self.predictions_history: List[Dict[str, float]] = []
        
        # Пороги для алертов
        self.alert_thresholds: Dict[str, float] = {}
        
        # Статистика
        self.anomalies_detected = 0
        self.scaling_events = 0
    
    def process_metrics(self,
                       metrics: Dict[str, float],
                       timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Обработка метрик инфраструктуры
        
        Args:
            metrics: Словарь метрик {metric_name: value}
            timestamp: Временная метка
        
        Returns:
            Результаты анализа и предсказания
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Преобразуем метрики в вектор
        metric_vector = self._metrics_to_vector(metrics)
        
        # Предсказание будущих значений
        with torch.no_grad():
            prediction_vector = self.forward(metric_vector.unsqueeze(0))
            predictions = self._vector_to_metrics(prediction_vector[0])
        
        # Обнаружение аномалий
        anomaly_score = self._detect_anomaly(metrics, predictions)
        
        # Обучение на новых данных
        # Используем предсказания как цели (самообучение)
        target_vector = metric_vector.repeat(self.prediction_horizon)
        reward = 1.0 if anomaly_score < 0.3 else 0.5  # Меньшая награда при аномалиях
        
        self.learn(metric_vector.unsqueeze(0), target_vector.unsqueeze(0), reward)
        
        # Сохранение истории
        self.metrics_history.append({
            'timestamp': timestamp.isoformat(),
            'metrics': metrics,
            'predictions': predictions,
            'anomaly_score': anomaly_score
        })
        
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        if anomaly_score > 0.5:
            self.anomalies_detected += 1
        
        return {
            'predictions': predictions,
            'anomaly_score': anomaly_score,
            'is_anomaly': anomaly_score > 0.5,
            'recommendations': self._generate_recommendations(metrics, predictions, anomaly_score)
        }
    
    def _metrics_to_vector(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Преобразование метрик в вектор"""
        # Нормализация метрик
        vector = []
        for i in range(self.metric_count):
            metric_name = f"metric_{i}"
            value = metrics.get(metric_name, 0.0)
            # Нормализация к [0, 1]
            normalized = max(0.0, min(1.0, value / 100.0))
            vector.append(normalized)
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def _vector_to_metrics(self, vector: torch.Tensor) -> Dict[str, float]:
        """Преобразование вектора обратно в метрики"""
        predictions = {}
        for i in range(self.metric_count):
            metric_name = f"metric_{i}"
            # Денормализация
            value = vector[i].item() * 100.0
            predictions[metric_name] = max(0.0, min(100.0, value))
        
        return predictions
    
    def _detect_anomaly(self,
                       current_metrics: Dict[str, float],
                       predictions: Dict[str, float]) -> float:
        """Обнаружение аномалий"""
        # Вычисляем отклонение от предсказаний
        deviations = []
        for key in current_metrics.keys():
            if key in predictions:
                deviation = abs(current_metrics[key] - predictions[key]) / 100.0
                deviations.append(deviation)
        
        anomaly_score = np.mean(deviations) if deviations else 0.0
        return anomaly_score
    
    def _generate_recommendations(self,
                                 metrics: Dict[str, float],
                                 predictions: Dict[str, float],
                                 anomaly_score: float) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Анализ метрик
        cpu_usage = metrics.get('metric_0', 0.0)  # Предполагаем CPU
        memory_usage = metrics.get('metric_1', 0.0)  # Предполагаем Memory
        
        if cpu_usage > 80:
            recommendations.append("Высокая загрузка CPU - рассмотрите масштабирование")
        
        if memory_usage > 85:
            recommendations.append("Высокое использование памяти - проверьте утечки")
        
        if anomaly_score > 0.5:
            recommendations.append("Обнаружена аномалия - требуется расследование")
        
        # Предсказания
        predicted_cpu = predictions.get('metric_0', 0.0)
        if predicted_cpu > 90:
            recommendations.append(f"Предсказана высокая нагрузка CPU ({predicted_cpu:.1f}%) - подготовьте масштабирование")
        
        return recommendations
    
    def recommend_scaling(self,
                         current_metrics: Dict[str, float],
                         target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Рекомендации по масштабированию
        
        Args:
            current_metrics: Текущие метрики
            target_metrics: Целевые метрики
        
        Returns:
            Рекомендации по масштабированию
        """
        # Вычисляем необходимые изменения
        scaling_factor = {}
        for key in current_metrics.keys():
            if key in target_metrics:
                current = current_metrics[key]
                target = target_metrics[key]
                if current > 0:
                    scaling_factor[key] = target / current
        
        # Рекомендации
        avg_scaling = np.mean(list(scaling_factor.values())) if scaling_factor else 1.0
        
        recommendation = {
            'scale_up': avg_scaling > 1.2,
            'scale_down': avg_scaling < 0.8,
            'scaling_factor': avg_scaling,
            'recommended_instances': max(1, int(avg_scaling * 3))  # Предполагаем 3 текущих инстанса
        }
        
        if recommendation['scale_up'] or recommendation['scale_down']:
            self.scaling_events += 1
        
        return recommendation
    
    def get_infrastructure_statistics(self) -> Dict[str, Any]:
        """Статистика инфраструктуры"""
        stats = self.get_statistics()
        stats.update({
            'anomalies_detected': self.anomalies_detected,
            'scaling_events': self.scaling_events,
            'metrics_processed': len(self.metrics_history)
        })
        return stats


class AdaptiveLoadBalancerENN(EmergentNeuralNetwork):
    """
    ENN для адаптивной балансировки нагрузки
    
    Использование:
    - Динамическое распределение запросов
    - Адаптация к изменяющимся паттернам нагрузки
    - Оптимизация задержек
    """
    
    def __init__(self,
                 server_count: int = 5,
                 **kwargs):
        """
        Args:
            server_count: Количество серверов
            **kwargs: Параметры базового ENN
        """
        # Вход: метрики серверов (CPU, Memory, Latency для каждого)
        input_size = server_count * 3
        # Выход: веса распределения для каждого сервера
        output_size = server_count
        
        super().__init__(input_size=input_size, output_size=output_size, **kwargs)
        
        self.server_count = server_count
        
        # История распределения
        self.distribution_history: List[Dict[int, float]] = []
        self.latency_history: List[float] = []
    
    def balance_load(self,
                    server_metrics: List[Dict[str, float]]) -> Dict[int, float]:
        """
        Балансировка нагрузки
        
        Args:
            server_metrics: Метрики для каждого сервера
        
        Returns:
            Веса распределения для каждого сервера
        """
        # Преобразуем метрики в вектор
        input_vector = self._server_metrics_to_vector(server_metrics)
        
        # Предсказываем оптимальное распределение
        with torch.no_grad():
            output_vector = self.forward(input_vector.unsqueeze(0))
            weights = torch.softmax(output_vector[0], dim=0)  # Нормализуем в вероятности
        
        # Преобразуем в словарь
        distribution = {
            i: weights[i].item() for i in range(self.server_count)
        }
        
        # Обучение на основе результата (упрощенно)
        # В реальности нужно измерять фактическую задержку
        avg_latency = np.mean([m.get('latency', 0.0) for m in server_metrics])
        reward = 1.0 / (avg_latency + 1.0)  # Больше награда за меньшую задержку
        
        # Целевое распределение (равномерное, но можно улучшить)
        target_distribution = torch.ones(self.server_count) / self.server_count
        
        self.learn(input_vector.unsqueeze(0), target_distribution.unsqueeze(0), reward)
        
        self.distribution_history.append(distribution)
        self.latency_history.append(avg_latency)
        
        return distribution
    
    def _server_metrics_to_vector(self, server_metrics: List[Dict[str, float]]) -> torch.Tensor:
        """Преобразование метрик серверов в вектор"""
        vector = []
        for metrics in server_metrics:
            cpu = metrics.get('cpu', 0.0) / 100.0
            memory = metrics.get('memory', 0.0) / 100.0
            latency = min(metrics.get('latency', 0.0) / 1000.0, 1.0)  # Нормализуем задержку
            vector.extend([cpu, memory, latency])
        
        # Дополняем до нужного размера
        while len(vector) < self.input_size:
            vector.append(0.0)
        
        return torch.tensor(vector[:self.input_size], dtype=torch.float32)


class CI_CDOptimizerENN(EmergentNeuralNetwork):
    """
    ENN для оптимизации CI/CD пайплайнов
    
    Использование:
    - Предсказание времени сборки
    - Оптимизация порядка тестов
    - Адаптивное кэширование
    """
    
    def __init__(self, **kwargs):
        """
        Вход: метрики коммита, изменений, истории
        Выход: предсказание времени сборки, приоритет тестов
        """
        super().__init__(input_size=15, output_size=5, **kwargs)
        
        self.build_history: List[Dict[str, Any]] = []
        self.optimization_savings = 0.0
    
    def predict_build_time(self,
                          commit_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Предсказание времени сборки
        
        Args:
            commit_metrics: Метрики коммита (измененные файлы, размер, сложность)
        
        Returns:
            Предсказания времени сборки и рекомендации
        """
        # Преобразуем метрики в вектор
        input_vector = self._commit_metrics_to_vector(commit_metrics)
        
        # Предсказание
        with torch.no_grad():
            output = self.forward(input_vector.unsqueeze(0))
        
        predictions = {
            'estimated_build_time': output[0, 0].item() * 3600,  # В секундах
            'test_priority': output[0, 1:].tolist(),
            'cache_hit_probability': output[0, 2].item()
        }
        
        return predictions
    
    def optimize_pipeline(self,
                         commit_metrics: Dict[str, Any],
                         actual_build_time: float) -> Dict[str, Any]:
        """
        Оптимизация пайплайна
        
        Args:
            commit_metrics: Метрики коммита
            actual_build_time: Фактическое время сборки
        
        Returns:
            Рекомендации по оптимизации
        """
        # Предсказание
        predictions = self.predict_build_time(commit_metrics)
        
        # Обучение на фактических данных
        input_vector = self._commit_metrics_to_vector(commit_metrics)
        target_vector = torch.tensor([
            actual_build_time / 3600.0,  # Нормализуем
            0.5, 0.5, 0.5, 0.5  # Приоритеты тестов (упрощенно)
        ])
        
        reward = 1.0 if abs(predictions['estimated_build_time'] - actual_build_time) < 60 else 0.5
        self.learn(input_vector.unsqueeze(0), target_vector.unsqueeze(0), reward)
        
        # Рекомендации
        recommendations = []
        
        if predictions['estimated_build_time'] > 600:  # Больше 10 минут
            recommendations.append("Рассмотрите параллелизацию тестов")
        
        if predictions['cache_hit_probability'] < 0.3:
            recommendations.append("Низкая вероятность кэш-попаданий - оптимизируйте зависимости")
        
        return {
            'predictions': predictions,
            'recommendations': recommendations,
            'optimization_potential': max(0, predictions['estimated_build_time'] - actual_build_time)
        }
    
    def _commit_metrics_to_vector(self, metrics: Dict[str, Any]) -> torch.Tensor:
        """Преобразование метрик коммита в вектор"""
        vector = [
            metrics.get('files_changed', 0) / 100.0,
            metrics.get('lines_added', 0) / 1000.0,
            metrics.get('lines_deleted', 0) / 1000.0,
            metrics.get('complexity', 0) / 100.0,
            metrics.get('test_files_changed', 0) / 50.0,
            # ... другие метрики
        ]
        
        # Дополняем до нужного размера
        while len(vector) < self.input_size:
            vector.append(0.0)
        
        return torch.tensor(vector[:self.input_size], dtype=torch.float32)


class SecurityAnomalyDetectorENN(EmergentNeuralNetwork):
    """
    ENN для обнаружения аномалий безопасности
    
    Использование:
    - Обнаружение подозрительной активности
    - Адаптация к новым типам атак
    - Причинно-следственный анализ инцидентов
    """
    
    def __init__(self, **kwargs):
        """
        Вход: метрики безопасности (логи, сетевой трафик, доступы)
        Выход: вероятность атаки, тип угрозы
        """
        super().__init__(input_size=20, output_size=3, **kwargs)
        
        self.threats_detected = 0
        self.false_positives = 0
        
        # Используем каузальность для анализа
        from core.causality import CausalityAnalyzer
        self.causality_analyzer = CausalityAnalyzer()
    
    def analyze_security_events(self,
                               security_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Анализ событий безопасности
        
        Args:
            security_metrics: Метрики безопасности
        
        Returns:
            Результаты анализа
        """
        # Преобразуем в вектор
        input_vector = self._security_metrics_to_vector(security_metrics)
        
        # Предсказание
        with torch.no_grad():
            output = self.forward(input_vector.unsqueeze(0))
        
        threat_probability = torch.sigmoid(output[0, 0]).item()
        threat_type = torch.softmax(output[0, 1:], dim=0).tolist()
        
        is_threat = threat_probability > 0.7
        
        if is_threat:
            self.threats_detected += 1
        
        return {
            'threat_probability': threat_probability,
            'threat_type': threat_type,
            'is_threat': is_threat,
            'recommendations': self._generate_security_recommendations(security_metrics, threat_probability)
        }
    
    def _security_metrics_to_vector(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Преобразование метрик безопасности в вектор"""
        vector = [
            metrics.get('failed_logins', 0) / 100.0,
            metrics.get('unusual_access_patterns', 0) / 10.0,
            metrics.get('network_anomalies', 0) / 100.0,
            metrics.get('suspicious_requests', 0) / 100.0,
            # ... другие метрики
        ]
        
        while len(vector) < self.input_size:
            vector.append(0.0)
        
        return torch.tensor(vector[:self.input_size], dtype=torch.float32)
    
    def _generate_security_recommendations(self,
                                          metrics: Dict[str, float],
                                          threat_probability: float) -> List[str]:
        """Генерация рекомендаций по безопасности"""
        recommendations = []
        
        if threat_probability > 0.7:
            recommendations.append("Высокая вероятность угрозы - требуется немедленное расследование")
        
        if metrics.get('failed_logins', 0) > 50:
            recommendations.append("Множественные неудачные попытки входа - рассмотрите блокировку IP")
        
        if metrics.get('unusual_access_patterns', 0) > 5:
            recommendations.append("Необычные паттерны доступа - проверьте права доступа")
        
        return recommendations

