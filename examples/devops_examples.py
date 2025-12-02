"""
Примеры использования ENN в DevOps и FullStack

Демонстрация реальных use cases:
1. Мониторинг инфраструктуры
2. Адаптивная балансировка нагрузки
3. Оптимизация CI/CD
4. Обнаружение аномалий безопасности
"""

import torch
import numpy as np
from datetime import datetime, timedelta
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.devops_infrastructure import (
    InfrastructureMonitorENN,
    AdaptiveLoadBalancerENN,
    CI_CDOptimizerENN,
    SecurityAnomalyDetectorENN
)


def example_infrastructure_monitoring():
    """Пример: Мониторинг инфраструктуры"""
    print("="*70)
    print("Пример 1: Мониторинг инфраструктуры")
    print("="*70)
    
    monitor = InfrastructureMonitorENN(metric_count=5)
    
    # Симуляция метрик инфраструктуры
    metrics_examples = [
        {'metric_0': 45.0, 'metric_1': 60.0, 'metric_2': 30.0, 'metric_3': 20.0, 'metric_4': 10.0},  # Норма
        {'metric_0': 55.0, 'metric_1': 65.0, 'metric_2': 35.0, 'metric_3': 25.0, 'metric_4': 15.0},
        {'metric_0': 65.0, 'metric_1': 70.0, 'metric_2': 40.0, 'metric_3': 30.0, 'metric_4': 20.0},
        {'metric_0': 85.0, 'metric_1': 90.0, 'metric_2': 50.0, 'metric_3': 40.0, 'metric_4': 30.0},  # Высокая нагрузка
        {'metric_0': 95.0, 'metric_1': 95.0, 'metric_2': 60.0, 'metric_3': 50.0, 'metric_4': 40.0},  # Критично
    ]
    
    print("\nОбработка метрик инфраструктуры...")
    for i, metrics in enumerate(metrics_examples):
        result = monitor.process_metrics(metrics, datetime.now() + timedelta(minutes=i))
        
        print(f"\nШаг {i+1}:")
        print(f"  CPU: {metrics['metric_0']:.1f}%")
        print(f"  Memory: {metrics['metric_1']:.1f}%")
        print(f"  Anomaly Score: {result['anomaly_score']:.3f}")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        
        if result['recommendations']:
            print(f"  Рекомендации:")
            for rec in result['recommendations']:
                print(f"    - {rec}")
    
    # Рекомендации по масштабированию
    print("\n" + "-"*70)
    print("Рекомендации по масштабированию:")
    current = metrics_examples[-1]
    target = {'metric_0': 70.0, 'metric_1': 75.0, 'metric_2': 40.0, 'metric_3': 30.0, 'metric_4': 20.0}
    scaling = monitor.recommend_scaling(current, target)
    print(f"  Масштабирование: {scaling['scale_up'] or scaling['scale_down']}")
    print(f"  Фактор: {scaling['scaling_factor']:.2f}")
    print(f"  Рекомендуемые инстансы: {scaling['recommended_instances']}")
    
    stats = monitor.get_infrastructure_statistics()
    print(f"\nСтатистика:")
    print(f"  Аномалий обнаружено: {stats['anomalies_detected']}")
    print(f"  Событий масштабирования: {stats['scaling_events']}")


def example_load_balancing():
    """Пример: Адаптивная балансировка нагрузки"""
    print("\n" + "="*70)
    print("Пример 2: Адаптивная балансировка нагрузки")
    print("="*70)
    
    balancer = AdaptiveLoadBalancerENN(server_count=3)
    
    # Симуляция метрик серверов
    server_metrics_examples = [
        [
            {'cpu': 30.0, 'memory': 40.0, 'latency': 50.0},
            {'cpu': 25.0, 'memory': 35.0, 'latency': 45.0},
            {'cpu': 35.0, 'memory': 45.0, 'latency': 55.0},
        ],
        [
            {'cpu': 60.0, 'memory': 70.0, 'latency': 100.0},  # Сервер 0 перегружен
            {'cpu': 30.0, 'memory': 40.0, 'latency': 50.0},
            {'cpu': 25.0, 'memory': 35.0, 'latency': 45.0},
        ],
        [
            {'cpu': 80.0, 'memory': 85.0, 'latency': 150.0},  # Критическая нагрузка
            {'cpu': 50.0, 'memory': 60.0, 'latency': 80.0},
            {'cpu': 20.0, 'memory': 30.0, 'latency': 40.0},  # Сервер 2 свободен
        ],
    ]
    
    print("\nБалансировка нагрузки...")
    for i, server_metrics in enumerate(server_metrics_examples):
        distribution = balancer.balance_load(server_metrics)
        
        print(f"\nШаг {i+1}:")
        for j, metrics in enumerate(server_metrics):
            print(f"  Сервер {j}: CPU={metrics['cpu']:.1f}%, "
                  f"Memory={metrics['memory']:.1f}%, "
                  f"Latency={metrics['latency']:.1f}ms")
        
        print(f"  Распределение нагрузки:")
        for server_id, weight in distribution.items():
            print(f"    Сервер {server_id}: {weight*100:.1f}%")


def example_cicd_optimization():
    """Пример: Оптимизация CI/CD"""
    print("\n" + "="*70)
    print("Пример 3: Оптимизация CI/CD пайплайнов")
    print("="*70)
    
    optimizer = CI_CDOptimizerENN()
    
    # Симуляция коммитов
    commits = [
        {
            'files_changed': 5,
            'lines_added': 50,
            'lines_deleted': 10,
            'complexity': 20,
            'test_files_changed': 2,
            'actual_build_time': 300  # 5 минут
        },
        {
            'files_changed': 15,
            'lines_added': 200,
            'lines_deleted': 50,
            'complexity': 60,
            'test_files_changed': 5,
            'actual_build_time': 900  # 15 минут
        },
        {
            'files_changed': 30,
            'lines_added': 500,
            'lines_deleted': 100,
            'complexity': 100,
            'test_files_changed': 10,
            'actual_build_time': 1800  # 30 минут
        },
    ]
    
    print("\nОптимизация CI/CD пайплайнов...")
    for i, commit in enumerate(commits):
        commit_metrics = {k: v for k, v in commit.items() if k != 'actual_build_time'}
        result = optimizer.optimize_pipeline(commit_metrics, commit['actual_build_time'])
        
        print(f"\nКоммит {i+1}:")
        print(f"  Файлов изменено: {commit['files_changed']}")
        print(f"  Строк добавлено: {commit['lines_added']}")
        print(f"  Предсказанное время сборки: {result['predictions']['estimated_build_time']:.0f} сек")
        print(f"  Фактическое время сборки: {commit['actual_build_time']} сек")
        
        if result['recommendations']:
            print(f"  Рекомендации:")
            for rec in result['recommendations']:
                print(f"    - {rec}")


def example_security_monitoring():
    """Пример: Мониторинг безопасности"""
    print("\n" + "="*70)
    print("Пример 4: Обнаружение аномалий безопасности")
    print("="*70)
    
    detector = SecurityAnomalyDetectorENN()
    
    # Симуляция событий безопасности
    security_events = [
        {
            'failed_logins': 2,
            'unusual_access_patterns': 0,
            'network_anomalies': 1,
            'suspicious_requests': 0,
        },
        {
            'failed_logins': 15,
            'unusual_access_patterns': 2,
            'network_anomalies': 5,
            'suspicious_requests': 3,
        },
        {
            'failed_logins': 80,  # Подозрительно много
            'unusual_access_patterns': 10,
            'network_anomalies': 50,
            'suspicious_requests': 20,
        },
    ]
    
    print("\nАнализ событий безопасности...")
    for i, metrics in enumerate(security_events):
        result = detector.analyze_security_events(metrics)
        
        print(f"\nСобытие {i+1}:")
        print(f"  Неудачных входов: {metrics['failed_logins']}")
        print(f"  Необычных паттернов: {metrics['unusual_access_patterns']}")
        print(f"  Вероятность угрозы: {result['threat_probability']:.3f}")
        print(f"  Угроза обнаружена: {result['is_threat']}")
        
        if result['recommendations']:
            print(f"  Рекомендации:")
            for rec in result['recommendations']:
                print(f"    - {rec}")


if __name__ == "__main__":
    example_infrastructure_monitoring()
    example_load_balancing()
    example_cicd_optimization()
    example_security_monitoring()
    
    print("\n" + "="*70)
    print("Все примеры DevOps применений выполнены!")
    print("="*70)

