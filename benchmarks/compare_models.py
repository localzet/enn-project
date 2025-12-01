"""
Сравнение ENN с базовыми моделями

Сравниваем с:
- MLP (Multi-Layer Perceptron)
- RNN (Recurrent Neural Network)
- NEAT (NeuroEvolution of Augmenting Topologies)
- Transformer (для последовательностей)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import EmergentNeuralNetwork


class MLPBaseline(nn.Module):
    """Базовый MLP для сравнения"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RNNBaseline(nn.Module):
    """Базовый RNN для сравнения"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Если x не последовательность, делаем её последовательностью
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def compare_with_baselines(dataset_name: str,
                          train_data: Tuple[torch.Tensor, torch.Tensor],
                          test_data: Tuple[torch.Tensor, torch.Tensor],
                          epochs: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Сравнение ENN с базовыми моделями
    
    Args:
        dataset_name: Название датасета
        train_data: (X_train, y_train)
        test_data: (X_test, y_test)
        epochs: Количество эпох обучения
    
    Returns:
        Словарь с результатами сравнения
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    input_size = X_train.shape[1]
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    results = {}
    
    # 1. ENN
    print(f"Обучение ENN на {dataset_name}...")
    enn = EmergentNeuralNetwork(input_size=input_size, output_size=output_size)
    
    start_time = time.time()
    enn_errors = []
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i:i+1]
            y = y_train[i:i+1]
            metrics = enn.learn(x, y, reward=1.0)
            enn_errors.append(metrics['error'])
    
    enn_time = time.time() - start_time
    
    with torch.no_grad():
        enn_predictions = enn.forward(X_test)
        enn_test_error = nn.functional.mse_loss(enn_predictions, y_test).item()
    
    results['ENN'] = {
        'train_error': np.mean(enn_errors[-100:]),
        'test_error': enn_test_error,
        'training_time': enn_time,
        'num_neurons': len(enn.neurons),
        'num_connections': len(enn.connections),
        'energy_budget': enn.energy_manager.budget
    }
    
    # 2. MLP
    print(f"Обучение MLP на {dataset_name}...")
    mlp = MLPBaseline(input_size, output_size)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    mlp_errors = []
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i:i+1]
            y = y_train[i:i+1]
            
            optimizer.zero_grad()
            pred = mlp(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            mlp_errors.append(loss.item())
    
    mlp_time = time.time() - start_time
    
    with torch.no_grad():
        mlp_predictions = mlp(X_test)
        mlp_test_error = criterion(mlp_predictions, y_test).item()
    
    results['MLP'] = {
        'train_error': np.mean(mlp_errors[-100:]),
        'test_error': mlp_test_error,
        'training_time': mlp_time,
        'num_parameters': sum(p.numel() for p in mlp.parameters())
    }
    
    # 3. RNN (если данные последовательные)
    if len(X_train.shape) == 3 or input_size > 10:
        print(f"Обучение RNN на {dataset_name}...")
        rnn = RNNBaseline(input_size, output_size)
        optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=0.01)
        
        start_time = time.time()
        rnn_errors = []
        for epoch in range(epochs):
            for i in range(len(X_train)):
                x = X_train[i:i+1]
                y = y_train[i:i+1]
                
                optimizer_rnn.zero_grad()
                pred = rnn(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer_rnn.step()
                
                rnn_errors.append(loss.item())
        
        rnn_time = time.time() - start_time
        
        with torch.no_grad():
            rnn_predictions = rnn(X_test)
            rnn_test_error = criterion(rnn_predictions, y_test).item()
        
        results['RNN'] = {
            'train_error': np.mean(rnn_errors[-100:]),
            'test_error': rnn_test_error,
            'training_time': rnn_time,
            'num_parameters': sum(p.numel() for p in rnn.parameters())
        }
    
    return results


def print_comparison_results(results: Dict[str, Dict[str, float]]):
    """Красивый вывод результатов сравнения"""
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Ошибка обучения: {metrics['train_error']:.6f}")
        print(f"  Ошибка теста: {metrics['test_error']:.6f}")
        print(f"  Время обучения: {metrics['training_time']:.2f} сек")
        
        if 'num_neurons' in metrics:
            print(f"  Нейронов: {metrics['num_neurons']}")
            print(f"  Связей: {metrics['num_connections']}")
            print(f"  Энергобюджет: {metrics['energy_budget']:.2f}")
        elif 'num_parameters' in metrics:
            print(f"  Параметров: {metrics['num_parameters']}")
    
    print("\n" + "="*70)

