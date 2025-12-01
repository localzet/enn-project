"""
Ядро Emergent Neural Network (ENN)

Модульное ядро с динамической топологией и энергобюджетом
"""

from .enn_core import EmergentNeuralNetwork, Neuron, Connection
from .topology_evolution import TopologyEvolution
from .energy_management import EnergyManager
from .stability_analysis import StabilityAnalyzer
from .causality import CausalityAnalyzer

__all__ = [
    'EmergentNeuralNetwork',
    'Neuron',
    'Connection',
    'TopologyEvolution',
    'EnergyManager',
    'StabilityAnalyzer',
    'CausalityAnalyzer'
]

