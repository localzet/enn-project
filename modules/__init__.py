"""
Специализированные модули для различных применений ENN
"""

from .autonomous_agents import AutonomousAgentENN
from .continual_learning import ContinualLearningENN
from .resource_constrained import ResourceConstrainedENN

__all__ = [
    'AutonomousAgentENN',
    'ContinualLearningENN',
    'ResourceConstrainedENN'
]

