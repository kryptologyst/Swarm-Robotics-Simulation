"""Swarm Robotics Simulation Framework.

A modern, research-ready swarm robotics simulation framework featuring advanced
multi-robot coordination algorithms, comprehensive evaluation metrics, and
interactive demos.
"""

__version__ = "1.0.0"
__author__ = "Swarm Robotics Team"
__email__ = "team@example.com"

from .simulation import SwarmSimulation
from .controllers.flocking import FlockingController
from .controllers.consensus import ConsensusController
from .controllers.formation import FormationController
from .controllers.coverage import CoverageController
from .evaluation import SwarmEvaluator, Metrics
from .visualization import SwarmVisualizer
from .utils import set_random_seeds, get_device, load_config, save_config

__all__ = [
    "SwarmSimulation",
    "FlockingController",
    "ConsensusController", 
    "FormationController",
    "CoverageController",
    "SwarmEvaluator",
    "Metrics",
    "SwarmVisualizer",
    "set_random_seeds",
    "get_device",
    "load_config",
    "save_config",
]
