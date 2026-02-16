"""Evaluation framework for swarm robotics performance."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
import pandas as pd

from ..simulation import SimulationResults, SwarmState
from ..controllers.base import SwarmController


@dataclass
class Metrics:
    """Container for swarm performance metrics."""
    
    # Coverage metrics
    coverage_area: float = 0.0
    coverage_efficiency: float = 0.0
    coverage_time: float = 0.0
    
    # Formation metrics
    formation_error: float = 0.0
    formation_stability: float = 0.0
    formation_convergence_time: float = 0.0
    
    # Collision metrics
    collision_rate: float = 0.0
    collision_count: int = 0
    near_collision_rate: float = 0.0
    
    # Energy metrics
    energy_consumption: float = 0.0
    energy_efficiency: float = 0.0
    velocity_variance: float = 0.0
    
    # Communication metrics
    communication_load: float = 0.0
    connectivity: float = 0.0
    consensus_error: float = 0.0
    
    # Task-specific metrics
    task_completion_time: float = 0.0
    task_success_rate: float = 0.0
    exploration_efficiency: float = 0.0
    
    # Stability metrics
    stability_margin: float = 0.0
    convergence_rate: float = 0.0
    robustness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "coverage_area": self.coverage_area,
            "coverage_efficiency": self.coverage_efficiency,
            "coverage_time": self.coverage_time,
            "formation_error": self.formation_error,
            "formation_stability": self.formation_stability,
            "formation_convergence_time": self.formation_convergence_time,
            "collision_rate": self.collision_rate,
            "collision_count": self.collision_count,
            "near_collision_rate": self.near_collision_rate,
            "energy_consumption": self.energy_consumption,
            "energy_efficiency": self.energy_efficiency,
            "velocity_variance": self.velocity_variance,
            "communication_load": self.communication_load,
            "connectivity": self.connectivity,
            "consensus_error": self.consensus_error,
            "task_completion_time": self.task_completion_time,
            "task_success_rate": self.task_success_rate,
            "exploration_efficiency": self.exploration_efficiency,
            "stability_margin": self.stability_margin,
            "convergence_rate": self.convergence_rate,
            "robustness_score": self.robustness_score,
        }


class MetricEvaluator(ABC):
    """Abstract base class for metric evaluators."""
    
    @abstractmethod
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> float:
        """Evaluate a specific metric.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Metric value
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass


class CoverageEvaluator(MetricEvaluator):
    """Evaluator for coverage metrics."""
    
    def __init__(self, target_area: Optional[np.ndarray] = None):
        """Initialize coverage evaluator.
        
        Args:
            target_area: Target area to cover [[x_min, y_min], [x_max, y_max]]
        """
        self.target_area = target_area
    
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> float:
        """Evaluate coverage efficiency.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Coverage efficiency (0-1)
        """
        if not results.positions:
            return 0.0
        
        # Get final positions
        final_positions = np.array(results.positions[-1])
        
        if self.target_area is not None:
            # Compute coverage of target area
            return self._compute_target_coverage(final_positions)
        else:
            # Compute area spanned by robots
            return self._compute_area_coverage(final_positions)
    
    def _compute_target_coverage(self, positions: np.ndarray) -> float:
        """Compute coverage of target area."""
        if len(positions) == 0:
            return 0.0
        
        # Simple coverage based on robot positions within target area
        target_min = self.target_area[0]
        target_max = self.target_area[1]
        
        covered_positions = 0
        for pos in positions:
            if (target_min[0] <= pos[0] <= target_max[0] and 
                target_min[1] <= pos[1] <= target_max[1]):
                covered_positions += 1
        
        total_area = np.prod(target_max - target_min)
        robot_area = len(positions) * np.pi * 0.5**2  # Assuming robot radius 0.5
        
        return min(robot_area / total_area, 1.0)
    
    def _compute_area_coverage(self, positions: np.ndarray) -> float:
        """Compute area coverage."""
        if len(positions) < 2:
            return 0.0
        
        # Compute convex hull area
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(positions)
            hull_area = hull.volume  # In 2D, volume is area
            return hull_area
        except:
            # Fallback to bounding box
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            return np.prod(max_pos - min_pos)
    
    def get_name(self) -> str:
        """Get metric name."""
        return "coverage_efficiency"


class FormationEvaluator(MetricEvaluator):
    """Evaluator for formation metrics."""
    
    def __init__(self, target_formation: Optional[List[np.ndarray]] = None):
        """Initialize formation evaluator.
        
        Args:
            target_formation: Target formation positions
        """
        self.target_formation = target_formation
    
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> float:
        """Evaluate formation error.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Formation error (lower is better)
        """
        if not results.positions or not self.target_formation:
            return 0.0
        
        final_positions = np.array(results.positions[-1])
        
        if len(final_positions) != len(self.target_formation):
            return float('inf')
        
        # Compute formation error
        total_error = 0.0
        for i, (actual_pos, target_pos) in enumerate(zip(final_positions, self.target_formation)):
            error = np.linalg.norm(actual_pos - target_pos)
            total_error += error
        
        return total_error / len(final_positions)
    
    def get_name(self) -> str:
        """Get metric name."""
        return "formation_error"


class CollisionEvaluator(MetricEvaluator):
    """Evaluator for collision metrics."""
    
    def __init__(self, collision_radius: float = 1.0):
        """Initialize collision evaluator.
        
        Args:
            collision_radius: Collision detection radius
        """
        self.collision_radius = collision_radius
    
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> float:
        """Evaluate collision rate.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Collision rate (collisions per robot per second)
        """
        if not results.positions:
            return 0.0
        
        collision_count = 0
        total_pairs = 0
        
        for step_positions in results.positions:
            positions = np.array(step_positions)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    total_pairs += 1
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < 2 * self.collision_radius:
                        collision_count += 1
        
        if total_pairs == 0:
            return 0.0
        
        # Normalize by time and number of robots
        total_time = results.total_time if results.total_time > 0 else 1.0
        num_robots = len(results.positions[0]) if results.positions else 1
        
        return collision_count / (total_pairs * total_time * num_robots)
    
    def get_name(self) -> str:
        """Get metric name."""
        return "collision_rate"


class EnergyEvaluator(MetricEvaluator):
    """Evaluator for energy consumption metrics."""
    
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> float:
        """Evaluate energy consumption.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Total energy consumption
        """
        if not results.accelerations:
            return 0.0
        
        total_energy = 0.0
        for step_accelerations in results.accelerations:
            for accel in step_accelerations:
                # Energy proportional to acceleration squared
                total_energy += np.linalg.norm(accel) ** 2
        
        return total_energy
    
    def get_name(self) -> str:
        """Get metric name."""
        return "energy_consumption"


class SwarmEvaluator:
    """Main evaluator for swarm performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize swarm evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluators
        self.evaluators: Dict[str, MetricEvaluator] = {}
        self._initialize_evaluators()
    
    def _initialize_evaluators(self) -> None:
        """Initialize metric evaluators."""
        # Coverage evaluator
        target_area = self.config.get("target_area", None)
        if target_area is not None:
            target_area = np.array(target_area)
        self.evaluators["coverage"] = CoverageEvaluator(target_area)
        
        # Formation evaluator
        target_formation = self.config.get("target_formation", None)
        if target_formation is not None:
            target_formation = [np.array(pos) for pos in target_formation]
        self.evaluators["formation"] = FormationEvaluator(target_formation)
        
        # Collision evaluator
        collision_radius = self.config.get("collision_radius", 1.0)
        self.evaluators["collision"] = CollisionEvaluator(collision_radius)
        
        # Energy evaluator
        self.evaluators["energy"] = EnergyEvaluator()
    
    def evaluate(self, results: SimulationResults, swarm_state: SwarmState) -> Metrics:
        """Evaluate swarm performance.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Performance metrics
        """
        metrics = Metrics()
        
        # Evaluate each metric
        for name, evaluator in self.evaluators.items():
            try:
                value = evaluator.evaluate(results, swarm_state)
                
                # Set metric value based on evaluator type
                if name == "coverage":
                    metrics.coverage_efficiency = value
                elif name == "formation":
                    metrics.formation_error = value
                elif name == "collision":
                    metrics.collision_rate = value
                elif name == "energy":
                    metrics.energy_consumption = value
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {name}: {e}")
        
        # Compute additional metrics
        self._compute_additional_metrics(metrics, results, swarm_state)
        
        return metrics
    
    def _compute_additional_metrics(self, metrics: Metrics, results: SimulationResults, swarm_state: SwarmState) -> None:
        """Compute additional metrics not covered by evaluators."""
        if not results.positions:
            return
        
        # Velocity variance
        if results.velocities:
            all_velocities = np.concatenate(results.velocities)
            metrics.velocity_variance = np.var(all_velocities, axis=0).sum()
        
        # Coverage area
        if results.positions:
            final_positions = np.array(results.positions[-1])
            if len(final_positions) > 0:
                min_pos = np.min(final_positions, axis=0)
                max_pos = np.max(final_positions, axis=0)
                metrics.coverage_area = np.prod(max_pos - min_pos)
        
        # Task completion time (simplified)
        metrics.task_completion_time = results.total_time
        
        # Stability margin (simplified)
        if results.velocities:
            velocity_norms = [np.linalg.norm(v) for step_velocities in results.velocities for v in step_velocities]
            if velocity_norms:
                metrics.stability_margin = 1.0 / (1.0 + np.std(velocity_norms))
    
    def compare_algorithms(self, results_dict: Dict[str, SimulationResults]) -> pd.DataFrame:
        """Compare performance of different algorithms.
        
        Args:
            results_dict: Dictionary mapping algorithm names to results
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for algorithm_name, results in results_dict.items():
            # Create dummy swarm state for evaluation
            swarm_state = SwarmState(
                robots=[],
                time=results.total_time,
                step=results.total_steps
            )
            
            metrics = self.evaluate(results, swarm_state)
            metrics_dict = metrics.to_dict()
            metrics_dict["algorithm"] = algorithm_name
            
            comparison_data.append(metrics_dict)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, results: SimulationResults, swarm_state: SwarmState) -> str:
        """Generate performance report.
        
        Args:
            results: Simulation results
            swarm_state: Current swarm state
            
        Returns:
            Formatted report string
        """
        metrics = self.evaluate(results, swarm_state)
        
        report = f"""
Swarm Performance Report
========================

Simulation Summary:
- Total Steps: {results.total_steps}
- Total Time: {results.total_time:.2f}s
- Number of Robots: {len(results.positions[0]) if results.positions else 0}

Performance Metrics:
- Coverage Efficiency: {metrics.coverage_efficiency:.3f}
- Formation Error: {metrics.formation_error:.3f}
- Collision Rate: {metrics.collision_rate:.3f}
- Energy Consumption: {metrics.energy_consumption:.3f}
- Velocity Variance: {metrics.velocity_variance:.3f}
- Stability Margin: {metrics.stability_margin:.3f}

Controller Info:
{json.dumps(results.controller_info, indent=2)}
"""
        
        return report
    
    def save_metrics(self, metrics: Metrics, filepath: Union[str, Path]) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Metrics to save
            filepath: Path to save file
        """
        metrics_dict = metrics.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")
    
    def load_metrics(self, filepath: Union[str, Path]) -> Metrics:
        """Load metrics from file.
        
        Args:
            filepath: Path to metrics file
            
        Returns:
            Loaded metrics
        """
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        
        return Metrics(**metrics_dict)
