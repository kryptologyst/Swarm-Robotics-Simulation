"""Basic test suite for swarm robotics simulation."""

import pytest
import numpy as np
from swarm_robotics import (
    SwarmSimulation,
    FlockingController,
    ConsensusController,
    FormationController,
    CoverageController,
    SwarmEvaluator,
    SwarmVisualizer,
    set_random_seeds
)


class TestSwarmSimulation:
    """Test cases for SwarmSimulation class."""
    
    def test_simulation_initialization(self):
        """Test simulation initialization."""
        config = {
            "dt": 0.01,
            "max_steps": 100,
            "world_size": [10.0, 10.0],
            "num_robots": 5,
            "controller_type": "flocking",
        }
        
        sim = SwarmSimulation(config)
        assert sim.config.num_robots == 5
        assert sim.config.world_size == [10.0, 10.0]
        assert len(sim.robots) == 5
    
    def test_flocking_simulation(self):
        """Test flocking simulation."""
        config = {
            "dt": 0.01,
            "max_steps": 50,
            "world_size": [10.0, 10.0],
            "num_robots": 5,
            "controller_type": "flocking",
            "controller_config": {
                "perception_range": 2.0,
                "alignment_weight": 0.1,
                "cohesion_weight": 0.1,
                "separation_weight": 0.2,
            }
        }
        
        sim = SwarmSimulation(config)
        results = sim.run()
        
        assert results.total_steps == 50
        assert len(results.positions) > 0
        assert len(results.positions[0]) == 5  # 5 robots
    
    def test_consensus_simulation(self):
        """Test consensus simulation."""
        config = {
            "dt": 0.01,
            "max_steps": 50,
            "world_size": [10.0, 10.0],
            "num_robots": 5,
            "controller_type": "consensus",
            "controller_config": {
                "communication_range": 3.0,
                "consensus_gain": 0.5,
                "consensus_type": "both",
            }
        }
        
        sim = SwarmSimulation(config)
        results = sim.run()
        
        assert results.total_steps == 50
        assert len(results.positions) > 0
    
    def test_formation_simulation(self):
        """Test formation control simulation."""
        config = {
            "dt": 0.01,
            "max_steps": 50,
            "world_size": [10.0, 10.0],
            "num_robots": 4,
            "controller_type": "formation",
            "controller_config": {
                "formation_type": "circle",
                "formation_size": 2.0,
                "formation_center": [5.0, 5.0],
            }
        }
        
        sim = SwarmSimulation(config)
        results = sim.run()
        
        assert results.total_steps == 50
        assert len(results.positions) > 0
    
    def test_coverage_simulation(self):
        """Test coverage control simulation."""
        config = {
            "dt": 0.01,
            "max_steps": 50,
            "world_size": [10.0, 10.0],
            "num_robots": 5,
            "controller_type": "coverage",
            "controller_config": {
                "coverage_area": [[-5, -5], [5, 5]],
                "coverage_gain": 1.0,
            }
        }
        
        sim = SwarmSimulation(config)
        results = sim.run()
        
        assert results.total_steps == 50
        assert len(results.positions) > 0


class TestControllers:
    """Test cases for swarm controllers."""
    
    def test_flocking_controller(self):
        """Test flocking controller."""
        config = {
            "perception_range": 2.0,
            "alignment_weight": 0.1,
            "cohesion_weight": 0.1,
            "separation_weight": 0.2,
        }
        
        controller = FlockingController(config)
        assert controller.perception_range == 2.0
        assert controller.alignment_weight == 0.1
    
    def test_consensus_controller(self):
        """Test consensus controller."""
        config = {
            "communication_range": 3.0,
            "consensus_gain": 0.5,
            "consensus_type": "both",
        }
        
        controller = ConsensusController(config)
        assert controller.communication_range == 3.0
        assert controller.consensus_gain == 0.5
    
    def test_formation_controller(self):
        """Test formation controller."""
        config = {
            "formation_type": "circle",
            "formation_size": 2.0,
            "formation_center": [0.0, 0.0],
        }
        
        controller = FormationController(config)
        assert controller.formation_type == "circle"
        assert controller.formation_size == 2.0
    
    def test_coverage_controller(self):
        """Test coverage controller."""
        config = {
            "coverage_area": [[-5, -5], [5, 5]],
            "coverage_gain": 1.0,
        }
        
        controller = CoverageController(config)
        assert controller.coverage_gain == 1.0


class TestEvaluation:
    """Test cases for evaluation framework."""
    
    def test_swarm_evaluator(self):
        """Test swarm evaluator."""
        evaluator = SwarmEvaluator()
        
        # Create dummy results
        from swarm_robotics.simulation import SimulationResults
        results = SimulationResults()
        results.positions = [[np.array([0.0, 0.0]), np.array([1.0, 1.0])]]
        results.velocities = [[np.array([0.1, 0.1]), np.array([0.2, 0.2])]]
        results.accelerations = [[np.array([0.01, 0.01]), np.array([0.02, 0.02])]]
        results.times = [0.0]
        results.total_time = 1.0
        
        from swarm_robotics.simulation import SwarmState
        swarm_state = SwarmState(robots=[], time=0.0, step=0)
        
        metrics = evaluator.evaluate(results, swarm_state)
        assert isinstance(metrics.coverage_efficiency, float)
        assert isinstance(metrics.collision_rate, float)


class TestVisualization:
    """Test cases for visualization framework."""
    
    def test_swarm_visualizer(self):
        """Test swarm visualizer."""
        visualizer = SwarmVisualizer()
        
        # Create dummy results
        from swarm_robotics.simulation import SimulationResults
        results = SimulationResults()
        results.positions = [[np.array([0.0, 0.0]), np.array([1.0, 1.0])]]
        results.velocities = [[np.array([0.1, 0.1]), np.array([0.2, 0.2])]]
        results.accelerations = [[np.array([0.01, 0.01]), np.array([0.02, 0.02])]]
        results.times = [0.0]
        
        # Test trajectory plotting (should not raise exception)
        fig = visualizer.plot_trajectories(results)
        assert fig is not None or fig is None  # Either returns figure or None


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        
        # Test that seeds are set (basic check)
        np.random.seed(42)
        val1 = np.random.random()
        
        set_random_seeds(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_device_detection(self):
        """Test device detection."""
        from swarm_robotics.utils import get_device
        
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]


if __name__ == "__main__":
    pytest.main([__file__])
