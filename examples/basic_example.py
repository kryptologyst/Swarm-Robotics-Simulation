#!/usr/bin/env python3
"""Example script demonstrating swarm robotics simulation."""

import numpy as np
import matplotlib.pyplot as plt
from swarm_robotics import (
    SwarmSimulation, 
    FlockingController, 
    ConsensusController,
    FormationController,
    CoverageController,
    SwarmEvaluator,
    SwarmVisualizer,
    set_random_seeds,
    load_config
)


def run_flocking_example():
    """Run flocking behavior example."""
    print("Running Flocking Example...")
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Create simulation with flocking controller
    config = {
        "dt": 0.01,
        "max_steps": 2000,
        "world_size": [30.0, 30.0],
        "num_robots": 15,
        "controller_type": "flocking",
        "controller_config": {
            "perception_range": 3.0,
            "alignment_weight": 0.1,
            "cohesion_weight": 0.1,
            "separation_weight": 0.2,
            "max_velocity": 2.0,
            "max_acceleration": 5.0,
        }
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    
    print(f"Flocking Results:")
    print(f"  Coverage: {results.coverage:.3f}")
    print(f"  Collision Rate: {results.collision_rate:.3f}")
    print(f"  Energy Consumption: {results.energy_consumption:.3f}")
    
    return results


def run_consensus_example():
    """Run consensus behavior example."""
    print("\nRunning Consensus Example...")
    
    set_random_seeds(42)
    
    config = {
        "dt": 0.01,
        "max_steps": 2000,
        "world_size": [30.0, 30.0],
        "num_robots": 15,
        "controller_type": "consensus",
        "controller_config": {
            "communication_range": 4.0,
            "consensus_gain": 0.5,
            "consensus_type": "both",
            "max_velocity": 2.0,
            "max_acceleration": 5.0,
        }
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    
    print(f"Consensus Results:")
    print(f"  Coverage: {results.coverage:.3f}")
    print(f"  Collision Rate: {results.collision_rate:.3f}")
    print(f"  Energy Consumption: {results.energy_consumption:.3f}")
    
    return results


def run_formation_example():
    """Run formation control example."""
    print("\nRunning Formation Control Example...")
    
    set_random_seeds(42)
    
    config = {
        "dt": 0.01,
        "max_steps": 2000,
        "world_size": [30.0, 30.0],
        "num_robots": 12,
        "controller_type": "formation",
        "controller_config": {
            "formation_type": "circle",
            "formation_size": 3.0,
            "formation_center": [15.0, 15.0],
            "formation_gain": 1.0,
            "collision_avoidance_gain": 2.0,
            "max_velocity": 2.0,
            "max_acceleration": 5.0,
        }
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    
    print(f"Formation Control Results:")
    print(f"  Coverage: {results.coverage:.3f}")
    print(f"  Collision Rate: {results.collision_rate:.3f}")
    print(f"  Energy Consumption: {results.energy_consumption:.3f}")
    
    return results


def run_coverage_example():
    """Run coverage control example."""
    print("\nRunning Coverage Control Example...")
    
    set_random_seeds(42)
    
    config = {
        "dt": 0.01,
        "max_steps": 2000,
        "world_size": [30.0, 30.0],
        "num_robots": 15,
        "controller_type": "coverage",
        "controller_config": {
            "coverage_area": [[-10, -10], [10, 10]],
            "coverage_gain": 1.0,
            "collision_avoidance_gain": 2.0,
            "coverage_density": "uniform",
            "max_velocity": 2.0,
            "max_acceleration": 5.0,
        }
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    
    print(f"Coverage Control Results:")
    print(f"  Coverage: {results.coverage:.3f}")
    print(f"  Collision Rate: {results.collision_rate:.3f}")
    print(f"  Energy Consumption: {results.energy_consumption:.3f}")
    
    return results


def compare_algorithms():
    """Compare different swarm algorithms."""
    print("\nComparing Swarm Algorithms...")
    
    # Run all examples
    results_dict = {
        "Flocking": run_flocking_example(),
        "Consensus": run_consensus_example(),
        "Formation": run_formation_example(),
        "Coverage": run_coverage_example(),
    }
    
    # Create evaluator
    evaluator = SwarmEvaluator()
    
    # Compare algorithms
    comparison_df = evaluator.compare_algorithms(results_dict)
    
    print("\nAlgorithm Comparison:")
    print(comparison_df[["algorithm", "coverage_efficiency", "collision_rate", "energy_consumption"]])
    
    return results_dict


def visualize_results(results_dict):
    """Visualize results from different algorithms."""
    print("\nCreating Visualizations...")
    
    visualizer = SwarmVisualizer()
    
    # Create comparison plot
    fig = visualizer.plot_comparison(results_dict, "coverage_efficiency")
    plt.show()
    
    # Create trajectory plots for each algorithm
    for algorithm_name, results in results_dict.items():
        fig = visualizer.plot_trajectories(results)
        plt.title(f"{algorithm_name} Trajectories")
        plt.show()


def main():
    """Main function."""
    print("Swarm Robotics Simulation Examples")
    print("=" * 40)
    
    try:
        # Compare algorithms
        results_dict = compare_algorithms()
        
        # Visualize results
        visualize_results(results_dict)
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
