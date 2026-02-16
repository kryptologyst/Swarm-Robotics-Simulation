#!/usr/bin/env python3
"""
Project 662: Swarm Robotics Simulation - Modernized Version

This is a modernized version of the original swarm robotics simulation.
The original simple implementation has been replaced with a comprehensive
framework featuring advanced algorithms, evaluation metrics, and visualization.

For the original simple implementation, see examples/basic_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from swarm_robotics import (
    SwarmSimulation, 
    FlockingController, 
    SwarmEvaluator,
    SwarmVisualizer,
    set_random_seeds
)


def run_original_flocking_simulation():
    """
    Run the original flocking simulation using the modern framework.
    
    This recreates the behavior of the original 0662.py script but with
    improved algorithms, safety features, and evaluation metrics.
    """
    print("Running Modernized Swarm Robotics Simulation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Configuration matching the original simulation
    config = {
        "dt": 0.1,  # Original time step
        "max_steps": 100,  # Original number of steps
        "world_size": [10.0, 10.0],  # Original area size
        "num_robots": 10,  # Original number of robots
        "robot_radius": 0.2,  # Smaller radius for better visualization
        "controller_type": "flocking",
        "controller_config": {
            "perception_range": 1.0,  # Original perception range
            "alignment_weight": 0.1,
            "cohesion_weight": 0.1,
            "separation_weight": 0.2,
            "max_velocity": 0.5,  # Reduced for better control
            "max_acceleration": 2.0,
        },
        "boundaries": "reflective",  # Keep robots in bounds
        "collision_detection": True,
        "save_frequency": 10,  # Save every 10 steps like original
    }
    
    # Create simulation
    sim = SwarmSimulation(config)
    
    print(f"Initialized {config['num_robots']} robots in {config['world_size']} world")
    print(f"Running for {config['max_steps']} steps with dt={config['dt']}")
    
    # Run simulation
    results = sim.run()
    
    # Print results
    print(f"\nSimulation Results:")
    print(f"  Total Steps: {results.total_steps}")
    print(f"  Total Time: {results.total_time:.2f}s")
    print(f"  Coverage: {results.coverage:.3f}")
    print(f"  Collision Rate: {results.collision_rate:.3f}")
    print(f"  Energy Consumption: {results.energy_consumption:.3f}")
    
    return results


def visualize_results(results):
    """Visualize the simulation results."""
    print("\nCreating Visualizations...")
    
    # Create visualizer
    visualizer = SwarmVisualizer({
        "figsize": [8, 8],  # Match original plot size
        "robot_radius": 0.2,
        "trajectory_length": 20,
    })
    
    # Plot trajectories
    fig = visualizer.plot_trajectories(results)
    plt.title("Swarm Robotics Simulation with Flocking Behavior")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()
    
    # Create animation
    print("Creating animation...")
    anim = visualizer.create_animation(results)
    if anim:
        plt.show()


def run_advanced_examples():
    """Run additional advanced examples."""
    print("\n" + "=" * 50)
    print("Running Advanced Swarm Algorithms")
    print("=" * 50)
    
    # Consensus example
    print("\n1. Consensus Algorithm:")
    config = {
        "dt": 0.1,
        "max_steps": 100,
        "world_size": [10.0, 10.0],
        "num_robots": 10,
        "controller_type": "consensus",
        "controller_config": {
            "communication_range": 2.0,
            "consensus_gain": 0.5,
            "consensus_type": "both",
        }
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    print(f"   Consensus Results: Coverage={results.coverage:.3f}, Collisions={results.collision_rate:.3f}")
    
    # Formation control example
    print("\n2. Formation Control:")
    config["controller_type"] = "formation"
    config["controller_config"] = {
        "formation_type": "circle",
        "formation_size": 2.0,
        "formation_center": [5.0, 5.0],
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    print(f"   Formation Results: Coverage={results.coverage:.3f}, Collisions={results.collision_rate:.3f}")
    
    # Coverage control example
    print("\n3. Coverage Control:")
    config["controller_type"] = "coverage"
    config["controller_config"] = {
        "coverage_area": [[-5, -5], [5, 5]],
        "coverage_gain": 1.0,
    }
    
    sim = SwarmSimulation(config)
    results = sim.run()
    print(f"   Coverage Results: Coverage={results.coverage:.3f}, Collisions={results.collision_rate:.3f}")


def main():
    """Main function."""
    try:
        # Run the modernized version of the original simulation
        results = run_original_flocking_simulation()
        
        # Visualize results
        visualize_results(results)
        
        # Run advanced examples
        run_advanced_examples()
        
        print("\n" + "=" * 50)
        print("Simulation completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise


if __name__ == "__main__":
    main()
