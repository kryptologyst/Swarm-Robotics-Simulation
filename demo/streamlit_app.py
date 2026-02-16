"""Streamlit demo application for swarm robotics simulation."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time

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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Swarm Robotics Simulation",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Swarm Robotics Simulation")
    st.markdown("Interactive simulation and visualization of swarm robotics algorithms")
    
    # Sidebar configuration
    st.sidebar.header("Simulation Configuration")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Flocking", "Consensus", "Formation Control", "Coverage Control"],
        help="Select the swarm control algorithm to simulate"
    )
    
    # Simulation parameters
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        num_robots = st.slider("Number of Robots", 5, 50, 20)
        world_size = st.slider("World Size", 10, 100, 50)
        max_steps = st.slider("Max Steps", 100, 5000, 1000)
    
    with col2:
        dt = st.slider("Time Step (dt)", 0.001, 0.1, 0.01, format="%.3f")
        perception_range = st.slider("Perception Range", 1.0, 10.0, 2.0)
        max_velocity = st.slider("Max Velocity", 0.5, 5.0, 2.0)
    
    # Algorithm-specific parameters
    st.sidebar.subheader("Algorithm Parameters")
    
    if algorithm == "Flocking":
        alignment_weight = st.sidebar.slider("Alignment Weight", 0.0, 1.0, 0.1)
        cohesion_weight = st.sidebar.slider("Cohesion Weight", 0.0, 1.0, 0.1)
        separation_weight = st.sidebar.slider("Separation Weight", 0.0, 1.0, 0.2)
        
        controller_config = {
            "perception_range": perception_range,
            "alignment_weight": alignment_weight,
            "cohesion_weight": cohesion_weight,
            "separation_weight": separation_weight,
            "max_velocity": max_velocity,
            "max_acceleration": 5.0,
        }
        controller_type = "flocking"
    
    elif algorithm == "Consensus":
        communication_range = st.sidebar.slider("Communication Range", 1.0, 10.0, 3.0)
        consensus_gain = st.sidebar.slider("Consensus Gain", 0.1, 2.0, 0.5)
        consensus_type = st.sidebar.selectbox("Consensus Type", ["position", "velocity", "both"])
        
        controller_config = {
            "communication_range": communication_range,
            "consensus_gain": consensus_gain,
            "consensus_type": consensus_type,
            "max_velocity": max_velocity,
            "max_acceleration": 5.0,
        }
        controller_type = "consensus"
    
    elif algorithm == "Formation Control":
        formation_type = st.sidebar.selectbox("Formation Type", ["line", "circle", "diamond"])
        formation_size = st.sidebar.slider("Formation Size", 1.0, 10.0, 3.0)
        formation_gain = st.sidebar.slider("Formation Gain", 0.1, 3.0, 1.0)
        
        controller_config = {
            "formation_type": formation_type,
            "formation_size": formation_size,
            "formation_center": [world_size/2, world_size/2],
            "formation_gain": formation_gain,
            "collision_avoidance_gain": 2.0,
            "max_velocity": max_velocity,
            "max_acceleration": 5.0,
        }
        controller_type = "formation"
    
    elif algorithm == "Coverage Control":
        coverage_gain = st.sidebar.slider("Coverage Gain", 0.1, 3.0, 1.0)
        coverage_area_size = st.sidebar.slider("Coverage Area Size", 5, 30, 15)
        
        controller_config = {
            "coverage_area": [[-coverage_area_size/2, -coverage_area_size/2], 
                            [coverage_area_size/2, coverage_area_size/2]],
            "coverage_gain": coverage_gain,
            "collision_avoidance_gain": 2.0,
            "coverage_density": "uniform",
            "max_velocity": max_velocity,
            "max_acceleration": 5.0,
        }
        controller_type = "coverage"
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        run_simulation(algorithm, controller_type, controller_config, 
                      num_robots, world_size, max_steps, dt)
    
    # Display information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This demo showcases different swarm robotics algorithms:
    
    - **Flocking**: Reynolds' boids algorithm
    - **Consensus**: Distributed agreement
    - **Formation**: Geometric formations
    - **Coverage**: Optimal area coverage
    """)


def run_simulation(algorithm, controller_type, controller_config, 
                  num_robots, world_size, max_steps, dt):
    """Run the simulation and display results."""
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Create simulation configuration
    config = {
        "dt": dt,
        "max_steps": max_steps,
        "world_size": [world_size, world_size],
        "num_robots": num_robots,
        "robot_radius": 0.5,
        "controller_type": controller_type,
        "controller_config": controller_config,
        "boundaries": "reflective",
        "collision_detection": True,
        "save_frequency": 10,
    }
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create simulation
    status_text.text("Initializing simulation...")
    progress_bar.progress(10)
    
    sim = SwarmSimulation(config)
    progress_bar.progress(20)
    
    # Run simulation
    status_text.text("Running simulation...")
    
    start_time = time.time()
    results = sim.run()
    end_time = time.time()
    
    progress_bar.progress(100)
    status_text.text(f"Simulation completed in {end_time - start_time:.2f}s")
    
    # Display results
    display_results(results, algorithm)


def display_results(results, algorithm):
    """Display simulation results."""
    
    st.header("📊 Simulation Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Coverage", f"{results.coverage:.3f}")
    
    with col2:
        st.metric("Collision Rate", f"{results.collision_rate:.3f}")
    
    with col3:
        st.metric("Energy Consumption", f"{results.energy_consumption:.3f}")
    
    with col4:
        st.metric("Total Time", f"{results.total_time:.2f}s")
    
    # Visualizations
    st.header("📈 Visualizations")
    
    # Create visualizer
    visualizer = SwarmVisualizer({
        "figsize": [10, 8],
        "robot_radius": 0.5,
        "trajectory_length": 50,
    })
    
    # Trajectory plot
    st.subheader("Robot Trajectories")
    
    fig = visualizer.plot_trajectories(results)
    if fig:
        st.pyplot(fig)
    
    # Performance metrics over time
    st.subheader("Performance Metrics Over Time")
    
    if len(results.times) > 1:
        # Create metrics over time plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Coverage over time (simplified)
        coverage_over_time = [results.coverage] * len(results.times)
        axes[0, 0].plot(results.times, coverage_over_time, 'b-', linewidth=2)
        axes[0, 0].set_title('Coverage Over Time')
        axes[0, 0].set_ylabel('Coverage')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity variance over time
        velocity_variance = []
        for step_velocities in results.velocities:
            all_velocities = np.concatenate(step_velocities)
            variance = np.var(all_velocities, axis=0).sum()
            velocity_variance.append(variance)
        
        axes[0, 1].plot(results.times, velocity_variance, 'r-', linewidth=2)
        axes[0, 1].set_title('Velocity Variance Over Time')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy over time
        energy_over_time = []
        for step_accelerations in results.accelerations:
            energy = sum(np.linalg.norm(accel)**2 for accel in step_accelerations)
            energy_over_time.append(energy)
        
        axes[1, 0].plot(results.times, energy_over_time, 'g-', linewidth=2)
        axes[1, 0].set_title('Energy Consumption Over Time')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Robot spread over time
        spread_over_time = []
        for step_positions in results.positions:
            positions = np.array(step_positions)
            if len(positions) > 1:
                centroid = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - centroid, axis=1)
                spread = np.std(distances)
                spread_over_time.append(spread)
            else:
                spread_over_time.append(0)
        
        axes[1, 1].plot(results.times, spread_over_time, 'm-', linewidth=2)
        axes[1, 1].set_title('Robot Spread Over Time')
        axes[1, 1].set_ylabel('Spread')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Interactive plotly visualization
    st.subheader("Interactive Visualization")
    
    if results.positions:
        # Create interactive plot
        fig = go.Figure()
        
        num_robots = len(results.positions[0])
        colors = px.colors.qualitative.Set3[:num_robots]
        
        for robot_id in range(num_robots):
            trajectory = [step[robot_id] for step in results.positions]
            trajectory = np.array(trajectory)
            
            # Add trajectory line
            fig.add_trace(go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode='lines+markers',
                name=f'Robot {robot_id}',
                line=dict(color=colors[robot_id], width=2),
                marker=dict(size=4),
                opacity=0.8
            ))
        
        fig.update_layout(
            title=f'{algorithm} Algorithm - Robot Trajectories',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            hovermode='closest',
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Algorithm comparison
    st.header("🔍 Algorithm Comparison")
    
    # Run quick comparison
    if st.button("Compare All Algorithms"):
        compare_algorithms(num_robots, world_size, max_steps, dt)


def compare_algorithms(num_robots, world_size, max_steps, dt):
    """Compare all algorithms."""
    
    st.subheader("Algorithm Performance Comparison")
    
    algorithms = {
        "Flocking": {
            "type": "flocking",
            "config": {
                "perception_range": 2.0,
                "alignment_weight": 0.1,
                "cohesion_weight": 0.1,
                "separation_weight": 0.2,
                "max_velocity": 2.0,
                "max_acceleration": 5.0,
            }
        },
        "Consensus": {
            "type": "consensus",
            "config": {
                "communication_range": 3.0,
                "consensus_gain": 0.5,
                "consensus_type": "both",
                "max_velocity": 2.0,
                "max_acceleration": 5.0,
            }
        },
        "Formation": {
            "type": "formation",
            "config": {
                "formation_type": "circle",
                "formation_size": 3.0,
                "formation_center": [world_size/2, world_size/2],
                "formation_gain": 1.0,
                "collision_avoidance_gain": 2.0,
                "max_velocity": 2.0,
                "max_acceleration": 5.0,
            }
        },
        "Coverage": {
            "type": "coverage",
            "config": {
                "coverage_area": [[-world_size/4, -world_size/4], [world_size/4, world_size/4]],
                "coverage_gain": 1.0,
                "collision_avoidance_gain": 2.0,
                "max_velocity": 2.0,
                "max_acceleration": 5.0,
            }
        }
    }
    
    # Run simulations
    results_dict = {}
    progress_bar = st.progress(0)
    
    for i, (name, algo_config) in enumerate(algorithms.items()):
        st.text(f"Running {name} algorithm...")
        
        config = {
            "dt": dt,
            "max_steps": max_steps,
            "world_size": [world_size, world_size],
            "num_robots": num_robots,
            "controller_type": algo_config["type"],
            "controller_config": algo_config["config"],
            "boundaries": "reflective",
            "collision_detection": True,
            "save_frequency": 10,
        }
        
        sim = SwarmSimulation(config)
        results = sim.run()
        results_dict[name] = results
        
        progress_bar.progress((i + 1) / len(algorithms))
    
    # Create comparison chart
    comparison_data = []
    for name, results in results_dict.items():
        comparison_data.append({
            "Algorithm": name,
            "Coverage": results.coverage,
            "Collision Rate": results.collision_rate,
            "Energy Consumption": results.energy_consumption,
            "Total Time": results.total_time,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(df, use_container_width=True)
    
    # Create comparison chart
    fig = go.Figure()
    
    metrics = ["Coverage", "Collision Rate", "Energy Consumption"]
    colors = ["green", "red", "blue"]
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Algorithm"],
            y=df[metric],
            marker_color=color,
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Algorithm Performance Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Value",
        barmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
