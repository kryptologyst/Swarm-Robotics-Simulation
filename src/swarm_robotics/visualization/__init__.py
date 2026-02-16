"""Visualization framework for swarm robotics."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import logging

from ..simulation import SimulationResults, SwarmState
from ..evaluation import Metrics


class SwarmVisualizer:
    """Main visualizer for swarm robotics simulations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize swarm visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Visualization parameters
        self.figsize = self.config.get("figsize", (10, 8))
        self.dpi = self.config.get("dpi", 100)
        self.robot_radius = self.config.get("robot_radius", 0.5)
        self.robot_colors = self.config.get("robot_colors", None)
        self.trajectory_length = self.config.get("trajectory_length", 50)
        
        # Animation parameters
        self.animation_interval = self.config.get("animation_interval", 50)
        self.save_animation = self.config.get("save_animation", False)
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
    
    def plot_trajectories(self, results: SimulationResults, save_path: Optional[str] = None) -> plt.Figure:
        """Plot robot trajectories.
        
        Args:
            results: Simulation results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not results.positions:
            self.logger.warning("No trajectory data to plot")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot trajectories for each robot
        num_robots = len(results.positions[0])
        colors = self._get_robot_colors(num_robots)
        
        for robot_id in range(num_robots):
            trajectory = [step[robot_id] for step in results.positions]
            trajectory = np.array(trajectory)
            
            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=colors[robot_id], alpha=0.7, linewidth=1)
            
            # Mark start and end positions
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      color=colors[robot_id], marker='o', s=50, label=f'Robot {robot_id} Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      color=colors[robot_id], marker='s', s=50, label=f'Robot {robot_id} End')
        
        # Setup plot
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Trajectories')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved trajectory plot to {save_path}")
        
        return fig
    
    def plot_metrics(self, metrics: Metrics, save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance metrics.
        
        Args:
            metrics: Performance metrics
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Coverage metrics
        coverage_data = [metrics.coverage_area, metrics.coverage_efficiency]
        coverage_labels = ['Coverage Area', 'Coverage Efficiency']
        axes[0].bar(coverage_labels, coverage_data, color=['skyblue', 'lightgreen'])
        axes[0].set_title('Coverage Metrics')
        axes[0].set_ylabel('Value')
        
        # Formation metrics
        formation_data = [metrics.formation_error, metrics.formation_stability]
        formation_labels = ['Formation Error', 'Formation Stability']
        axes[1].bar(formation_labels, formation_data, color=['lightcoral', 'lightblue'])
        axes[1].set_title('Formation Metrics')
        axes[1].set_ylabel('Value')
        
        # Collision metrics
        collision_data = [metrics.collision_rate, metrics.collision_count]
        collision_labels = ['Collision Rate', 'Collision Count']
        axes[2].bar(collision_labels, collision_data, color=['red', 'darkred'])
        axes[2].set_title('Collision Metrics')
        axes[2].set_ylabel('Value')
        
        # Energy metrics
        energy_data = [metrics.energy_consumption, metrics.energy_efficiency]
        energy_labels = ['Energy Consumption', 'Energy Efficiency']
        axes[3].bar(energy_labels, energy_data, color=['orange', 'gold'])
        axes[3].set_title('Energy Metrics')
        axes[3].set_ylabel('Value')
        
        # Stability metrics
        stability_data = [metrics.stability_margin, metrics.convergence_rate]
        stability_labels = ['Stability Margin', 'Convergence Rate']
        axes[4].bar(stability_labels, stability_data, color=['purple', 'pink'])
        axes[4].set_title('Stability Metrics')
        axes[4].set_ylabel('Value')
        
        # Task metrics
        task_data = [metrics.task_completion_time, metrics.task_success_rate]
        task_labels = ['Completion Time', 'Success Rate']
        axes[5].bar(task_labels, task_data, color=['green', 'lightgreen'])
        axes[5].set_title('Task Metrics')
        axes[5].set_ylabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved metrics plot to {save_path}")
        
        return fig
    
    def create_animation(self, results: SimulationResults, save_path: Optional[str] = None) -> animation.FuncAnimation:
        """Create animation of swarm movement.
        
        Args:
            results: Simulation results
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation
        """
        if not results.positions:
            self.logger.warning("No trajectory data for animation")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Setup plot
        ax.set_xlim(0, 50)  # Default world size
        ax.set_ylim(0, 50)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Swarm Animation')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Initialize robot circles
        num_robots = len(results.positions[0])
        colors = self._get_robot_colors(num_robots)
        robots = []
        
        for i in range(num_robots):
            circle = Circle((0, 0), self.robot_radius, 
                          color=colors[i], alpha=0.8)
            ax.add_patch(circle)
            robots.append(circle)
        
        # Initialize trajectory lines
        trajectories = []
        for i in range(num_robots):
            line, = ax.plot([], [], color=colors[i], alpha=0.5, linewidth=1)
            trajectories.append(line)
        
        def animate(frame):
            """Animation function."""
            if frame >= len(results.positions):
                return robots + trajectories
            
            positions = results.positions[frame]
            
            # Update robot positions
            for i, (robot, pos) in enumerate(zip(robots, positions)):
                robot.center = (pos[0], pos[1])
            
            # Update trajectories
            for i, (trajectory, pos) in enumerate(zip(trajectories, positions)):
                # Get recent trajectory
                start_frame = max(0, frame - self.trajectory_length)
                recent_positions = results.positions[start_frame:frame+1]
                recent_trajectory = [step[i] for step in recent_positions]
                
                if recent_trajectory:
                    trajectory.set_data([p[0] for p in recent_trajectory],
                                      [p[1] for p in recent_trajectory])
            
            return robots + trajectories
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(results.positions),
            interval=self.animation_interval, blit=True, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            self.logger.info(f"Saved animation to {save_path}")
        
        return anim
    
    def plot_comparison(self, results_dict: Dict[str, SimulationResults], 
                       metric: str = "coverage_efficiency", 
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of different algorithms.
        
        Args:
            results_dict: Dictionary mapping algorithm names to results
            metric: Metric to compare
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        algorithms = list(results_dict.keys())
        values = []
        
        for algorithm, results in results_dict.items():
            # Extract metric value (simplified)
            if metric == "coverage_efficiency":
                value = results.coverage
            elif metric == "collision_rate":
                value = results.collision_rate
            elif metric == "energy_consumption":
                value = results.energy_consumption
            else:
                value = 0.0
            
            values.append(value)
        
        # Create bar plot
        bars = ax.bar(algorithms, values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def create_interactive_plot(self, results: SimulationResults) -> go.Figure:
        """Create interactive plotly visualization.
        
        Args:
            results: Simulation results
            
        Returns:
            Plotly figure
        """
        if not results.positions:
            self.logger.warning("No trajectory data for interactive plot")
            return None
        
        fig = go.Figure()
        
        # Add trajectories for each robot
        num_robots = len(results.positions[0])
        colors = self._get_robot_colors(num_robots)
        
        for robot_id in range(num_robots):
            trajectory = [step[robot_id] for step in results.positions]
            trajectory = np.array(trajectory)
            
            # Add trajectory line
            fig.add_trace(go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode='lines',
                name=f'Robot {robot_id}',
                line=dict(color=colors[robot_id], width=2),
                opacity=0.7
            ))
            
            # Add start and end markers
            fig.add_trace(go.Scatter(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                mode='markers',
                name=f'Robot {robot_id} Start',
                marker=dict(color=colors[robot_id], size=10, symbol='circle'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                mode='markers',
                name=f'Robot {robot_id} End',
                marker=dict(color=colors[robot_id], size=10, symbol='square'),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='Swarm Trajectories (Interactive)',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            hovermode='closest',
            width=800,
            height=600
        )
        
        return fig
    
    def plot_metrics_over_time(self, results: SimulationResults, 
                              metrics_over_time: List[Metrics],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot metrics evolution over time.
        
        Args:
            results: Simulation results
            metrics_over_time: List of metrics at each time step
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not metrics_over_time:
            self.logger.warning("No metrics over time data")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = results.times[:len(metrics_over_time)]
        
        # Coverage over time
        coverage = [m.coverage_efficiency for m in metrics_over_time]
        axes[0, 0].plot(times, coverage, 'b-', linewidth=2)
        axes[0, 0].set_title('Coverage Efficiency Over Time')
        axes[0, 0].set_ylabel('Coverage Efficiency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Formation error over time
        formation_error = [m.formation_error for m in metrics_over_time]
        axes[0, 1].plot(times, formation_error, 'r-', linewidth=2)
        axes[0, 1].set_title('Formation Error Over Time')
        axes[0, 1].set_ylabel('Formation Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Collision rate over time
        collision_rate = [m.collision_rate for m in metrics_over_time]
        axes[1, 0].plot(times, collision_rate, 'g-', linewidth=2)
        axes[1, 0].set_title('Collision Rate Over Time')
        axes[1, 0].set_ylabel('Collision Rate')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy consumption over time
        energy = [m.energy_consumption for m in metrics_over_time]
        axes[1, 1].plot(times, energy, 'm-', linewidth=2)
        axes[1, 1].set_title('Energy Consumption Over Time')
        axes[1, 1].set_ylabel('Energy Consumption')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved metrics over time plot to {save_path}")
        
        return fig
    
    def _get_robot_colors(self, num_robots: int) -> List[str]:
        """Get colors for robots.
        
        Args:
            num_robots: Number of robots
            
        Returns:
            List of colors
        """
        if self.robot_colors and len(self.robot_colors) >= num_robots:
            return self.robot_colors[:num_robots]
        
        # Generate colors using colormap
        cmap = plt.cm.tab20
        colors = [cmap(i / num_robots) for i in range(num_robots)]
        return colors
    
    def save_all_plots(self, results: SimulationResults, metrics: Metrics, 
                      output_dir: Union[str, Path]) -> None:
        """Save all visualization plots.
        
        Args:
            results: Simulation results
            metrics: Performance metrics
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory plot
        self.plot_trajectories(results, str(output_dir / "trajectories.png"))
        
        # Save metrics plot
        self.plot_metrics(metrics, str(output_dir / "metrics.png"))
        
        # Save animation
        self.create_animation(results, str(output_dir / "animation.gif"))
        
        # Save interactive plot
        interactive_fig = self.create_interactive_plot(results)
        if interactive_fig:
            interactive_fig.write_html(str(output_dir / "interactive_plot.html"))
        
        self.logger.info(f"Saved all plots to {output_dir}")
    
    def create_dashboard(self, results: SimulationResults, metrics: Metrics) -> go.Figure:
        """Create comprehensive dashboard.
        
        Args:
            results: Simulation results
            metrics: Performance metrics
            
        Returns:
            Plotly dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trajectories', 'Coverage Metrics', 
                          'Performance Metrics', 'Energy Consumption'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add trajectory plot
        if results.positions:
            num_robots = len(results.positions[0])
            colors = self._get_robot_colors(num_robots)
            
            for robot_id in range(num_robots):
                trajectory = [step[robot_id] for step in results.positions]
                trajectory = np.array(trajectory)
                
                fig.add_trace(
                    go.Scatter(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        mode='lines',
                        name=f'Robot {robot_id}',
                        line=dict(color=colors[robot_id]),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add coverage metrics
        coverage_data = [metrics.coverage_area, metrics.coverage_efficiency]
        coverage_labels = ['Coverage Area', 'Coverage Efficiency']
        
        fig.add_trace(
            go.Bar(x=coverage_labels, y=coverage_data, name='Coverage'),
            row=1, col=2
        )
        
        # Add performance metrics
        perf_data = [metrics.formation_error, metrics.collision_rate]
        perf_labels = ['Formation Error', 'Collision Rate']
        
        fig.add_trace(
            go.Bar(x=perf_labels, y=perf_data, name='Performance'),
            row=2, col=1
        )
        
        # Add energy consumption over time
        if results.times:
            energy_data = [0] * len(results.times)  # Simplified
            fig.add_trace(
                go.Scatter(x=results.times, y=energy_data, name='Energy'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Swarm Robotics Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
