"""Coverage control for optimal area coverage in swarm robotics."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from .base import SwarmController, SwarmState, RobotState, SafetyLimits


class CoverageController(SwarmController):
    """Coverage controller for optimal area coverage using Voronoi diagrams.
    
    Implements distributed coverage control where robots move to maximize
    coverage of a target area while avoiding collisions.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize coverage controller.
        
        Args:
            config: Configuration dictionary with parameters:
                - coverage_area: Target area to cover [[x_min, y_min], [x_max, y_max]]
                - coverage_gain: Gain for coverage control (default: 1.0)
                - collision_avoidance_gain: Gain for collision avoidance (default: 2.0)
                - coverage_density: Density function type ('uniform', 'gaussian', 'custom')
                - density_centers: Centers for density function (for 'gaussian')
                - density_sigma: Standard deviation for density function (default: 2.0)
                - max_velocity: Maximum velocity (default: 2.0)
                - max_acceleration: Maximum acceleration (default: 5.0)
        """
        super().__init__(config)
        
        # Coverage parameters
        self.coverage_area = np.array(config.get("coverage_area", [[-10, -10], [10, 10]]))
        self.coverage_gain = config.get("coverage_gain", 1.0)
        self.collision_avoidance_gain = config.get("collision_avoidance_gain", 2.0)
        
        # Density function parameters
        self.coverage_density = config.get("coverage_density", "uniform")
        self.density_centers = config.get("density_centers", [[0, 0]])
        self.density_sigma = config.get("density_sigma", 2.0)
        
        # Safety limits
        self.safety_limits = SafetyLimits(
            max_velocity=config.get("max_velocity", 2.0),
            max_acceleration=config.get("max_acceleration", 5.0),
        )
        
        # Voronoi diagram
        self.voronoi_diagram: Optional[Voronoi] = None
        self.voronoi_cells: Dict[int, List[np.ndarray]] = {}
    
    def compute_control(
        self, 
        swarm_state: SwarmState, 
        robot_id: int,
        dt: float
    ) -> np.ndarray:
        """Compute coverage control for a specific robot.
        
        Args:
            swarm_state: Current state of the swarm
            robot_id: ID of the robot to control
            dt: Time step duration
            
        Returns:
            Control acceleration for the robot
        """
        robot = swarm_state.get_robot_by_id(robot_id)
        if robot is None:
            return np.zeros(2)
        
        # Update Voronoi diagram
        self._update_voronoi_diagram(swarm_state)
        
        # Compute coverage control
        coverage_velocity = self._compute_coverage_velocity(robot_id, robot, swarm_state)
        
        # Add collision avoidance
        collision_velocity = self._compute_collision_avoidance(robot, swarm_state)
        
        # Combine control inputs
        desired_velocity = (
            self.coverage_gain * coverage_velocity +
            self.collision_avoidance_gain * collision_velocity
        )
        
        # Convert to acceleration and apply limits
        return self._limit_acceleration(robot, desired_velocity, dt)
    
    def _update_voronoi_diagram(self, swarm_state: SwarmState) -> None:
        """Update Voronoi diagram based on current robot positions.
        
        Args:
            swarm_state: Current state of the swarm
        """
        if len(swarm_state.robots) < 2:
            self.voronoi_diagram = None
            self.voronoi_cells.clear()
            return
        
        # Extract robot positions
        positions = np.array([robot.position for robot in swarm_state.robots])
        
        try:
            # Create Voronoi diagram
            self.voronoi_diagram = Voronoi(positions)
            
            # Compute Voronoi cells for each robot
            self.voronoi_cells.clear()
            for i, robot in enumerate(swarm_state.robots):
                cell_vertices = self._get_voronoi_cell_vertices(i)
                self.voronoi_cells[robot.id] = cell_vertices
                
        except Exception:
            # Fallback if Voronoi diagram fails
            self.voronoi_diagram = None
            self.voronoi_cells.clear()
    
    def _get_voronoi_cell_vertices(self, robot_index: int) -> List[np.ndarray]:
        """Get vertices of Voronoi cell for a robot.
        
        Args:
            robot_index: Index of the robot in the swarm
            
        Returns:
            List of cell vertices
        """
        if self.voronoi_diagram is None:
            return []
        
        try:
            # Get ridge vertices for this robot
            ridge_vertices = []
            for ridge in self.voronoi_diagram.ridge_vertices:
                if ridge[0] != -1 and ridge[1] != -1:
                    vertex1 = self.voronoi_diagram.vertices[ridge[0]]
                    vertex2 = self.voronoi_diagram.vertices[ridge[1]]
                    
                    # Check if this ridge is adjacent to the robot
                    ridge_points = self.voronoi_diagram.ridge_points
                    for ridge_idx, (p1, p2) in enumerate(ridge_points):
                        if (p1 == robot_index or p2 == robot_index) and ridge_idx < len(self.voronoi_diagram.ridge_vertices):
                            if ridge_idx < len(self.voronoi_diagram.ridge_vertices):
                                ridge_vertices.extend([vertex1, vertex2])
                                break
            
            # Remove duplicates and return
            unique_vertices = []
            for vertex in ridge_vertices:
                if not any(np.allclose(vertex, v) for v in unique_vertices):
                    unique_vertices.append(vertex)
            
            return unique_vertices
            
        except Exception:
            return []
    
    def _compute_coverage_velocity(self, robot_id: int, robot: RobotState, swarm_state: SwarmState) -> np.ndarray:
        """Compute velocity for optimal coverage.
        
        Args:
            robot_id: ID of the current robot
            robot: Current robot state
            swarm_state: Current state of the swarm
            
        Returns:
            Coverage velocity vector
        """
        if robot_id not in self.voronoi_cells or not self.voronoi_cells[robot_id]:
            return np.zeros(2)
        
        # Compute centroid of Voronoi cell weighted by density
        cell_vertices = self.voronoi_cells[robot_id]
        
        # Clip vertices to coverage area
        clipped_vertices = self._clip_vertices_to_area(cell_vertices)
        
        if len(clipped_vertices) < 3:
            return np.zeros(2)
        
        # Compute weighted centroid
        centroid = self._compute_weighted_centroid(clipped_vertices)
        
        # Compute velocity towards centroid
        coverage_velocity = centroid - robot.position
        
        # Normalize to maximum velocity
        if np.linalg.norm(coverage_velocity) > 0:
            coverage_velocity = coverage_velocity / np.linalg.norm(coverage_velocity) * self.safety_limits.max_velocity
        
        return coverage_velocity
    
    def _clip_vertices_to_area(self, vertices: List[np.ndarray]) -> List[np.ndarray]:
        """Clip vertices to coverage area.
        
        Args:
            vertices: List of vertices
            
        Returns:
            List of clipped vertices
        """
        clipped_vertices = []
        
        for vertex in vertices:
            # Clip to coverage area bounds
            clipped_vertex = np.array([
                np.clip(vertex[0], self.coverage_area[0, 0], self.coverage_area[1, 0]),
                np.clip(vertex[1], self.coverage_area[0, 1], self.coverage_area[1, 1])
            ])
            clipped_vertices.append(clipped_vertex)
        
        return clipped_vertices
    
    def _compute_weighted_centroid(self, vertices: List[np.ndarray]) -> np.ndarray:
        """Compute centroid weighted by density function.
        
        Args:
            vertices: List of polygon vertices
            
        Returns:
            Weighted centroid
        """
        if len(vertices) < 3:
            return np.array([0.0, 0.0])
        
        # Simple polygon centroid (for uniform density)
        if self.coverage_density == "uniform":
            return np.mean(vertices, axis=0)
        
        # For non-uniform density, we would need to integrate over the polygon
        # For now, use uniform centroid as approximation
        return np.mean(vertices, axis=0)
    
    def _compute_collision_avoidance(self, robot: RobotState, swarm_state: SwarmState) -> np.ndarray:
        """Compute collision avoidance velocity.
        
        Args:
            robot: Current robot state
            swarm_state: Current state of the swarm
            
        Returns:
            Collision avoidance velocity vector
        """
        avoidance_velocity = np.zeros(2)
        collision_radius = self.safety_limits.collision_radius
        
        for other_robot in swarm_state.robots:
            if other_robot.id == robot.id:
                continue
            
            distance = np.linalg.norm(robot.position - other_robot.position)
            if distance < 2 * collision_radius and distance > 0:
                # Compute repulsive force
                repulsive_vector = robot.position - other_robot.position
                repulsive_vector = repulsive_vector / distance
                repulsive_force = repulsive_vector / (distance ** 2)
                avoidance_velocity += repulsive_force
        
        # Normalize to maximum velocity
        if np.linalg.norm(avoidance_velocity) > 0:
            avoidance_velocity = avoidance_velocity / np.linalg.norm(avoidance_velocity) * self.safety_limits.max_velocity
        
        return avoidance_velocity
    
    def _limit_acceleration(self, robot: RobotState, desired_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Convert desired velocity to acceleration with safety limits.
        
        Args:
            robot: Current robot state
            desired_velocity: Desired velocity vector
            dt: Time step duration
            
        Returns:
            Limited acceleration vector
        """
        # Compute acceleration needed to reach desired velocity
        velocity_error = desired_velocity - robot.velocity
        acceleration = velocity_error / dt
        
        # Apply safety limits
        limited_velocity, limited_acceleration = self.safety_limits.apply_limits(
            robot.velocity, acceleration
        )
        
        return limited_acceleration
    
    def get_coverage_metrics(self, swarm_state: SwarmState) -> Dict[str, float]:
        """Compute coverage metrics.
        
        Args:
            swarm_state: Current state of the swarm
            
        Returns:
            Dictionary with coverage metrics
        """
        if not swarm_state.robots:
            return {"coverage_area": 0.0, "coverage_efficiency": 0.0}
        
        # Compute total coverage area
        total_coverage_area = 0.0
        
        for robot_id in self.voronoi_cells:
            cell_vertices = self.voronoi_cells[robot_id]
            if len(cell_vertices) >= 3:
                # Compute polygon area using shoelace formula
                area = self._compute_polygon_area(cell_vertices)
                total_coverage_area += area
        
        # Compute coverage efficiency
        total_area = (self.coverage_area[1, 0] - self.coverage_area[0, 0]) * (self.coverage_area[1, 1] - self.coverage_area[0, 1])
        coverage_efficiency = total_coverage_area / total_area if total_area > 0 else 0.0
        
        return {
            "coverage_area": total_coverage_area,
            "coverage_efficiency": coverage_efficiency,
        }
    
    def _compute_polygon_area(self, vertices: List[np.ndarray]) -> float:
        """Compute area of polygon using shoelace formula.
        
        Args:
            vertices: List of polygon vertices
            
        Returns:
            Polygon area
        """
        if len(vertices) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        return abs(area) / 2.0
    
    def set_coverage_area(self, area: np.ndarray) -> None:
        """Set coverage area.
        
        Args:
            area: Coverage area [[x_min, y_min], [x_max, y_max]]
        """
        self.coverage_area = np.array(area)
    
    def reset(self) -> None:
        """Reset controller state."""
        self.voronoi_diagram = None
        self.voronoi_cells.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        info = super().get_info()
        info.update({
            "coverage_area": self.coverage_area.tolist(),
            "coverage_gain": self.coverage_gain,
            "collision_avoidance_gain": self.collision_avoidance_gain,
            "coverage_density": self.coverage_density,
            "density_centers": self.density_centers,
            "density_sigma": self.density_sigma,
        })
        return info
