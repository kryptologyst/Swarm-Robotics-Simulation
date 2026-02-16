"""Formation control for maintaining geometric formations in swarm robotics."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from .base import SwarmController, SwarmState, RobotState, SafetyLimits


class FormationController(SwarmController):
    """Formation controller for maintaining geometric formations.
    
    Supports various formation types including line, circle, diamond, and custom formations.
    Uses virtual structure approach with formation maintenance and collision avoidance.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize formation controller.
        
        Args:
            config: Configuration dictionary with parameters:
                - formation_type: Type of formation ('line', 'circle', 'diamond', 'custom')
                - formation_size: Size parameter for formation (default: 2.0)
                - formation_center: Center position of formation (default: [0, 0])
                - formation_angle: Rotation angle of formation (default: 0.0)
                - formation_positions: Custom formation positions (for 'custom' type)
                - formation_gain: Gain for formation control (default: 1.0)
                - collision_avoidance_gain: Gain for collision avoidance (default: 2.0)
                - max_velocity: Maximum velocity (default: 2.0)
                - max_acceleration: Maximum acceleration (default: 5.0)
        """
        super().__init__(config)
        
        # Formation parameters
        self.formation_type = config.get("formation_type", "circle")
        self.formation_size = config.get("formation_size", 2.0)
        self.formation_center = np.array(config.get("formation_center", [0.0, 0.0]))
        self.formation_angle = config.get("formation_angle", 0.0)
        self.formation_positions = config.get("formation_positions", None)
        
        # Control gains
        self.formation_gain = config.get("formation_gain", 1.0)
        self.collision_avoidance_gain = config.get("collision_avoidance_gain", 2.0)
        
        # Safety limits
        self.safety_limits = SafetyLimits(
            max_velocity=config.get("max_velocity", 2.0),
            max_acceleration=config.get("max_acceleration", 5.0),
        )
        
        # Formation assignment
        self.robot_assignments: Dict[int, int] = {}  # robot_id -> formation_index
    
    def compute_control(
        self, 
        swarm_state: SwarmState, 
        robot_id: int,
        dt: float
    ) -> np.ndarray:
        """Compute formation control for a specific robot.
        
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
        
        # Assign formation positions if not already assigned
        if robot_id not in self.robot_assignments:
            self._assign_formation_positions(swarm_state)
        
        # Get target formation position
        formation_index = self.robot_assignments.get(robot_id, 0)
        target_position = self._get_formation_position(formation_index, swarm_state)
        
        # Compute formation control
        formation_velocity = self._compute_formation_velocity(robot, target_position)
        
        # Add collision avoidance
        collision_velocity = self._compute_collision_avoidance(robot, swarm_state)
        
        # Combine control inputs
        desired_velocity = (
            self.formation_gain * formation_velocity +
            self.collision_avoidance_gain * collision_velocity
        )
        
        # Convert to acceleration and apply limits
        return self._limit_acceleration(robot, desired_velocity, dt)
    
    def _assign_formation_positions(self, swarm_state: SwarmState) -> None:
        """Assign formation positions to robots.
        
        Args:
            swarm_state: Current state of the swarm
        """
        num_robots = len(swarm_state.robots)
        
        if self.formation_type == "custom" and self.formation_positions is not None:
            # Use custom formation positions
            formation_positions = self.formation_positions
        else:
            # Generate formation positions
            formation_positions = self._generate_formation_positions(num_robots)
        
        # Assign positions to robots (closest assignment)
        robot_positions = [robot.position for robot in swarm_state.robots]
        distances = cdist(robot_positions, formation_positions)
        
        # Use Hungarian algorithm for optimal assignment (simplified greedy version)
        assigned_robots = set()
        assigned_formations = set()
        
        for _ in range(min(num_robots, len(formation_positions))):
            min_dist = float('inf')
            best_robot_idx = -1
            best_formation_idx = -1
            
            for robot_idx, robot in enumerate(swarm_state.robots):
                if robot.id in assigned_robots:
                    continue
                
                for formation_idx, formation_pos in enumerate(formation_positions):
                    if formation_idx in assigned_formations:
                        continue
                    
                    if distances[robot_idx, formation_idx] < min_dist:
                        min_dist = distances[robot_idx, formation_idx]
                        best_robot_idx = robot_idx
                        best_formation_idx = formation_idx
            
            if best_robot_idx >= 0 and best_formation_idx >= 0:
                robot_id = swarm_state.robots[best_robot_idx].id
                self.robot_assignments[robot_id] = best_formation_idx
                assigned_robots.add(robot_id)
                assigned_formations.add(best_formation_idx)
    
    def _generate_formation_positions(self, num_robots: int) -> List[np.ndarray]:
        """Generate formation positions based on formation type.
        
        Args:
            num_robots: Number of robots in the swarm
            
        Returns:
            List of formation positions
        """
        positions = []
        
        if self.formation_type == "line":
            # Line formation
            for i in range(num_robots):
                x = (i - (num_robots - 1) / 2) * self.formation_size / num_robots
                positions.append(np.array([x, 0.0]))
        
        elif self.formation_type == "circle":
            # Circle formation
            for i in range(num_robots):
                angle = 2 * np.pi * i / num_robots
                x = self.formation_size * np.cos(angle)
                y = self.formation_size * np.sin(angle)
                positions.append(np.array([x, y]))
        
        elif self.formation_type == "diamond":
            # Diamond formation
            if num_robots >= 4:
                positions = [
                    np.array([0.0, self.formation_size]),      # Top
                    np.array([self.formation_size, 0.0]),      # Right
                    np.array([0.0, -self.formation_size]),    # Bottom
                    np.array([-self.formation_size, 0.0]),     # Left
                ]
                # Add additional robots in the center
                for i in range(4, num_robots):
                    positions.append(np.array([0.0, 0.0]))
            else:
                # Fallback to circle for fewer robots
                for i in range(num_robots):
                    angle = 2 * np.pi * i / num_robots
                    x = self.formation_size * np.cos(angle)
                    y = self.formation_size * np.sin(angle)
                    positions.append(np.array([x, y]))
        
        else:
            # Default to circle formation
            for i in range(num_robots):
                angle = 2 * np.pi * i / num_robots
                x = self.formation_size * np.cos(angle)
                y = self.formation_size * np.sin(angle)
                positions.append(np.array([x, y]))
        
        return positions
    
    def _get_formation_position(self, formation_index: int, swarm_state: SwarmState) -> np.ndarray:
        """Get target formation position for a robot.
        
        Args:
            formation_index: Index in the formation
            swarm_state: Current state of the swarm
            
        Returns:
            Target formation position
        """
        if self.formation_type == "custom" and self.formation_positions is not None:
            if formation_index < len(self.formation_positions):
                base_position = np.array(self.formation_positions[formation_index])
            else:
                base_position = np.array([0.0, 0.0])
        else:
            formation_positions = self._generate_formation_positions(len(swarm_state.robots))
            if formation_index < len(formation_positions):
                base_position = formation_positions[formation_index]
            else:
                base_position = np.array([0.0, 0.0])
        
        # Apply rotation
        cos_angle = np.cos(self.formation_angle)
        sin_angle = np.sin(self.formation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_position = rotation_matrix @ base_position
        
        # Apply translation
        target_position = rotated_position + self.formation_center
        
        return target_position
    
    def _compute_formation_velocity(self, robot: RobotState, target_position: np.ndarray) -> np.ndarray:
        """Compute velocity to maintain formation.
        
        Args:
            robot: Current robot state
            target_position: Target formation position
            
        Returns:
            Formation velocity vector
        """
        position_error = target_position - robot.position
        formation_velocity = position_error
        
        # Normalize to maximum velocity
        if np.linalg.norm(formation_velocity) > 0:
            formation_velocity = formation_velocity / np.linalg.norm(formation_velocity) * self.safety_limits.max_velocity
        
        return formation_velocity
    
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
    
    def set_formation_center(self, center: np.ndarray) -> None:
        """Set formation center position.
        
        Args:
            center: New formation center position
        """
        self.formation_center = np.array(center)
    
    def set_formation_angle(self, angle: float) -> None:
        """Set formation rotation angle.
        
        Args:
            angle: Rotation angle in radians
        """
        self.formation_angle = angle
    
    def get_formation_error(self, swarm_state: SwarmState) -> float:
        """Compute formation error.
        
        Args:
            swarm_state: Current state of the swarm
            
        Returns:
            Average formation error
        """
        if not swarm_state.robots:
            return 0.0
        
        total_error = 0.0
        count = 0
        
        for robot in swarm_state.robots:
            if robot.id in self.robot_assignments:
                formation_index = self.robot_assignments[robot.id]
                target_position = self._get_formation_position(formation_index, swarm_state)
                error = np.linalg.norm(robot.position - target_position)
                total_error += error
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def reset(self) -> None:
        """Reset controller state."""
        self.robot_assignments.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        info = super().get_info()
        info.update({
            "formation_type": self.formation_type,
            "formation_size": self.formation_size,
            "formation_center": self.formation_center.tolist(),
            "formation_angle": self.formation_angle,
            "formation_gain": self.formation_gain,
            "collision_avoidance_gain": self.collision_avoidance_gain,
        })
        return info
