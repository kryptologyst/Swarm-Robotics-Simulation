"""Base classes and interfaces for swarm robotics controllers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    """State of a single robot in the swarm."""
    
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: float
    angular_velocity: float
    id: int
    
    def __post_init__(self) -> None:
        """Validate robot state after initialization."""
        if len(self.position) != 2:
            raise ValueError("Position must be 2D")
        if len(self.velocity) != 2:
            raise ValueError("Velocity must be 2D")
        if len(self.acceleration) != 2:
            raise ValueError("Acceleration must be 2D")


@dataclass
class SwarmState:
    """State of the entire swarm."""
    
    robots: List[RobotState]
    time: float
    step: int
    
    def get_robot_by_id(self, robot_id: int) -> Optional[RobotState]:
        """Get robot state by ID."""
        for robot in self.robots:
            if robot.id == robot_id:
                return robot
        return None
    
    def get_neighbors(self, robot_id: int, radius: float) -> List[RobotState]:
        """Get neighboring robots within radius."""
        robot = self.get_robot_by_id(robot_id)
        if robot is None:
            return []
        
        neighbors = []
        for other in self.robots:
            if other.id != robot_id:
                distance = np.linalg.norm(robot.position - other.position)
                if distance <= radius:
                    neighbors.append(other)
        return neighbors


class SwarmController(ABC):
    """Abstract base class for swarm controllers."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize controller with configuration.
        
        Args:
            config: Controller configuration parameters
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute_control(
        self, 
        swarm_state: SwarmState, 
        robot_id: int,
        dt: float
    ) -> np.ndarray:
        """Compute control input for a specific robot.
        
        Args:
            swarm_state: Current state of the swarm
            robot_id: ID of the robot to control
            dt: Time step duration
            
        Returns:
            Control input (acceleration) for the robot
        """
        pass
    
    def update(self, swarm_state: SwarmState, dt: float) -> Dict[int, np.ndarray]:
        """Update all robots in the swarm.
        
        Args:
            swarm_state: Current state of the swarm
            dt: Time step duration
            
        Returns:
            Dictionary mapping robot IDs to control inputs
        """
        controls = {}
        for robot in swarm_state.robots:
            controls[robot.id] = self.compute_control(swarm_state, robot.id, dt)
        return controls
    
    def reset(self) -> None:
        """Reset controller state."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        return {
            "name": self.name,
            "config": self.config,
        }


class SafetyLimits:
    """Safety limits for robot motion."""
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 5.0,
        max_angular_velocity: float = 1.0,
        collision_radius: float = 0.5,
    ) -> None:
        """Initialize safety limits.
        
        Args:
            max_velocity: Maximum linear velocity (m/s)
            max_acceleration: Maximum linear acceleration (m/s²)
            max_angular_velocity: Maximum angular velocity (rad/s)
            collision_radius: Collision detection radius (m)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.collision_radius = collision_radius
    
    def apply_limits(self, velocity: np.ndarray, acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply safety limits to velocity and acceleration.
        
        Args:
            velocity: Current velocity
            acceleration: Desired acceleration
            
        Returns:
            Limited velocity and acceleration
        """
        # Limit acceleration
        accel_norm = np.linalg.norm(acceleration)
        if accel_norm > self.max_acceleration:
            acceleration = acceleration / accel_norm * self.max_acceleration
        
        # Limit velocity
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.max_velocity:
            velocity = velocity / vel_norm * self.max_velocity
        
        return velocity, acceleration
    
    def check_collision(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if two positions are in collision.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            True if collision detected
        """
        distance = np.linalg.norm(pos1 - pos2)
        return distance < 2 * self.collision_radius
