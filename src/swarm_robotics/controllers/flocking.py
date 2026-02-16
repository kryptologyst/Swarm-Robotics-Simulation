"""Flocking controller implementation based on Reynolds' boids algorithm."""

from typing import Dict, Any, List
import numpy as np
from .base import SwarmController, SwarmState, RobotState, SafetyLimits


class FlockingController(SwarmController):
    """Flocking controller implementing Reynolds' boids algorithm.
    
    The flocking behavior is based on three simple rules:
    1. Alignment: Steer towards the average heading of neighbors
    2. Cohesion: Steer towards the average position of neighbors  
    3. Separation: Avoid crowding neighbors
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize flocking controller.
        
        Args:
            config: Configuration dictionary with parameters:
                - perception_range: Range for neighbor detection (default: 2.0)
                - alignment_weight: Weight for alignment behavior (default: 0.1)
                - cohesion_weight: Weight for cohesion behavior (default: 0.1)
                - separation_weight: Weight for separation behavior (default: 0.2)
                - max_velocity: Maximum velocity (default: 2.0)
                - max_acceleration: Maximum acceleration (default: 5.0)
        """
        super().__init__(config)
        
        # Flocking parameters
        self.perception_range = config.get("perception_range", 2.0)
        self.alignment_weight = config.get("alignment_weight", 0.1)
        self.cohesion_weight = config.get("cohesion_weight", 0.1)
        self.separation_weight = config.get("separation_weight", 0.2)
        
        # Safety limits
        self.safety_limits = SafetyLimits(
            max_velocity=config.get("max_velocity", 2.0),
            max_acceleration=config.get("max_acceleration", 5.0),
        )
        
        # Optional target following
        self.target_position = config.get("target_position", None)
        self.target_weight = config.get("target_weight", 0.0)
    
    def compute_control(
        self, 
        swarm_state: SwarmState, 
        robot_id: int,
        dt: float
    ) -> np.ndarray:
        """Compute flocking control for a specific robot.
        
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
        
        # Get neighbors within perception range
        neighbors = swarm_state.get_neighbors(robot_id, self.perception_range)
        
        if not neighbors:
            # No neighbors, maintain current velocity or move towards target
            if self.target_position is not None:
                target_velocity = self._compute_target_velocity(robot)
                return self._limit_acceleration(robot, target_velocity, dt)
            return np.zeros(2)
        
        # Compute flocking behaviors
        alignment = self._compute_alignment(robot, neighbors)
        cohesion = self._compute_cohesion(robot, neighbors)
        separation = self._compute_separation(robot, neighbors)
        
        # Combine behaviors
        desired_velocity = (
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion +
            self.separation_weight * separation
        )
        
        # Add target following if specified
        if self.target_position is not None:
            target_velocity = self._compute_target_velocity(robot)
            desired_velocity += self.target_weight * target_velocity
        
        # Convert to acceleration and apply limits
        return self._limit_acceleration(robot, desired_velocity, dt)
    
    def _compute_alignment(self, robot: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        """Compute alignment behavior.
        
        Args:
            robot: Current robot state
            neighbors: List of neighboring robot states
            
        Returns:
            Alignment velocity vector
        """
        if not neighbors:
            return np.zeros(2)
        
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_velocity
    
    def _compute_cohesion(self, robot: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        """Compute cohesion behavior.
        
        Args:
            robot: Current robot state
            neighbors: List of neighboring robot states
            
        Returns:
            Cohesion velocity vector
        """
        if not neighbors:
            return np.zeros(2)
        
        center_of_mass = np.mean([n.position for n in neighbors], axis=0)
        cohesion_vector = center_of_mass - robot.position
        
        # Normalize to desired speed
        if np.linalg.norm(cohesion_vector) > 0:
            cohesion_vector = cohesion_vector / np.linalg.norm(cohesion_vector) * self.safety_limits.max_velocity
        
        return cohesion_vector
    
    def _compute_separation(self, robot: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        """Compute separation behavior.
        
        Args:
            robot: Current robot state
            neighbors: List of neighboring robot states
            
        Returns:
            Separation velocity vector
        """
        if not neighbors:
            return np.zeros(2)
        
        separation_vector = np.zeros(2)
        
        for neighbor in neighbors:
            distance = np.linalg.norm(robot.position - neighbor.position)
            if distance > 0:
                # Avoid collision - steer away from neighbor
                avoid_vector = robot.position - neighbor.position
                avoid_vector = avoid_vector / distance  # Normalize
                avoid_vector = avoid_vector / distance  # Weight by inverse distance
                separation_vector += avoid_vector
        
        # Normalize to desired speed
        if np.linalg.norm(separation_vector) > 0:
            separation_vector = separation_vector / np.linalg.norm(separation_vector) * self.safety_limits.max_velocity
        
        return separation_vector
    
    def _compute_target_velocity(self, robot: RobotState) -> np.ndarray:
        """Compute velocity towards target position.
        
        Args:
            robot: Current robot state
            
        Returns:
            Target velocity vector
        """
        if self.target_position is None:
            return np.zeros(2)
        
        target_vector = np.array(self.target_position) - robot.position
        distance = np.linalg.norm(target_vector)
        
        if distance > 0:
            target_vector = target_vector / distance * self.safety_limits.max_velocity
        
        return target_vector
    
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
    
    def set_target(self, target_position: np.ndarray) -> None:
        """Set target position for the flock.
        
        Args:
            target_position: Target position [x, y]
        """
        self.target_position = np.array(target_position)
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        info = super().get_info()
        info.update({
            "perception_range": self.perception_range,
            "alignment_weight": self.alignment_weight,
            "cohesion_weight": self.cohesion_weight,
            "separation_weight": self.separation_weight,
            "target_position": self.target_position,
        })
        return info
