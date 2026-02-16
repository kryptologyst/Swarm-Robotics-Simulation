"""Consensus controller for distributed agreement in swarm robotics."""

from typing import Dict, Any, List, Optional
import numpy as np
from .base import SwarmController, SwarmState, RobotState, SafetyLimits


class ConsensusController(SwarmController):
    """Consensus controller for distributed agreement on position and velocity.
    
    Implements distributed consensus algorithms where robots reach agreement
    on their states through local communication with neighbors.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize consensus controller.
        
        Args:
            config: Configuration dictionary with parameters:
                - communication_range: Range for communication (default: 3.0)
                - consensus_gain: Consensus gain parameter (default: 0.5)
                - consensus_type: Type of consensus ('position', 'velocity', 'both')
                - max_velocity: Maximum velocity (default: 2.0)
                - max_acceleration: Maximum acceleration (default: 5.0)
                - target_formation: Target formation positions (optional)
        """
        super().__init__(config)
        
        # Consensus parameters
        self.communication_range = config.get("communication_range", 3.0)
        self.consensus_gain = config.get("consensus_gain", 0.5)
        self.consensus_type = config.get("consensus_type", "both")
        
        # Safety limits
        self.safety_limits = SafetyLimits(
            max_velocity=config.get("max_velocity", 2.0),
            max_acceleration=config.get("max_acceleration", 5.0),
        )
        
        # Target formation
        self.target_formation = config.get("target_formation", None)
        self.formation_weight = config.get("formation_weight", 0.1)
        
        # Consensus state tracking
        self.consensus_positions: Dict[int, np.ndarray] = {}
        self.consensus_velocities: Dict[int, np.ndarray] = {}
    
    def compute_control(
        self, 
        swarm_state: SwarmState, 
        robot_id: int,
        dt: float
    ) -> np.ndarray:
        """Compute consensus control for a specific robot.
        
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
        
        # Get neighbors within communication range
        neighbors = swarm_state.get_neighbors(robot_id, self.communication_range)
        
        if not neighbors:
            # No neighbors, maintain current velocity
            return np.zeros(2)
        
        # Initialize consensus states if not present
        if robot_id not in self.consensus_positions:
            self.consensus_positions[robot_id] = robot.position.copy()
        if robot_id not in self.consensus_velocities:
            self.consensus_velocities[robot_id] = robot.velocity.copy()
        
        # Compute consensus updates
        position_consensus = self._compute_position_consensus(robot_id, neighbors)
        velocity_consensus = self._compute_velocity_consensus(robot_id, neighbors)
        
        # Update consensus states
        if self.consensus_type in ["position", "both"]:
            self.consensus_positions[robot_id] += position_consensus * dt
        
        if self.consensus_type in ["velocity", "both"]:
            self.consensus_velocities[robot_id] += velocity_consensus * dt
        
        # Compute desired velocity based on consensus
        desired_velocity = self._compute_desired_velocity(robot_id, robot)
        
        # Add formation control if specified
        if self.target_formation is not None:
            formation_velocity = self._compute_formation_velocity(robot_id, robot)
            desired_velocity += self.formation_weight * formation_velocity
        
        # Convert to acceleration and apply limits
        return self._limit_acceleration(robot, desired_velocity, dt)
    
    def _compute_position_consensus(self, robot_id: int, neighbors: List[RobotState]) -> np.ndarray:
        """Compute position consensus update.
        
        Args:
            robot_id: ID of the current robot
            neighbors: List of neighboring robot states
            
        Returns:
            Position consensus update vector
        """
        if self.consensus_type not in ["position", "both"]:
            return np.zeros(2)
        
        consensus_update = np.zeros(2)
        current_pos = self.consensus_positions[robot_id]
        
        for neighbor in neighbors:
            neighbor_pos = self.consensus_positions.get(neighbor.id, neighbor.position)
            consensus_update += neighbor_pos - current_pos
        
        return self.consensus_gain * consensus_update
    
    def _compute_velocity_consensus(self, robot_id: int, neighbors: List[RobotState]) -> np.ndarray:
        """Compute velocity consensus update.
        
        Args:
            robot_id: ID of the current robot
            neighbors: List of neighboring robot states
            
        Returns:
            Velocity consensus update vector
        """
        if self.consensus_type not in ["velocity", "both"]:
            return np.zeros(2)
        
        consensus_update = np.zeros(2)
        current_vel = self.consensus_velocities[robot_id]
        
        for neighbor in neighbors:
            neighbor_vel = self.consensus_velocities.get(neighbor.id, neighbor.velocity)
            consensus_update += neighbor_vel - current_vel
        
        return self.consensus_gain * consensus_update
    
    def _compute_desired_velocity(self, robot_id: int, robot: RobotState) -> np.ndarray:
        """Compute desired velocity based on consensus state.
        
        Args:
            robot_id: ID of the current robot
            robot: Current robot state
            
        Returns:
            Desired velocity vector
        """
        if self.consensus_type == "position":
            # Move towards consensus position
            target_pos = self.consensus_positions[robot_id]
            desired_velocity = target_pos - robot.position
        elif self.consensus_type == "velocity":
            # Use consensus velocity
            desired_velocity = self.consensus_velocities[robot_id]
        else:  # both
            # Combine position and velocity consensus
            target_pos = self.consensus_positions[robot_id]
            position_velocity = target_pos - robot.position
            velocity_consensus = self.consensus_velocities[robot_id]
            desired_velocity = 0.5 * position_velocity + 0.5 * velocity_consensus
        
        # Normalize to maximum velocity
        if np.linalg.norm(desired_velocity) > 0:
            desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * self.safety_limits.max_velocity
        
        return desired_velocity
    
    def _compute_formation_velocity(self, robot_id: int, robot: RobotState) -> np.ndarray:
        """Compute velocity to maintain formation.
        
        Args:
            robot_id: ID of the current robot
            robot: Current robot state
            
        Returns:
            Formation velocity vector
        """
        if self.target_formation is None or robot_id >= len(self.target_formation):
            return np.zeros(2)
        
        target_pos = np.array(self.target_formation[robot_id])
        formation_vector = target_pos - robot.position
        
        # Normalize to maximum velocity
        if np.linalg.norm(formation_vector) > 0:
            formation_vector = formation_vector / np.linalg.norm(formation_vector) * self.safety_limits.max_velocity
        
        return formation_vector
    
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
    
    def set_formation(self, formation_positions: List[np.ndarray]) -> None:
        """Set target formation positions.
        
        Args:
            formation_positions: List of target positions for each robot
        """
        self.target_formation = [np.array(pos) for pos in formation_positions]
    
    def get_consensus_error(self, swarm_state: SwarmState) -> Dict[str, float]:
        """Compute consensus error metrics.
        
        Args:
            swarm_state: Current state of the swarm
            
        Returns:
            Dictionary with consensus error metrics
        """
        if not swarm_state.robots:
            return {"position_error": 0.0, "velocity_error": 0.0}
        
        positions = [robot.position for robot in swarm_state.robots]
        velocities = [robot.velocity for robot in swarm_state.robots]
        
        # Compute variance as consensus error
        position_variance = np.var(positions, axis=0).sum()
        velocity_variance = np.var(velocities, axis=0).sum()
        
        return {
            "position_error": float(position_variance),
            "velocity_error": float(velocity_variance),
        }
    
    def reset(self) -> None:
        """Reset controller state."""
        self.consensus_positions.clear()
        self.consensus_velocities.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        info = super().get_info()
        info.update({
            "communication_range": self.communication_range,
            "consensus_gain": self.consensus_gain,
            "consensus_type": self.consensus_type,
            "target_formation": self.target_formation,
        })
        return info
