"""Core simulation framework for swarm robotics."""

from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import time
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import logging

from .controllers.base import SwarmController, SwarmState, RobotState, SafetyLimits
from .controllers.flocking import FlockingController
from .controllers.consensus import ConsensusController
from .controllers.formation import FormationController
from .controllers.coverage import CoverageController


@dataclass
class SimulationConfig:
    """Configuration for swarm simulation."""
    
    # Simulation parameters
    dt: float = 0.01
    max_steps: int = 10000
    real_time: bool = False
    
    # World parameters
    world_size: List[float] = field(default_factory=lambda: [50.0, 50.0])
    boundaries: str = "reflective"  # reflective, periodic, absorbing
    
    # Robot parameters
    num_robots: int = 20
    robot_radius: float = 0.5
    max_velocity: float = 2.0
    max_acceleration: float = 5.0
    
    # Controller parameters
    controller_type: str = "flocking"
    controller_config: Dict[str, Any] = field(default_factory=dict)
    
    # Safety parameters
    collision_detection: bool = True
    emergency_stop: bool = False
    
    # Logging parameters
    log_level: str = "INFO"
    save_trajectories: bool = True
    save_frequency: int = 10


@dataclass
class SimulationResults:
    """Results from swarm simulation."""
    
    # Trajectory data
    positions: List[List[np.ndarray]] = field(default_factory=list)
    velocities: List[List[np.ndarray]] = field(default_factory=list)
    accelerations: List[List[np.ndarray]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    
    # Performance metrics
    coverage: float = 0.0
    collision_rate: float = 0.0
    formation_error: float = 0.0
    energy_consumption: float = 0.0
    convergence_time: float = 0.0
    
    # Simulation info
    total_steps: int = 0
    total_time: float = 0.0
    controller_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_robot_trajectory(self, robot_id: int) -> List[np.ndarray]:
        """Get trajectory for a specific robot."""
        if robot_id < len(self.positions[0]):
            return [step[robot_id] for step in self.positions]
        return []


class SwarmSimulation:
    """Main simulation class for swarm robotics."""
    
    def __init__(self, config: Optional[Union[SimulationConfig, Dict[str, Any]]] = None):
        """Initialize swarm simulation.
        
        Args:
            config: Simulation configuration
        """
        if isinstance(config, dict):
            self.config = SimulationConfig(**config)
        elif config is None:
            self.config = SimulationConfig()
        else:
            self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize simulation state
        self.robots: List[RobotState] = []
        self.controller: Optional[SwarmController] = None
        self.safety_limits: SafetyLimits = SafetyLimits(
            max_velocity=self.config.max_velocity,
            max_acceleration=self.config.max_acceleration,
            collision_radius=self.config.robot_radius,
        )
        
        # Simulation state
        self.current_time = 0.0
        self.current_step = 0
        self.is_running = False
        self.emergency_stop = False
        
        # Results storage
        self.results = SimulationResults()
        
        # Initialize robots and controller
        self._initialize_robots()
        self._initialize_controller()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_robots(self) -> None:
        """Initialize robot states."""
        self.robots.clear()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for i in range(self.config.num_robots):
            # Random initial position within world bounds
            position = np.random.uniform(
                [self.config.robot_radius, self.config.robot_radius],
                [self.config.world_size[0] - self.config.robot_radius, 
                 self.config.world_size[1] - self.config.robot_radius]
            )
            
            # Random initial velocity
            velocity = np.random.uniform(-0.5, 0.5, 2)
            
            # Create robot state
            robot = RobotState(
                position=position,
                velocity=velocity,
                acceleration=np.zeros(2),
                orientation=0.0,
                angular_velocity=0.0,
                id=i
            )
            
            self.robots.append(robot)
        
        self.logger.info(f"Initialized {len(self.robots)} robots")
    
    def _initialize_controller(self) -> None:
        """Initialize swarm controller."""
        controller_config = self.config.controller_config.copy()
        controller_config.update({
            "max_velocity": self.config.max_velocity,
            "max_acceleration": self.config.max_acceleration,
        })
        
        if self.config.controller_type == "flocking":
            self.controller = FlockingController(controller_config)
        elif self.config.controller_type == "consensus":
            self.controller = ConsensusController(controller_config)
        elif self.config.controller_type == "formation":
            self.controller = FormationController(controller_config)
        elif self.config.controller_type == "coverage":
            self.controller = CoverageController(controller_config)
        else:
            raise ValueError(f"Unknown controller type: {self.config.controller_type}")
        
        self.logger.info(f"Initialized {self.config.controller_type} controller")
    
    def run(self, steps: Optional[int] = None, dt: Optional[float] = None) -> SimulationResults:
        """Run the simulation.
        
        Args:
            steps: Number of simulation steps (overrides config)
            dt: Time step duration (overrides config)
            
        Returns:
            Simulation results
        """
        steps = steps or self.config.max_steps
        dt = dt or self.config.dt
        
        self.logger.info(f"Starting simulation for {steps} steps with dt={dt}")
        
        # Reset results
        self.results = SimulationResults()
        self.results.controller_info = self.controller.get_info() if self.controller else {}
        
        # Reset simulation state
        self.current_time = 0.0
        self.current_step = 0
        self.is_running = True
        self.emergency_stop = False
        
        # Reset controller
        if self.controller:
            self.controller.reset()
        
        start_time = time.time()
        
        try:
            for step in range(steps):
                if self.emergency_stop:
                    self.logger.warning("Emergency stop triggered")
                    break
                
                # Update simulation
                self._update_step(dt)
                
                # Store results
                if step % self.config.save_frequency == 0:
                    self._save_step()
                
                # Real-time simulation
                if self.config.real_time:
                    elapsed = time.time() - start_time
                    target_time = step * dt
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                
                self.current_step += 1
                self.current_time += dt
            
            # Final save
            self._save_step()
            
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False
        
        # Compute final metrics
        self._compute_final_metrics()
        
        total_time = time.time() - start_time
        self.results.total_steps = self.current_step
        self.results.total_time = total_time
        
        self.logger.info(f"Simulation completed in {total_time:.2f}s")
        
        return self.results
    
    def _update_step(self, dt: float) -> None:
        """Update simulation for one time step.
        
        Args:
            dt: Time step duration
        """
        # Create swarm state
        swarm_state = SwarmState(
            robots=self.robots.copy(),
            time=self.current_time,
            step=self.current_step
        )
        
        # Compute control inputs
        if self.controller:
            controls = self.controller.update(swarm_state, dt)
        else:
            controls = {robot.id: np.zeros(2) for robot in self.robots}
        
        # Update robot states
        for robot in self.robots:
            # Get control input
            acceleration = controls.get(robot.id, np.zeros(2))
            
            # Apply safety limits
            robot.velocity, robot.acceleration = self.safety_limits.apply_limits(
                robot.velocity, acceleration
            )
            
            # Update position
            robot.position += robot.velocity * dt
            
            # Apply boundary conditions
            self._apply_boundary_conditions(robot)
            
            # Check for collisions
            if self.config.collision_detection:
                self._check_collisions(robot)
    
    def _apply_boundary_conditions(self, robot: RobotState) -> None:
        """Apply boundary conditions to robot.
        
        Args:
            robot: Robot state to update
        """
        if self.config.boundaries == "reflective":
            # Reflective boundaries
            if robot.position[0] < self.config.robot_radius:
                robot.position[0] = self.config.robot_radius
                robot.velocity[0] = -robot.velocity[0]
            elif robot.position[0] > self.config.world_size[0] - self.config.robot_radius:
                robot.position[0] = self.config.world_size[0] - self.config.robot_radius
                robot.velocity[0] = -robot.velocity[0]
            
            if robot.position[1] < self.config.robot_radius:
                robot.position[1] = self.config.robot_radius
                robot.velocity[1] = -robot.velocity[1]
            elif robot.position[1] > self.config.world_size[1] - self.config.robot_radius:
                robot.position[1] = self.config.world_size[1] - self.config.robot_radius
                robot.velocity[1] = -robot.velocity[1]
        
        elif self.config.boundaries == "periodic":
            # Periodic boundaries
            robot.position[0] = robot.position[0] % self.config.world_size[0]
            robot.position[1] = robot.position[1] % self.config.world_size[1]
        
        elif self.config.boundaries == "absorbing":
            # Absorbing boundaries (robots disappear)
            if (robot.position[0] < 0 or robot.position[0] > self.config.world_size[0] or
                robot.position[1] < 0 or robot.position[1] > self.config.world_size[1]):
                robot.velocity = np.zeros(2)
                robot.acceleration = np.zeros(2)
    
    def _check_collisions(self, robot: RobotState) -> None:
        """Check for collisions with other robots.
        
        Args:
            robot: Robot to check collisions for
        """
        for other_robot in self.robots:
            if other_robot.id != robot.id:
                if self.safety_limits.check_collision(robot.position, other_robot.position):
                    # Simple collision response - separate robots
                    direction = robot.position - other_robot.position
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        separation_distance = 2 * self.config.robot_radius
                        robot.position = other_robot.position + direction * separation_distance
    
    def _save_step(self) -> None:
        """Save current simulation state."""
        self.results.positions.append([robot.position.copy() for robot in self.robots])
        self.results.velocities.append([robot.velocity.copy() for robot in self.robots])
        self.results.accelerations.append([robot.acceleration.copy() for robot in self.robots])
        self.results.times.append(self.current_time)
    
    def _compute_final_metrics(self) -> None:
        """Compute final performance metrics."""
        if not self.results.positions:
            return
        
        # Compute coverage (simplified)
        if len(self.results.positions) > 0:
            final_positions = self.results.positions[-1]
            # Simple coverage metric based on area spanned by robots
            positions_array = np.array(final_positions)
            if len(positions_array) > 0:
                min_pos = np.min(positions_array, axis=0)
                max_pos = np.max(positions_array, axis=0)
                covered_area = np.prod(max_pos - min_pos)
                total_area = np.prod(self.config.world_size)
                self.results.coverage = covered_area / total_area
        
        # Compute collision rate
        collision_count = 0
        total_pairs = 0
        
        for step_positions in self.results.positions:
            for i, pos1 in enumerate(step_positions):
                for j, pos2 in enumerate(step_positions[i+1:], i+1):
                    total_pairs += 1
                    if self.safety_limits.check_collision(pos1, pos2):
                        collision_count += 1
        
        if total_pairs > 0:
            self.results.collision_rate = collision_count / total_pairs
        
        # Compute energy consumption (simplified)
        total_energy = 0.0
        for step_accelerations in self.results.accelerations:
            for accel in step_accelerations:
                total_energy += np.linalg.norm(accel) ** 2
        
        self.results.energy_consumption = total_energy
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.is_running = False
    
    def emergency_stop_simulation(self) -> None:
        """Trigger emergency stop."""
        self.emergency_stop = True
        self.is_running = False
        self.logger.warning("Emergency stop activated")
    
    def get_current_state(self) -> SwarmState:
        """Get current swarm state."""
        return SwarmState(
            robots=self.robots.copy(),
            time=self.current_time,
            step=self.current_step
        )
    
    def add_robot(self, position: np.ndarray, velocity: np.ndarray = None) -> int:
        """Add a new robot to the swarm.
        
        Args:
            position: Initial position
            velocity: Initial velocity (default: zero)
            
        Returns:
            Robot ID
        """
        if velocity is None:
            velocity = np.zeros(2)
        
        robot_id = len(self.robots)
        robot = RobotState(
            position=position,
            velocity=velocity,
            acceleration=np.zeros(2),
            orientation=0.0,
            angular_velocity=0.0,
            id=robot_id
        )
        
        self.robots.append(robot)
        self.logger.info(f"Added robot {robot_id} at position {position}")
        
        return robot_id
    
    def remove_robot(self, robot_id: int) -> bool:
        """Remove a robot from the swarm.
        
        Args:
            robot_id: ID of robot to remove
            
        Returns:
            True if robot was removed, False if not found
        """
        for i, robot in enumerate(self.robots):
            if robot.id == robot_id:
                self.robots.pop(i)
                self.logger.info(f"Removed robot {robot_id}")
                return True
        
        return False
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save simulation configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            "dt": self.config.dt,
            "max_steps": self.config.max_steps,
            "real_time": self.config.real_time,
            "world_size": self.config.world_size,
            "boundaries": self.config.boundaries,
            "num_robots": self.config.num_robots,
            "robot_radius": self.config.robot_radius,
            "max_velocity": self.config.max_velocity,
            "max_acceleration": self.config.max_acceleration,
            "controller_type": self.config.controller_type,
            "controller_config": self.config.controller_config,
            "collision_detection": self.config.collision_detection,
            "emergency_stop": self.config.emergency_stop,
            "log_level": self.config.log_level,
            "save_trajectories": self.config.save_trajectories,
            "save_frequency": self.config.save_frequency,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Saved configuration to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'SwarmSimulation':
        """Load simulation from configuration file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            SwarmSimulation instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
