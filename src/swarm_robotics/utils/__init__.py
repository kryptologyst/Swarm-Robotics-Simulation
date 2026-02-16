"""Utility functions for swarm robotics simulation."""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import yaml
import json
import logging
from pathlib import Path
import random
import torch
import os


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def normalize_vector(vector: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """Normalize vector to maximum norm.
    
    Args:
        vector: Input vector
        max_norm: Maximum norm value
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm > max_norm:
        return vector / norm * max_norm
    return vector


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix.
    
    Args:
        positions: Array of positions
        
    Returns:
        Distance matrix
    """
    return np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)


def find_neighbors(positions: np.ndarray, radius: float) -> List[List[int]]:
    """Find neighbors within radius for each position.
    
    Args:
        positions: Array of positions
        radius: Neighbor radius
        
    Returns:
        List of neighbor indices for each position
    """
    distances = compute_distance_matrix(positions)
    neighbors = []
    
    for i in range(len(positions)):
        neighbor_indices = np.where((distances[i] <= radius) & (distances[i] > 0))[0]
        neighbors.append(neighbor_indices.tolist())
    
    return neighbors


def compute_centroid(positions: np.ndarray) -> np.ndarray:
    """Compute centroid of positions.
    
    Args:
        positions: Array of positions
        
    Returns:
        Centroid position
    """
    return np.mean(positions, axis=0)


def compute_variance(positions: np.ndarray) -> float:
    """Compute variance of positions.
    
    Args:
        positions: Array of positions
        
    Returns:
        Position variance
    """
    centroid = compute_centroid(positions)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return np.var(distances)


def check_collision(pos1: np.ndarray, pos2: np.ndarray, radius: float) -> bool:
    """Check if two positions are in collision.
    
    Args:
        pos1: First position
        pos2: Second position
        radius: Collision radius
        
    Returns:
        True if collision detected
    """
    distance = np.linalg.norm(pos1 - pos2)
    return distance < 2 * radius


def apply_boundary_conditions(position: np.ndarray, world_size: np.ndarray, 
                            boundary_type: str = "reflective") -> Tuple[np.ndarray, np.ndarray]:
    """Apply boundary conditions to position.
    
    Args:
        position: Current position
        world_size: World size [width, height]
        boundary_type: Type of boundary condition
        
    Returns:
        Updated position and velocity
    """
    new_position = position.copy()
    velocity = np.zeros(2)  # Placeholder for velocity update
    
    if boundary_type == "reflective":
        # Reflective boundaries
        if new_position[0] < 0:
            new_position[0] = 0
            velocity[0] = -velocity[0]  # Reflect velocity
        elif new_position[0] > world_size[0]:
            new_position[0] = world_size[0]
            velocity[0] = -velocity[0]
        
        if new_position[1] < 0:
            new_position[1] = 0
            velocity[1] = -velocity[1]
        elif new_position[1] > world_size[1]:
            new_position[1] = world_size[1]
            velocity[1] = -velocity[1]
    
    elif boundary_type == "periodic":
        # Periodic boundaries
        new_position[0] = new_position[0] % world_size[0]
        new_position[1] = new_position[1] % world_size[1]
    
    elif boundary_type == "absorbing":
        # Absorbing boundaries (position stays, velocity becomes zero)
        if (new_position[0] < 0 or new_position[0] > world_size[0] or
            new_position[1] < 0 or new_position[1] > world_size[1]):
            velocity = np.zeros(2)
    
    return new_position, velocity


def interpolate_trajectory(positions: List[np.ndarray], num_points: int) -> List[np.ndarray]:
    """Interpolate trajectory to have specified number of points.
    
    Args:
        positions: List of positions
        num_points: Number of interpolated points
        
    Returns:
        Interpolated positions
    """
    if len(positions) <= 1:
        return positions
    
    positions_array = np.array(positions)
    t_original = np.linspace(0, 1, len(positions))
    t_new = np.linspace(0, 1, num_points)
    
    interpolated = []
    for i in range(positions_array.shape[1]):
        interpolated_coord = np.interp(t_new, t_original, positions_array[:, i])
        interpolated.append(interpolated_coord)
    
    interpolated_positions = []
    for i in range(num_points):
        interpolated_positions.append(np.array([interpolated[0][i], interpolated[1][i]]))
    
    return interpolated_positions


def compute_formation_positions(formation_type: str, num_robots: int, 
                               size: float = 2.0) -> List[np.ndarray]:
    """Compute formation positions for given type and number of robots.
    
    Args:
        formation_type: Type of formation
        num_robots: Number of robots
        size: Formation size
        
    Returns:
        List of formation positions
    """
    positions = []
    
    if formation_type == "line":
        for i in range(num_robots):
            x = (i - (num_robots - 1) / 2) * size / num_robots
            positions.append(np.array([x, 0.0]))
    
    elif formation_type == "circle":
        for i in range(num_robots):
            angle = 2 * np.pi * i / num_robots
            x = size * np.cos(angle)
            y = size * np.sin(angle)
            positions.append(np.array([x, y]))
    
    elif formation_type == "diamond":
        if num_robots >= 4:
            positions = [
                np.array([0.0, size]),      # Top
                np.array([size, 0.0]),      # Right
                np.array([0.0, -size]),     # Bottom
                np.array([-size, 0.0]),     # Left
            ]
            # Add additional robots in the center
            for i in range(4, num_robots):
                positions.append(np.array([0.0, 0.0]))
        else:
            # Fallback to circle for fewer robots
            return compute_formation_positions("circle", num_robots, size)
    
    else:
        # Default to circle formation
        return compute_formation_positions("circle", num_robots, size)
    
    return positions


def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Convert all numpy objects
    converted_results = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def create_output_directory(base_dir: Union[str, Path], 
                          experiment_name: str) -> Path:
    """Create output directory for experiment.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of experiment
        
    Returns:
        Created directory path
    """
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["simulation", "world", "robots", "controllers"]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate simulation config
    sim_config = config["simulation"]
    if "dt" not in sim_config or sim_config["dt"] <= 0:
        logging.error("Invalid simulation dt")
        return False
    
    if "max_steps" not in sim_config or sim_config["max_steps"] <= 0:
        logging.error("Invalid max_steps")
        return False
    
    # Validate world config
    world_config = config["world"]
    if "size" not in world_config or len(world_config["size"]) != 2:
        logging.error("Invalid world size")
        return False
    
    # Validate robot config
    robot_config = config["robots"]
    if "num_robots" not in robot_config or robot_config["num_robots"] <= 0:
        logging.error("Invalid number of robots")
        return False
    
    return True


def get_controller_class(controller_type: str):
    """Get controller class by type.
    
    Args:
        controller_type: Type of controller
        
    Returns:
        Controller class
    """
    from ..controllers.flocking import FlockingController
    from ..controllers.consensus import ConsensusController
    from ..controllers.formation import FormationController
    from ..controllers.coverage import CoverageController
    
    controller_map = {
        "flocking": FlockingController,
        "consensus": ConsensusController,
        "formation": FormationController,
        "coverage": CoverageController,
    }
    
    if controller_type not in controller_map:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    return controller_map[controller_type]


def format_time(seconds: float) -> str:
    """Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"
