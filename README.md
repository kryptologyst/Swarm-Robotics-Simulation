# Swarm Robotics Simulation

Research-ready swarm robotics simulation framework featuring advanced multi-robot coordination algorithms, comprehensive evaluation metrics, and interactive demos.

## Features

- **Advanced Swarm Algorithms**: Flocking, consensus, formation control, coverage control
- **Multiple Simulation Backends**: PyBullet, MuJoCo, custom physics
- **ROS 2 Integration**: Full ROS 2 support with launch files and visualization
- **Comprehensive Evaluation**: Performance metrics, leaderboards, and analysis tools
- **Interactive Demos**: Streamlit/Gradio interfaces and RViz2 visualization
- **Safety-First Design**: Built-in safety limits and emergency stop mechanisms

## Quick Start

### Prerequisites

- Python 3.10+
- ROS 2 Humble (optional, for ROS integration)
- CUDA-capable GPU (optional, for accelerated simulation)

### Installation

```bash
# Clone and setup
git clone https://github.com/kryptologyst/Swarm-Robotics-Simulation.git
cd Swarm-Robotics-Simulation

# Install dependencies
pip install -e .

# For ROS 2 integration (optional)
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### Basic Usage

```python
from swarm_robotics import SwarmSimulation, FlockingController

# Create simulation
sim = SwarmSimulation(
    num_robots=20,
    world_size=(50, 50),
    controller=FlockingController()
)

# Run simulation
results = sim.run(steps=1000, dt=0.01)
print(f"Coverage: {results.coverage:.2%}")
print(f"Collision Rate: {results.collision_rate:.3f}")
```

### ROS 2 Launch

```bash
# Launch with RViz2 visualization
ros2 launch swarm_robotics simulation.launch.py

# Launch with custom parameters
ros2 launch swarm_robotics simulation.launch.py num_robots:=50 world_size:=100
```

## Architecture

```
src/
├── controllers/          # Swarm control algorithms
├── simulation/          # Physics simulation backends
├── evaluation/         # Metrics and analysis
├── visualization/      # Plotting and demos
└── utils/              # Common utilities

robots/
├── urdf/               # Robot descriptions
└── meshes/             # 3D models

config/
├── controllers.yaml    # Controller parameters
├── simulation.yaml     # Simulation settings
└── evaluation.yaml     # Metrics configuration

launch/                 # ROS 2 launch files
tests/                  # Unit and integration tests
notebooks/              # Jupyter notebooks
assets/                 # Results and visualizations
```

## Algorithms

### Implemented Controllers

1. **Flocking Controller**: Reynolds' boids algorithm with alignment, cohesion, and separation
2. **Consensus Controller**: Distributed consensus for position and velocity agreement
3. **Formation Controller**: Maintain geometric formations (line, circle, diamond)
4. **Coverage Controller**: Optimal area coverage using Voronoi diagrams
5. **Distributed MPC**: Model Predictive Control for collision avoidance

### Evaluation Metrics

- **Coverage**: Percentage of target area covered
- **Collision Rate**: Collisions per robot per second
- **Formation Error**: Deviation from desired formation
- **Energy Efficiency**: Total energy consumption
- **Convergence Time**: Time to reach consensus/formation
- **Communication Load**: Messages per robot per second

## Demos

### Interactive Web Demo

```bash
streamlit run demo/app.py
```

Features:
- Real-time parameter adjustment
- Live visualization
- Performance metrics
- Algorithm comparison

### ROS 2 Visualization

```bash
ros2 launch swarm_robotics rviz_demo.launch.py
```

Features:
- 3D robot visualization
- Trajectory playback
- Sensor data streams
- Interactive markers

## Configuration

All parameters are configurable via YAML files:

```yaml
# config/simulation.yaml
simulation:
  backend: "pybullet"  # pybullet, mujoco, custom
  dt: 0.01
  max_steps: 10000
  
world:
  size: [50, 50]
  obstacles: true
  boundaries: "reflective"  # reflective, periodic, absorbing

robots:
  num_robots: 20
  radius: 0.5
  max_velocity: 2.0
  max_acceleration: 5.0
```

## Safety Features

- **Velocity Limits**: Configurable maximum velocities
- **Acceleration Limits**: Smooth motion constraints
- **Collision Avoidance**: Multi-level collision detection
- **Emergency Stop**: Immediate halt capability
- **Boundary Handling**: Safe boundary interactions
- **Communication Timeouts**: Robust to communication failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this simulation in your research, please cite:

```bibtex
@software{swarm_robotics_simulation,
  title={Swarm Robotics Simulation Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Swarm-Robotics-Simulation}
}
```
# Swarm-Robotics-Simulation
