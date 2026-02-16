#!/usr/bin/env python3
"""Setup script for swarm robotics simulation."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Swarm Robotics Simulation")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("Failed to install package. Please check your Python environment.")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("Warning: Failed to install development dependencies")
    
    # Install ROS 2 dependencies (optional)
    ros2_available = run_command("which ros2", "Checking for ROS 2")
    if ros2_available:
        print("ROS 2 detected. Installing ROS 2 dependencies...")
        run_command("pip install -e .[ros2]", "Installing ROS 2 dependencies")
    else:
        print("ROS 2 not detected. Skipping ROS 2 dependencies.")
        print("To install ROS 2 dependencies later, run: pip install -e .[ros2]")
    
    # Run tests
    print("\nRunning tests...")
    if run_command("python -m pytest tests/ -v", "Running test suite"):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please check the output above.")
    
    # Run linting
    print("\nRunning code quality checks...")
    run_command("black --check src/", "Checking code formatting with black")
    run_command("ruff check src/", "Checking code quality with ruff")
    
    # Create necessary directories
    print("\nCreating necessary directories...")
    directories = ["results", "logs", "assets", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the basic example: python 0662.py")
    print("2. Try the advanced examples: python examples/basic_example.py")
    print("3. Launch the Streamlit demo: streamlit run demo/streamlit_app.py")
    print("4. For ROS 2: ros2 launch swarm_robotics simulation.launch.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
