#!/usr/bin/env python3
"""ROS 2 launch file for swarm robotics simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for swarm robotics simulation."""
    
    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='20',
        description='Number of robots in the swarm'
    )
    
    world_size_arg = DeclareLaunchArgument(
        'world_size',
        default_value='50',
        description='Size of the simulation world'
    )
    
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='flocking',
        description='Swarm control algorithm (flocking, consensus, formation, coverage)'
    )
    
    max_steps_arg = DeclareLaunchArgument(
        'max_steps',
        default_value='5000',
        description='Maximum simulation steps'
    )
    
    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.01',
        description='Simulation time step'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz2'
    )
    
    # Swarm simulation node
    swarm_sim_node = Node(
        package='swarm_robotics',
        executable='swarm_simulation_node',
        name='swarm_simulation',
        parameters=[{
            'num_robots': LaunchConfiguration('num_robots'),
            'world_size': LaunchConfiguration('world_size'),
            'algorithm': LaunchConfiguration('algorithm'),
            'max_steps': LaunchConfiguration('max_steps'),
            'dt': LaunchConfiguration('dt'),
            'robot_radius': 0.5,
            'max_velocity': 2.0,
            'max_acceleration': 5.0,
            'boundaries': 'reflective',
            'collision_detection': True,
            'save_frequency': 10,
        }],
        output='screen'
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('swarm_robotics'),
            'rviz',
            'swarm_simulation.rviz'
        )],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    # TF2 static transform publisher for world frame
    world_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        output='screen'
    )
    
    # Log info
    log_info = LogInfo(
        msg=['Starting swarm robotics simulation with ',
             LaunchConfiguration('num_robots'), ' robots using ',
             LaunchConfiguration('algorithm'), ' algorithm']
    )
    
    return LaunchDescription([
        num_robots_arg,
        world_size_arg,
        algorithm_arg,
        max_steps_arg,
        dt_arg,
        use_rviz_arg,
        log_info,
        swarm_sim_node,
        rviz_node,
        world_tf_node,
    ])
