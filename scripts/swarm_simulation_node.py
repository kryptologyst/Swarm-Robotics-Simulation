"""ROS 2 node for swarm robotics simulation."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import numpy as np
import time
from typing import List, Dict, Any

from swarm_robotics import SwarmSimulation, set_random_seeds


class SwarmSimulationNode(Node):
    """ROS 2 node for swarm robotics simulation."""
    
    def __init__(self):
        """Initialize the swarm simulation node."""
        super().__init__('swarm_simulation')
        
        # Declare parameters
        self.declare_parameter('num_robots', 20)
        self.declare_parameter('world_size', 50.0)
        self.declare_parameter('algorithm', 'flocking')
        self.declare_parameter('max_steps', 5000)
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('robot_radius', 0.5)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('max_acceleration', 5.0)
        self.declare_parameter('boundaries', 'reflective')
        self.declare_parameter('collision_detection', True)
        self.declare_parameter('save_frequency', 10)
        
        # Get parameters
        self.num_robots = self.get_parameter('num_robots').value
        self.world_size = self.get_parameter('world_size').value
        self.algorithm = self.get_parameter('algorithm').value
        self.max_steps = self.get_parameter('max_steps').value
        self.dt = self.get_parameter('dt').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.boundaries = self.get_parameter('boundaries').value
        self.collision_detection = self.get_parameter('collision_detection').value
        self.save_frequency = self.get_parameter('save_frequency').value
        
        # Initialize simulation
        self._initialize_simulation()
        
        # Create publishers
        self._create_publishers()
        
        # Create subscribers
        self._create_subscribers()
        
        # Create timer for simulation update
        self.timer = self.create_timer(self.dt, self.simulation_callback)
        
        # Simulation state
        self.current_step = 0
        self.is_running = True
        
        self.get_logger().info(f'Swarm simulation node initialized with {self.num_robots} robots')
    
    def _initialize_simulation(self):
        """Initialize the swarm simulation."""
        set_random_seeds(42)
        
        # Create simulation configuration
        config = {
            "dt": self.dt,
            "max_steps": self.max_steps,
            "world_size": [self.world_size, self.world_size],
            "num_robots": self.num_robots,
            "robot_radius": self.robot_radius,
            "controller_type": self.algorithm,
            "controller_config": {
                "max_velocity": self.max_velocity,
                "max_acceleration": self.max_acceleration,
            },
            "boundaries": self.boundaries,
            "collision_detection": self.collision_detection,
            "save_frequency": self.save_frequency,
        }
        
        # Algorithm-specific configuration
        if self.algorithm == "flocking":
            config["controller_config"].update({
                "perception_range": 2.0,
                "alignment_weight": 0.1,
                "cohesion_weight": 0.1,
                "separation_weight": 0.2,
            })
        elif self.algorithm == "consensus":
            config["controller_config"].update({
                "communication_range": 3.0,
                "consensus_gain": 0.5,
                "consensus_type": "both",
            })
        elif self.algorithm == "formation":
            config["controller_config"].update({
                "formation_type": "circle",
                "formation_size": 3.0,
                "formation_center": [self.world_size/2, self.world_size/2],
                "formation_gain": 1.0,
                "collision_avoidance_gain": 2.0,
            })
        elif self.algorithm == "coverage":
            config["controller_config"].update({
                "coverage_area": [[-self.world_size/4, -self.world_size/4], 
                                [self.world_size/4, self.world_size/4]],
                "coverage_gain": 1.0,
                "collision_avoidance_gain": 2.0,
            })
        
        # Create simulation
        self.simulation = SwarmSimulation(config)
        
        # Initialize robot states
        self.robot_states = []
        for i in range(self.num_robots):
            self.robot_states.append({
                'position': np.zeros(2),
                'velocity': np.zeros(2),
                'orientation': 0.0,
            })
    
    def _create_publishers(self):
        """Create ROS 2 publishers."""
        # Robot pose publishers
        self.pose_publishers = []
        self.odom_publishers = []
        self.cmd_vel_publishers = []
        
        for i in range(self.num_robots):
            # Pose publisher
            pose_pub = self.create_publisher(
                PoseStamped,
                f'/robot_{i}/pose',
                10
            )
            self.pose_publishers.append(pose_pub)
            
            # Odometry publisher
            odom_pub = self.create_publisher(
                Odometry,
                f'/robot_{i}/odom',
                10
            )
            self.odom_publishers.append(odom_pub)
            
            # Command velocity publisher
            cmd_vel_pub = self.create_publisher(
                Twist,
                f'/robot_{i}/cmd_vel',
                10
            )
            self.cmd_vel_publishers.append(cmd_vel_pub)
        
        # Visualization publishers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/swarm_markers',
            10
        )
        
        self.trajectory_publisher = self.create_publisher(
            MarkerArray,
            '/swarm_trajectories',
            10
        )
    
    def _create_subscribers(self):
        """Create ROS 2 subscribers."""
        # Emergency stop subscriber
        self.emergency_stop_sub = self.create_subscription(
            std_msgs.msg.Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        
        # Command subscribers for each robot
        self.cmd_subscribers = []
        for i in range(self.num_robots):
            cmd_sub = self.create_subscription(
                Twist,
                f'/robot_{i}/cmd_vel',
                lambda msg, robot_id=i: self.cmd_vel_callback(msg, robot_id),
                10
            )
            self.cmd_subscribers.append(cmd_sub)
    
    def simulation_callback(self):
        """Main simulation callback."""
        if not self.is_running:
            return
        
        # Update simulation
        self._update_simulation()
        
        # Publish robot states
        self._publish_robot_states()
        
        # Publish visualization markers
        self._publish_visualization()
        
        # Check if simulation should stop
        if self.current_step >= self.max_steps:
            self.get_logger().info('Simulation completed')
            self.is_running = False
        
        self.current_step += 1
    
    def _update_simulation(self):
        """Update the simulation state."""
        # Get current swarm state
        swarm_state = self.simulation.get_current_state()
        
        # Update robot states
        for i, robot in enumerate(swarm_state.robots):
            self.robot_states[i]['position'] = robot.position.copy()
            self.robot_states[i]['velocity'] = robot.velocity.copy()
            self.robot_states[i]['orientation'] = robot.orientation
    
    def _publish_robot_states(self):
        """Publish robot states to ROS topics."""
        current_time = self.get_clock().now().to_msg()
        
        for i, robot_state in enumerate(self.robot_states):
            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = current_time
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = float(robot_state['position'][0])
            pose_msg.pose.position.y = float(robot_state['position'][1])
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            
            self.pose_publishers[i].publish(pose_msg)
            
            # Publish odometry
            odom_msg = Odometry()
            odom_msg.header.stamp = current_time
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = f'robot_{i}'
            odom_msg.pose.pose = pose_msg.pose
            odom_msg.twist.twist.linear.x = float(robot_state['velocity'][0])
            odom_msg.twist.twist.linear.y = float(robot_state['velocity'][1])
            
            self.odom_publishers[i].publish(odom_msg)
    
    def _publish_visualization(self):
        """Publish visualization markers."""
        current_time = self.get_clock().now().to_msg()
        
        # Robot markers
        marker_array = MarkerArray()
        
        for i, robot_state in enumerate(self.robot_states):
            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(robot_state['position'][0])
            marker.pose.position.y = float(robot_state['position'][1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = self.robot_radius * 2
            marker.scale.y = self.robot_radius * 2
            marker.scale.z = 0.1
            
            # Color
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.marker_publisher.publish(marker_array)
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop command."""
        if msg.data:
            self.get_logger().warn('Emergency stop activated!')
            self.is_running = False
            self.simulation.emergency_stop_simulation()
    
    def cmd_vel_callback(self, msg, robot_id):
        """Handle command velocity for a specific robot."""
        # This could be used to override the swarm controller
        # For now, we'll just log the command
        self.get_logger().debug(f'Received cmd_vel for robot {robot_id}: '
                               f'linear=({msg.linear.x}, {msg.linear.y}), '
                               f'angular={msg.angular.z}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = SwarmSimulationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down swarm simulation node')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
