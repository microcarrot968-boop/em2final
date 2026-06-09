#!/usr/bin/env python3
"""
Navigation System Module
Handles path planning and decision making for the self-driving RC car.
"""

import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np
from datetime import datetime
import csv
from sensor_manager import SensorManager
from rc_car_controller import RCCarController


class NavigationSystem:
    """Handles navigation and decision making for the RC car."""
    
    def __init__(self, config_file: str = 'navigation_config.json'):
        """
        Initialize Navigation System.
        
        Args:
            config_file (str): Path to configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize components
        self.sensor_manager = SensorManager()
        self.car_controller = RCCarController()
        
        # Navigation parameters
        self.safe_distance = self.config['safety']['safe_distance']
        self.critical_distance = self.config['safety']['critical_distance']
        self.max_speed = self.config['driving']['max_speed']
        self.min_speed = self.config['driving']['min_speed']
        self.turn_sensitivity = self.config['driving']['turn_sensitivity']
        
        # Line following parameters
        self.line_follow_speed = self.config['line_following']['speed']
        self.line_turn_strength = self.config['line_following']['turn_strength']
        
        # Obstacle avoidance parameters
        self.avoidance_distance = self.config['obstacle_avoidance']['avoidance_distance']
        self.avoidance_speed = self.config['obstacle_avoidance']['speed']
        
        # State tracking
        self.current_mode = 'manual'
        self.last_decision = None
        self.decision_history = []
        self.is_running = False
        
        # Data logging
        self.log_file = 'navigation_log.csv'
        self.setup_logging()
        
        print("Navigation System initialized successfully!")
        print(f"Safe distance: {self.safe_distance}cm")
        print(f"Critical distance: {self.critical_distance}cm")
        print(f"Max speed: {self.max_speed}")
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        default_config = {
            "safety": {
                "safe_distance": 50.0,
                "critical_distance": 20.0,
                "emergency_stop_distance": 10.0
            },
            "driving": {
                "max_speed": 0.8,
                "min_speed": 0.1,
                "turn_sensitivity": 0.5,
                "acceleration_rate": 0.1,
                "deceleration_rate": 0.2
            },
            "line_following": {
                "speed": 0.4,
                "turn_strength": 0.6,
                "line_lost_timeout": 2.0
            },
            "obstacle_avoidance": {
                "avoidance_distance": 40.0,
                "speed": 0.3,
                "turn_angle": 0.7,
                "backup_distance": 15.0
            },
            "object_detection": {
                "person_stop_distance": 30.0,
                "vehicle_caution_distance": 50.0,
                "confidence_threshold": 0.7
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded navigation configuration from {self.config_file}")
                return config
            except Exception as e:
                print(f"Error loading config: {e}. Using default configuration.")
                return default_config
        else:
            print(f"Config file {self.config_file} not found. Using default configuration.")
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: dict):
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Navigation configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_logging(self):
        """Setup data logging."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'mode', 'distance_front', 'line_position',
                    'obstacle_detected', 'speed', 'steering', 'action',
                    'decision_time_ms'
                ])
    
    def log_decision(self, sensor_data: Dict, decision: Dict, decision_time: float):
        """Log navigation decision."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.current_mode,
                    sensor_data.get('distance_front', 0),
                    sensor_data.get('line_position', 0),
                    sensor_data.get('obstacle_detected', False),
                    decision.get('speed', 0),
                    decision.get('steering', 0),
                    decision.get('action', 'unknown'),
                    decision_time
                ])
        except Exception as e:
            print(f"Error logging decision: {e}")
    
    def make_autonomous_decision(self, sensor_data: Dict) -> Dict:
        """
        Make autonomous driving decision based on sensor data.
        
        Args:
            sensor_data (Dict): Current sensor readings
            
        Returns:
            Dict: Driving decision (speed, steering, action)
        """
        start_time = time.perf_counter()
        
        distance = sensor_data.get('distance_front', float('inf'))
        line_position = sensor_data.get('line_position', 0)
        obstacle_detected = sensor_data.get('obstacle_detected', False)
        
        # Initialize decision
        decision = {
            'speed': 0.0,
            'steering': 0.0,
            'action': 'stop'
        }
        
        # Emergency stop for critical distance
        if distance < self.config['safety']['emergency_stop_distance']:
            decision = {
                'speed': 0.0,
                'steering': 0.0,
                'action': 'emergency_stop'
            }
        # Obstacle avoidance
        elif obstacle_detected or distance < self.avoidance_distance:
            decision = self.avoid_obstacle(sensor_data)
        # Line following
        elif abs(line_position) > 0.1:  # Line detected
            decision = self.follow_line(sensor_data)
        # Normal driving
        else:
            decision = self.normal_driving(sensor_data)
        
        # Apply speed limits
        decision['speed'] = max(self.min_speed, min(self.max_speed, decision['speed']))
        decision['steering'] = max(-1.0, min(1.0, decision['steering']))
        
        decision_time = (time.perf_counter() - start_time) * 1000
        self.log_decision(sensor_data, decision, decision_time)
        
        return decision
    
    def avoid_obstacle(self, sensor_data: Dict) -> Dict:
        """Obstacle avoidance logic."""
        distance = sensor_data.get('distance_front', float('inf'))
        ir_readings = {
            'left': sensor_data.get('left_ir', 0),
            'right': sensor_data.get('right_ir', 0),
            'front': sensor_data.get('front_ir', 0)
        }
        
        # Determine turn direction based on IR sensors
        if ir_readings['left'] == 0 and ir_readings['right'] == 1:
            # Obstacle on right, turn left
            steering = -self.config['obstacle_avoidance']['turn_angle']
        elif ir_readings['left'] == 1 and ir_readings['right'] == 0:
            # Obstacle on left, turn right
            steering = self.config['obstacle_avoidance']['turn_angle']
        else:
            # Obstacle ahead, turn based on distance
            if distance < self.critical_distance:
                # Very close, sharp turn
                steering = 0.8 if np.random.random() > 0.5 else -0.8
            else:
                # Moderate turn
                steering = 0.5 if np.random.random() > 0.5 else -0.5
        
        return {
            'speed': self.avoidance_speed,
            'steering': steering,
            'action': 'avoid_obstacle'
        }
    
    def follow_line(self, sensor_data: Dict) -> Dict:
        """Line following logic."""
        line_position = sensor_data.get('line_position', 0)
        
        # Calculate steering based on line position
        steering = -line_position * self.line_turn_strength
        
        return {
            'speed': self.line_follow_speed,
            'steering': steering,
            'action': 'follow_line'
        }
    
    def normal_driving(self, sensor_data: Dict) -> Dict:
        """Normal driving logic."""
        distance = sensor_data.get('distance_front', float('inf'))
        
        # Adjust speed based on distance
        if distance > self.safe_distance:
            speed = self.max_speed
        elif distance > self.critical_distance:
            speed = self.min_speed + (self.max_speed - self.min_speed) * (distance - self.critical_distance) / (self.safe_distance - self.critical_distance)
        else:
            speed = self.min_speed
        
        return {
            'speed': speed,
            'steering': 0.0,
            'action': 'normal_driving'
        }
    
    def execute_decision(self, decision: Dict):
        """Execute driving decision."""
        self.car_controller.set_motor_speed(decision['speed'], 'forward')
        self.car_controller.set_steering(decision['steering'])
        
        # Special actions
        if decision['action'] == 'emergency_stop':
            self.car_controller.emergency_stop()
        elif decision['action'] == 'avoid_obstacle':
            self.car_controller.beep(0.1)  # Short beep for obstacle
    
    def run_autonomous_mode(self, duration: int = 60):
        """
        Run autonomous driving mode.
        
        Args:
            duration (int): Duration in seconds
        """
        print(f"Starting autonomous driving for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        self.current_mode = 'autonomous'
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Make decision
                decision = self.make_autonomous_decision(sensor_data)
                
                # Execute decision
                self.execute_decision(decision)
                
                # Print status
                print(f"Mode: {self.current_mode}, Distance: {sensor_data['distance_front']:.1f}cm, "
                      f"Speed: {decision['speed']:.2f}, Steering: {decision['steering']:.2f}, "
                      f"Action: {decision['action']}")
                
                # Store decision history
                self.decision_history.append({
                    'timestamp': time.time(),
                    'sensor_data': sensor_data,
                    'decision': decision
                })
                
                time.sleep(0.1)  # 10Hz control loop
                
        except KeyboardInterrupt:
            print("Autonomous driving stopped by user")
        finally:
            self.stop()
    
    def run_line_following_mode(self, duration: int = 60):
        """
        Run line following mode.
        
        Args:
            duration (int): Duration in seconds
        """
        print(f"Starting line following for {duration} seconds...")
        print("Place the car on a line and press Enter to start")
        input()
        
        self.current_mode = 'line_following'
        self.is_running = True
        start_time = time.time()
        line_lost_time = 0
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Check if line is detected
                if abs(sensor_data['line_position']) > 0.1:
                    line_lost_time = 0
                    decision = self.follow_line(sensor_data)
                else:
                    line_lost_time += 0.1
                    if line_lost_time > self.config['line_following']['line_lost_timeout']:
                        decision = {'speed': 0.0, 'steering': 0.0, 'action': 'line_lost'}
                    else:
                        decision = {'speed': self.line_follow_speed * 0.5, 'steering': 0.0, 'action': 'search_line'}
                
                # Execute decision
                self.execute_decision(decision)
                
                # Print status
                print(f"Mode: {self.current_mode}, Line pos: {sensor_data['line_position']:.2f}, "
                      f"Speed: {decision['speed']:.2f}, Steering: {decision['steering']:.2f}, "
                      f"Action: {decision['action']}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Line following stopped by user")
        finally:
            self.stop()
    
    def run_obstacle_avoidance_mode(self, duration: int = 60):
        """
        Run obstacle avoidance mode.
        
        Args:
            duration (int): Duration in seconds
        """
        print(f"Starting obstacle avoidance for {duration} seconds...")
        print("Place obstacles in the path and press Enter to start")
        input()
        
        self.current_mode = 'obstacle_avoidance'
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Make decision
                decision = self.make_autonomous_decision(sensor_data)
                
                # Execute decision
                self.execute_decision(decision)
                
                # Print status
                print(f"Mode: {self.current_mode}, Distance: {sensor_data['distance_front']:.1f}cm, "
                      f"Speed: {decision['speed']:.2f}, Steering: {decision['steering']:.2f}, "
                      f"Action: {decision['action']}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Obstacle avoidance stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the navigation system."""
        self.is_running = False
        self.car_controller.stop_motor()
        self.car_controller.center_steering()
        print("Navigation system stopped")
    
    def get_status(self) -> Dict:
        """Get current navigation system status."""
        return {
            'mode': self.current_mode,
            'running': self.is_running,
            'decision_count': len(self.decision_history),
            'config': self.config
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        self.sensor_manager.cleanup()
        self.car_controller.cleanup()
        print("Navigation system cleaned up")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Navigation System')
    parser.add_argument('--mode', choices=['autonomous', 'line-following', 'obstacle-avoidance'],
                       default='autonomous', help='Navigation mode')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds')
    parser.add_argument('--config', default='navigation_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    navigation = NavigationSystem(args.config)
    
    try:
        if args.mode == 'autonomous':
            navigation.run_autonomous_mode(args.duration)
        elif args.mode == 'line-following':
            navigation.run_line_following_mode(args.duration)
        elif args.mode == 'obstacle-avoidance':
            navigation.run_obstacle_avoidance_mode(args.duration)
    finally:
        navigation.cleanup()


if __name__ == "__main__":
    main()
