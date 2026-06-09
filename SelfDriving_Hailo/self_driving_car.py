#!/usr/bin/env python3
"""
Self-Driving Car Main Module
Orchestrates the complete self-driving RC car system.
"""

import argparse
import sys
import time
import signal
from typing import Optional
import json
import os
from datetime import datetime

from rc_car_controller import RCCarController
from sensor_manager import SensorManager
from navigation_system import NavigationSystem
from object_detector import ObjectDetector
from model_manager import ModelManager


class SelfDrivingCar:
    """Main self-driving car system."""
    
    def __init__(self, config_file: str = 'self_driving_config.json'):
        """
        Initialize Self-Driving Car.
        
        Args:
            config_file (str): Path to configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize components
        self.car_controller = RCCarController()
        self.sensor_manager = SensorManager()
        self.navigation_system = NavigationSystem()
        
        # Object detection (optional)
        self.object_detector = None
        self.model_manager = None
        self.enable_object_detection = self.config.get('object_detection', {}).get('enabled', False)
        
        if self.enable_object_detection:
            self.setup_object_detection()
        
        # System state
        self.is_running = False
        self.current_mode = 'manual'
        self.emergency_stop_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("Self-Driving Car System initialized successfully!")
        print(f"Object detection: {'Enabled' if self.enable_object_detection else 'Disabled'}")
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        default_config = {
            "object_detection": {
                "enabled": False,
                "model_path": "models",
                "confidence_threshold": 0.7
            },
            "safety": {
                "emergency_stop_button": True,
                "max_speed_limit": 0.8,
                "auto_stop_on_error": True
            },
            "logging": {
                "enabled": True,
                "log_file": "self_driving_log.csv",
                "log_level": "INFO"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded configuration from {self.config_file}")
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
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_object_detection(self):
        """Setup object detection system."""
        try:
            print("Setting up object detection...")
            self.model_manager = ModelManager(self.config['object_detection']['model_path'])
            
            if not self.model_manager.download_model():
                print("Failed to download model. Disabling object detection.")
                self.enable_object_detection = False
                return
            
            model_info = self.model_manager.get_model_info()
            self.object_detector = ObjectDetector(
                model_path=model_info['model_path'],
                labels=model_info['labels'],
                csv_path='object_detection_log.csv'
            )
            
            print("Object detection setup complete!")
            
        except Exception as e:
            print(f"Error setting up object detection: {e}")
            self.enable_object_detection = False
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.emergency_stop_requested = True
        self.stop()
    
    def run_autonomous_mode(self, speed: float = 0.5, duration: int = 60):
        """
        Run autonomous driving mode.
        
        Args:
            speed (float): Maximum speed (0.0 to 1.0)
            duration (int): Duration in seconds
        """
        print(f"Starting autonomous driving mode...")
        print(f"Speed limit: {speed}, Duration: {duration}s")
        print("Press Ctrl+C to stop")
        
        self.current_mode = 'autonomous'
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and not self.emergency_stop_requested and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Object detection (if enabled)
                if self.enable_object_detection and self.object_detector:
                    try:
                        # This would require camera integration
                        # For now, we'll simulate object detection
                        detected_objects = self.simulate_object_detection()
                        sensor_data.update(detected_objects)
                    except Exception as e:
                        print(f"Object detection error: {e}")
                
                # Make navigation decision
                decision = self.navigation_system.make_autonomous_decision(sensor_data)
                
                # Apply speed limit
                decision['speed'] = min(decision['speed'], speed)
                
                # Execute decision
                self.navigation_system.execute_decision(decision)
                
                # Print status
                self.print_status(sensor_data, decision)
                
                time.sleep(0.1)  # 10Hz control loop
                
        except KeyboardInterrupt:
            print("Autonomous driving stopped by user")
        finally:
            self.stop()
    
    def run_line_following_mode(self, speed: float = 0.3, duration: int = 60):
        """
        Run line following mode.
        
        Args:
            speed (float): Maximum speed (0.0 to 1.0)
            duration (int): Duration in seconds
        """
        print(f"Starting line following mode...")
        print(f"Speed limit: {speed}, Duration: {duration}s")
        print("Place the car on a line and press Enter to start")
        input()
        
        self.current_mode = 'line_following'
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and not self.emergency_stop_requested and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Line following logic
                if abs(sensor_data['line_position']) > 0.1:
                    # Line detected
                    decision = {
                        'speed': min(speed, 0.4),
                        'steering': -sensor_data['line_position'] * 0.6,
                        'action': 'follow_line'
                    }
                else:
                    # No line detected, slow down and search
                    decision = {
                        'speed': min(speed * 0.3, 0.2),
                        'steering': 0.0,
                        'action': 'search_line'
                    }
                
                # Execute decision
                self.navigation_system.execute_decision(decision)
                
                # Print status
                self.print_status(sensor_data, decision)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Line following stopped by user")
        finally:
            self.stop()
    
    def run_obstacle_avoidance_mode(self, speed: float = 0.4, duration: int = 60):
        """
        Run obstacle avoidance mode.
        
        Args:
            speed (float): Maximum speed (0.0 to 1.0)
            duration (int): Duration in seconds
        """
        print(f"Starting obstacle avoidance mode...")
        print(f"Speed limit: {speed}, Duration: {duration}s")
        print("Place obstacles in the path and press Enter to start")
        input()
        
        self.current_mode = 'obstacle_avoidance'
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and not self.emergency_stop_requested and (time.time() - start_time) < duration:
                # Get sensor data
                sensor_data = self.sensor_manager.get_navigation_data()
                
                # Obstacle avoidance logic
                if sensor_data['obstacle_detected']:
                    # Obstacle detected, avoid it
                    ir_readings = {
                        'left': sensor_data['left_ir'],
                        'right': sensor_data['right_ir'],
                        'front': sensor_data['front_ir']
                    }
                    
                    if ir_readings['left'] == 0 and ir_readings['right'] == 1:
                        # Turn left
                        decision = {
                            'speed': min(speed * 0.5, 0.3),
                            'steering': -0.7,
                            'action': 'avoid_left'
                        }
                    elif ir_readings['left'] == 1 and ir_readings['right'] == 0:
                        # Turn right
                        decision = {
                            'speed': min(speed * 0.5, 0.3),
                            'steering': 0.7,
                            'action': 'avoid_right'
                        }
                    else:
                        # Random turn
                        decision = {
                            'speed': min(speed * 0.3, 0.2),
                            'steering': 0.8 if time.time() % 2 > 1 else -0.8,
                            'action': 'avoid_random'
                        }
                else:
                    # No obstacle, drive forward
                    decision = {
                        'speed': speed,
                        'steering': 0.0,
                        'action': 'forward'
                    }
                
                # Execute decision
                self.navigation_system.execute_decision(decision)
                
                # Print status
                self.print_status(sensor_data, decision)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Obstacle avoidance stopped by user")
        finally:
            self.stop()
    
    def simulate_object_detection(self) -> dict:
        """Simulate object detection for testing."""
        # This would be replaced with actual camera-based object detection
        import random
        objects = ['person', 'car', 'bicycle', 'none']
        detected_object = random.choice(objects)
        confidence = random.uniform(0.3, 0.95) if detected_object != 'none' else 0.0
        
        return {
            'detected_object': detected_object,
            'object_confidence': confidence,
            'object_distance': random.uniform(10, 100)
        }
    
    def print_status(self, sensor_data: dict, decision: dict):
        """Print current system status."""
        print(f"[{self.current_mode}] "
              f"Dist: {sensor_data.get('distance_front', 0):.1f}cm, "
              f"Line: {sensor_data.get('line_position', 0):.2f}, "
              f"Speed: {decision.get('speed', 0):.2f}, "
              f"Steering: {decision.get('steering', 0):.2f}, "
              f"Action: {decision.get('action', 'unknown')}")
    
    def test_system(self):
        """Test all system components."""
        print("=== Self-Driving Car System Test ===")
        
        # Test sensors
        print("Testing sensors...")
        sensor_data = self.sensor_manager.get_navigation_data()
        print(f"Sensor data: {sensor_data}")
        
        # Test motor control
        print("Testing motor control...")
        self.car_controller.set_motor_speed(0.2, 'forward')
        time.sleep(1)
        self.car_controller.stop_motor()
        
        # Test steering
        print("Testing steering...")
        self.car_controller.set_steering(0.3)
        time.sleep(1)
        self.car_controller.center_steering()
        
        # Test buzzer
        print("Testing buzzer...")
        self.car_controller.beep(0.2)
        
        print("System test complete!")
    
    def stop(self):
        """Stop the self-driving car system."""
        self.is_running = False
        self.car_controller.stop_motor()
        self.car_controller.center_steering()
        print("Self-driving car system stopped")
    
    def get_status(self) -> dict:
        """Get current system status."""
        return {
            'mode': self.current_mode,
            'running': self.is_running,
            'object_detection_enabled': self.enable_object_detection,
            'emergency_stop_requested': self.emergency_stop_requested,
            'config': self.config
        }
    
    def cleanup(self):
        """Cleanup all resources."""
        self.stop()
        self.car_controller.cleanup()
        self.sensor_manager.cleanup()
        self.navigation_system.cleanup()
        print("Self-driving car system cleaned up")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Self-Driving RC Car System')
    parser.add_argument('--mode', choices=['autonomous', 'line-following', 'obstacle-avoidance', 'test'],
                       default='autonomous', help='Operating mode')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='Maximum speed (0.0 to 1.0)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds')
    parser.add_argument('--config', default='self_driving_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Validate speed
    if not 0.0 <= args.speed <= 1.0:
        print("Error: Speed must be between 0.0 and 1.0")
        sys.exit(1)
    
    car = SelfDrivingCar(args.config)
    
    try:
        if args.mode == 'test':
            car.test_system()
        elif args.mode == 'autonomous':
            car.run_autonomous_mode(args.speed, args.duration)
        elif args.mode == 'line-following':
            car.run_line_following_mode(args.speed, args.duration)
        elif args.mode == 'obstacle-avoidance':
            car.run_obstacle_avoidance_mode(args.speed, args.duration)
    finally:
        car.cleanup()


if __name__ == "__main__":
    main()
