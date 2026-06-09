#!/usr/bin/env python3
"""
Safety Monitor Module
Provides safety features and emergency stop functionality for the self-driving RC car.
"""

import RPi.GPIO as GPIO
import time
import argparse
import sys
from typing import Dict, List, Optional, Callable
import json
import os
import threading
from datetime import datetime
import csv

from rc_car_controller import RCCarController
from sensor_manager import SensorManager


class SafetyMonitor:
    """Safety monitoring and emergency stop system."""
    
    def __init__(self, config_file: str = 'safety_config.json'):
        """
        Initialize Safety Monitor.
        
        Args:
            config_file (str): Path to configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize components
        self.car_controller = RCCarController()
        self.sensor_manager = SensorManager()
        
        # Safety parameters
        self.emergency_stop_distance = self.config['emergency_stop_distance']
        self.critical_distance = self.config['critical_distance']
        self.max_speed_limit = self.config['max_speed_limit']
        self.safety_check_interval = self.config['safety_check_interval']
        
        # Emergency stop button
        self.emergency_button_pin = self.config['emergency_button_pin']
        self.setup_emergency_button()
        
        # Safety state
        self.is_monitoring = False
        self.emergency_stop_active = False
        self.safety_violations = []
        self.last_safety_check = time.time()
        
        # Callbacks
        self.emergency_callbacks = []
        self.safety_violation_callbacks = []
        
        # Data logging
        self.safety_log_file = 'safety_log.csv'
        self.setup_safety_logging()
        
        print("Safety Monitor initialized successfully!")
        print(f"Emergency stop distance: {self.emergency_stop_distance}cm")
        print(f"Critical distance: {self.critical_distance}cm")
        print(f"Max speed limit: {self.max_speed_limit}")
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        default_config = {
            "emergency_stop_distance": 10.0,
            "critical_distance": 20.0,
            "max_speed_limit": 0.8,
            "safety_check_interval": 0.05,  # 20Hz
            "emergency_button_pin": 21,
            "auto_recovery": True,
            "recovery_timeout": 5.0,
            "safety_logging": True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded safety configuration from {self.config_file}")
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
            print(f"Safety configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_emergency_button(self):
        """Setup emergency stop button."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.emergency_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Add interrupt for emergency button
        GPIO.add_event_detect(self.emergency_button_pin, GPIO.FALLING, 
                             callback=self.emergency_button_callback, bouncetime=200)
        
        print(f"Emergency stop button configured on GPIO {self.emergency_button_pin}")
    
    def emergency_button_callback(self, channel):
        """Emergency button callback."""
        print("EMERGENCY STOP BUTTON PRESSED!")
        self.trigger_emergency_stop("Emergency button pressed")
    
    def setup_safety_logging(self):
        """Setup safety event logging."""
        if not os.path.exists(self.safety_log_file):
            with open(self.safety_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_type', 'severity', 'description',
                    'distance', 'speed', 'steering', 'action_taken'
                ])
    
    def log_safety_event(self, event_type: str, severity: str, description: str, 
                        sensor_data: Dict = None, action_taken: str = ""):
        """Log safety event."""
        try:
            with open(self.safety_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    event_type,
                    severity,
                    description,
                    sensor_data.get('distance_front', 0) if sensor_data else 0,
                    sensor_data.get('speed', 0) if sensor_data else 0,
                    sensor_data.get('steering', 0) if sensor_data else 0,
                    action_taken
                ])
        except Exception as e:
            print(f"Error logging safety event: {e}")
    
    def add_emergency_callback(self, callback: Callable):
        """Add emergency stop callback."""
        self.emergency_callbacks.append(callback)
    
    def add_safety_violation_callback(self, callback: Callable):
        """Add safety violation callback."""
        self.safety_violation_callbacks.append(callback)
    
    def trigger_emergency_stop(self, reason: str = "Safety violation"):
        """Trigger emergency stop."""
        print(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Stop the car immediately
        self.car_controller.emergency_stop()
        self.emergency_stop_active = True
        
        # Log the event
        self.log_safety_event("emergency_stop", "critical", reason)
        
        # Call emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(reason)
            except Exception as e:
                print(f"Error in emergency callback: {e}")
        
        # Sound alarm
        for _ in range(3):
            self.car_controller.beep(0.2)
            time.sleep(0.1)
    
    def check_safety_violations(self, sensor_data: Dict, current_speed: float, 
                               current_steering: float) -> List[str]:
        """
        Check for safety violations.
        
        Args:
            sensor_data (Dict): Current sensor readings
            current_speed (float): Current speed
            current_steering (float): Current steering angle
            
        Returns:
            List[str]: List of safety violations
        """
        violations = []
        
        # Check distance violations
        distance = sensor_data.get('distance_front', float('inf'))
        if distance < self.emergency_stop_distance:
            violations.append(f"Emergency stop distance violated: {distance:.1f}cm")
        elif distance < self.critical_distance:
            violations.append(f"Critical distance violated: {distance:.1f}cm")
        
        # Check speed violations
        if current_speed > self.max_speed_limit:
            violations.append(f"Speed limit exceeded: {current_speed:.2f} > {self.max_speed_limit}")
        
        # Check sensor failures
        if distance == float('inf'):
            violations.append("Ultrasonic sensor failure")
        
        # Check for stuck condition
        if hasattr(self, 'last_positions'):
            if len(self.last_positions) >= 10:
                # Check if car is stuck (same position for too long)
                recent_positions = self.last_positions[-10:]
                if all(pos == recent_positions[0] for pos in recent_positions):
                    violations.append("Car appears to be stuck")
        
        return violations
    
    def monitor_safety(self, get_current_state: Callable):
        """
        Start safety monitoring.
        
        Args:
            get_current_state (Callable): Function to get current car state
        """
        print("Starting safety monitoring...")
        self.is_monitoring = True
        self.last_positions = []
        
        try:
            while self.is_monitoring:
                # Get current state
                current_state = get_current_state()
                sensor_data = current_state.get('sensor_data', {})
                speed = current_state.get('speed', 0)
                steering = current_state.get('steering', 0)
                
                # Check for safety violations
                violations = self.check_safety_violations(sensor_data, speed, steering)
                
                if violations:
                    for violation in violations:
                        print(f"SAFETY VIOLATION: {violation}")
                        self.log_safety_event("safety_violation", "warning", violation, sensor_data)
                        
                        # Call violation callbacks
                        for callback in self.safety_violation_callbacks:
                            try:
                                callback(violation, sensor_data)
                            except Exception as e:
                                print(f"Error in violation callback: {e}")
                
                # Store position for stuck detection
                self.last_positions.append((speed, steering))
                if len(self.last_positions) > 20:
                    self.last_positions.pop(0)
                
                # Check emergency button
                if GPIO.input(self.emergency_button_pin) == GPIO.LOW:
                    self.trigger_emergency_stop("Emergency button pressed")
                
                time.sleep(self.safety_check_interval)
                
        except KeyboardInterrupt:
            print("Safety monitoring stopped by user")
        finally:
            self.is_monitoring = False
    
    def start_monitoring_thread(self, get_current_state: Callable):
        """Start safety monitoring in a separate thread."""
        self.monitor_thread = threading.Thread(
            target=self.monitor_safety,
            args=(get_current_state,),
            daemon=True
        )
        self.monitor_thread.start()
        print("Safety monitoring thread started")
    
    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.is_monitoring = False
        print("Safety monitoring stopped")
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        self.emergency_stop_active = False
        print("Emergency stop reset")
    
    def get_safety_status(self) -> Dict:
        """Get current safety status."""
        return {
            'monitoring': self.is_monitoring,
            'emergency_stop_active': self.emergency_stop_active,
            'violations_count': len(self.safety_violations),
            'last_check': self.last_safety_check,
            'config': self.config
        }
    
    def test_safety_systems(self):
        """Test safety systems."""
        print("=== Safety Systems Test ===")
        
        # Test emergency button
        print("Testing emergency button...")
        print("Press the emergency button to test")
        print("Press Ctrl+C to skip")
        
        try:
            while True:
                if GPIO.input(self.emergency_button_pin) == GPIO.LOW:
                    print("Emergency button detected!")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Emergency button test skipped")
        
        # Test distance monitoring
        print("Testing distance monitoring...")
        for i in range(5):
            sensor_data = self.sensor_manager.read_all_sensors()
            violations = self.check_safety_violations(sensor_data, 0.5, 0.0)
            print(f"Distance: {sensor_data['ultrasonic_distance']:.1f}cm, "
                  f"Violations: {len(violations)}")
            time.sleep(1)
        
        # Test emergency stop
        print("Testing emergency stop...")
        self.trigger_emergency_stop("Test emergency stop")
        time.sleep(2)
        self.reset_emergency_stop()
        
        print("Safety systems test complete!")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        self.car_controller.cleanup()
        self.sensor_manager.cleanup()
        GPIO.cleanup()
        print("Safety Monitor cleaned up")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Safety Monitor')
    parser.add_argument('--test-safety', action='store_true',
                       help='Test safety systems')
    parser.add_argument('--config', default='safety_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    safety_monitor = SafetyMonitor(args.config)
    
    try:
        if args.test_safety:
            safety_monitor.test_safety_systems()
        else:
            print("Safety Monitor ready")
            print("Use --test-safety to test safety systems")
    finally:
        safety_monitor.cleanup()


if __name__ == "__main__":
    main()
