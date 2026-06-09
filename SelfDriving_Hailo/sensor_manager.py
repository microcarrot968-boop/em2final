#!/usr/bin/env python3
"""
Sensor Manager Module
Handles all sensor data collection and processing for the self-driving RC car.
"""

import RPi.GPIO as GPIO
import time
import argparse
import sys
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np
from datetime import datetime


class SensorManager:
    """Manages all sensors for the RC car."""
    
    def __init__(self, config_file: str = 'sensor_config.json'):
        """
        Initialize Sensor Manager.
        
        Args:
            config_file (str): Path to configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # GPIO Pin Configuration
        self.ULTRASONIC_TRIGGER = self.config['pins']['ultrasonic_trigger']
        self.ULTRASONIC_ECHO = self.config['pins']['ultrasonic_echo']
        self.LEFT_IR = self.config['pins']['left_ir']
        self.RIGHT_IR = self.config['pins']['right_ir']
        self.FRONT_IR = self.config['pins']['front_ir']
        
        # Sensor Configuration
        self.MAX_DISTANCE = self.config['ultrasonic']['max_distance']
        self.MIN_DISTANCE = self.config['ultrasonic']['min_distance']
        self.SOUND_SPEED = self.config['ultrasonic']['sound_speed']
        
        # IR Sensor Configuration
        self.IR_THRESHOLD = self.config['ir']['threshold']
        self.IR_DEBOUNCE_TIME = self.config['ir']['debounce_time']
        
        # Initialize GPIO
        self.setup_gpio()
        
        # Sensor state
        self.last_ir_readings = {'left': 0, 'right': 0, 'front': 0}
        self.last_ir_time = time.time()
        self.is_initialized = True
        
        print("Sensor Manager initialized successfully!")
        print(f"Ultrasonic Trigger: GPIO {self.ULTRASONIC_TRIGGER}")
        print(f"Ultrasonic Echo: GPIO {self.ULTRASONIC_ECHO}")
        print(f"Left IR: GPIO {self.LEFT_IR}")
        print(f"Right IR: GPIO {self.RIGHT_IR}")
        print(f"Front IR: GPIO {self.FRONT_IR}")
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        default_config = {
            "pins": {
                "ultrasonic_trigger": 23,
                "ultrasonic_echo": 24,
                "left_ir": 9,
                "right_ir": 10,
                "front_ir": 25
            },
            "ultrasonic": {
                "max_distance": 400,
                "min_distance": 2,
                "sound_speed": 34300,
                "timeout": 0.1
            },
            "ir": {
                "threshold": 0.5,
                "debounce_time": 0.01
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded sensor configuration from {self.config_file}")
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
            print(f"Sensor configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_gpio(self):
        """Setup GPIO pins for sensors."""
        GPIO.setmode(GPIO.BCM)
        
        # Ultrasonic sensor
        GPIO.setup(self.ULTRASONIC_TRIGGER, GPIO.OUT)
        GPIO.setup(self.ULTRASONIC_ECHO, GPIO.IN)
        
        # IR sensors
        GPIO.setup(self.LEFT_IR, GPIO.IN)
        GPIO.setup(self.RIGHT_IR, GPIO.IN)
        GPIO.setup(self.FRONT_IR, GPIO.IN)
        
        # Initialize ultrasonic trigger
        GPIO.output(self.ULTRASONIC_TRIGGER, GPIO.LOW)
    
    def read_ultrasonic(self) -> float:
        """
        Read distance from ultrasonic sensor.
        
        Returns:
            float: Distance in centimeters
        """
        if not self.is_initialized:
            return float('inf')
        
        try:
            # Send trigger pulse
            GPIO.output(self.ULTRASONIC_TRIGGER, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.ULTRASONIC_TRIGGER, GPIO.LOW)
            
            # Wait for echo start
            start_time = time.time()
            timeout = self.config['ultrasonic']['timeout']
            
            while GPIO.input(self.ULTRASONIC_ECHO) == 0:
                if time.time() - start_time > timeout:
                    return float('inf')
                start_time = time.time()
            
            # Wait for echo end
            end_time = time.time()
            while GPIO.input(self.ULTRASONIC_ECHO) == 1:
                if time.time() - start_time > timeout:
                    return float('inf')
                end_time = time.time()
            
            # Calculate distance
            duration = end_time - start_time
            distance = (duration * self.SOUND_SPEED) / 2
            
            # Clamp to valid range
            distance = max(self.MIN_DISTANCE, min(self.MAX_DISTANCE, distance))
            
            return distance
            
        except Exception as e:
            print(f"Error reading ultrasonic sensor: {e}")
            return float('inf')
    
    def read_ir_sensors(self) -> Dict[str, int]:
        """
        Read all IR sensors.
        
        Returns:
            Dict[str, int]: Dictionary with sensor readings (0 or 1)
        """
        if not self.is_initialized:
            return {'left': 0, 'right': 0, 'front': 0}
        
        try:
            # Read with debouncing
            current_time = time.time()
            if current_time - self.last_ir_time < self.IR_DEBOUNCE_TIME:
                return self.last_ir_readings
            
            readings = {
                'left': GPIO.input(self.LEFT_IR),
                'right': GPIO.input(self.RIGHT_IR),
                'front': GPIO.input(self.FRONT_IR)
            }
            
            self.last_ir_readings = readings
            self.last_ir_time = current_time
            
            return readings
            
        except Exception as e:
            print(f"Error reading IR sensors: {e}")
            return {'left': 0, 'right': 0, 'front': 0}
    
    def read_all_sensors(self) -> Dict[str, float]:
        """
        Read all sensors and return comprehensive data.
        
        Returns:
            Dict[str, float]: Dictionary with all sensor readings
        """
        ultrasonic_distance = self.read_ultrasonic()
        ir_readings = self.read_ir_sensors()
        
        return {
            'ultrasonic_distance': ultrasonic_distance,
            'left_ir': ir_readings['left'],
            'right_ir': ir_readings['right'],
            'front_ir': ir_readings['front'],
            'timestamp': datetime.now().isoformat()
        }
    
    def is_obstacle_detected(self, threshold: float = 30.0) -> bool:
        """
        Check if obstacle is detected.
        
        Args:
            threshold (float): Distance threshold in cm
            
        Returns:
            bool: True if obstacle detected
        """
        distance = self.read_ultrasonic()
        return distance < threshold
    
    def is_line_detected(self) -> Dict[str, bool]:
        """
        Check if line is detected by IR sensors.
        
        Returns:
            Dict[str, bool]: Dictionary with line detection status
        """
        ir_readings = self.read_ir_sensors()
        return {
            'left_line': ir_readings['left'] == 1,
            'right_line': ir_readings['right'] == 1,
            'front_line': ir_readings['front'] == 1
        }
    
    def get_navigation_data(self) -> Dict[str, float]:
        """
        Get processed navigation data.
        
        Returns:
            Dict[str, float]: Processed navigation data
        """
        sensor_data = self.read_all_sensors()
        
        # Process line following data
        line_data = self.is_line_detected()
        
        # Calculate line position (-1 = left, 0 = center, 1 = right)
        if line_data['left_line'] and not line_data['right_line']:
            line_position = -1.0  # Line on left, need to turn right
        elif line_data['right_line'] and not line_data['left_line']:
            line_position = 1.0   # Line on right, need to turn left
        elif line_data['left_line'] and line_data['right_line']:
            line_position = 0.0   # Line centered
        else:
            line_position = 0.0   # No line detected, go straight
        
        return {
            'distance_front': sensor_data['ultrasonic_distance'],
            'line_position': line_position,
            'obstacle_detected': sensor_data['ultrasonic_distance'] < 30.0,
            'left_ir': sensor_data['left_ir'],
            'right_ir': sensor_data['right_ir'],
            'front_ir': sensor_data['front_ir'],
            'timestamp': sensor_data['timestamp']
        }
    
    def calibrate_sensors(self, duration: int = 10):
        """
        Calibrate sensors by collecting baseline readings.
        
        Args:
            duration (int): Calibration duration in seconds
        """
        print(f"Calibrating sensors for {duration} seconds...")
        print("Move objects around sensors during calibration")
        
        ultrasonic_readings = []
        ir_readings = {'left': [], 'right': [], 'front': []}
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Collect ultrasonic readings
            distance = self.read_ultrasonic()
            if distance != float('inf'):
                ultrasonic_readings.append(distance)
            
            # Collect IR readings
            ir_data = self.read_ir_sensors()
            ir_readings['left'].append(ir_data['left'])
            ir_readings['right'].append(ir_data['right'])
            ir_readings['front'].append(ir_data['front'])
            
            time.sleep(0.1)
        
        # Calculate calibration values
        if ultrasonic_readings:
            ultrasonic_mean = np.mean(ultrasonic_readings)
            ultrasonic_std = np.std(ultrasonic_readings)
            print(f"Ultrasonic calibration: mean={ultrasonic_mean:.1f}cm, std={ultrasonic_std:.1f}cm")
        
        for sensor in ['left', 'right', 'front']:
            if ir_readings[sensor]:
                ir_mean = np.mean(ir_readings[sensor])
                print(f"IR {sensor} calibration: mean={ir_mean:.2f}")
        
        print("Sensor calibration complete")
    
    def get_status(self) -> Dict:
        """Get current sensor manager status."""
        return {
            'initialized': self.is_initialized,
            'config': self.config,
            'last_readings': self.last_ir_readings
        }
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        GPIO.cleanup()
        self.is_initialized = False
        print("Sensor Manager cleaned up")


def test_ultrasonic():
    """Test ultrasonic sensor."""
    print("=== Ultrasonic Sensor Test ===")
    sensor_manager = SensorManager()
    
    try:
        print("Testing ultrasonic sensor...")
        print("Place objects at different distances")
        print("Press Ctrl+C to stop")
        
        while True:
            distance = sensor_manager.read_ultrasonic()
            if distance == float('inf'):
                print("Ultrasonic: No reading")
            else:
                print(f"Ultrasonic: {distance:.1f} cm")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        sensor_manager.cleanup()


def test_ir_sensors():
    """Test IR sensors."""
    print("=== IR Sensors Test ===")
    sensor_manager = SensorManager()
    
    try:
        print("Testing IR sensors...")
        print("Place objects near sensors to test detection")
        print("Press Ctrl+C to stop")
        
        while True:
            ir_readings = sensor_manager.read_ir_sensors()
            print(f"IR - Left: {ir_readings['left']}, Right: {ir_readings['right']}, Front: {ir_readings['front']}")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        sensor_manager.cleanup()


def test_all_sensors():
    """Test all sensors."""
    print("=== All Sensors Test ===")
    sensor_manager = SensorManager()
    
    try:
        print("Testing all sensors...")
        print("Press Ctrl+C to stop")
        
        while True:
            sensor_data = sensor_manager.read_all_sensors()
            print(f"Distance: {sensor_data['ultrasonic_distance']:.1f}cm, "
                  f"IR L:{sensor_data['left_ir']} R:{sensor_data['right_ir']} F:{sensor_data['front_ir']}")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("Test stopped by user")
    finally:
        sensor_manager.cleanup()


def calibrate_sensors():
    """Calibrate all sensors."""
    print("=== Sensor Calibration ===")
    sensor_manager = SensorManager()
    
    try:
        sensor_manager.calibrate_sensors(duration=10)
    finally:
        sensor_manager.cleanup()


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Sensor Manager')
    parser.add_argument('--test-ultrasonic', action='store_true',
                       help='Test ultrasonic sensor')
    parser.add_argument('--test-ir', action='store_true',
                       help='Test IR sensors')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all sensors')
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate sensors')
    parser.add_argument('--config', default='sensor_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.test_ultrasonic:
        test_ultrasonic()
    elif args.test_ir:
        test_ir_sensors()
    elif args.test_all:
        test_all_sensors()
    elif args.calibrate:
        calibrate_sensors()
    else:
        print("Sensor Manager ready for use")
        print("Use --test-ultrasonic, --test-ir, --test-all, or --calibrate to test components")


if __name__ == "__main__":
    main()
