#!/usr/bin/env python3
"""
Calibration Tools Module
Provides calibration utilities for the self-driving RC car system.
"""

import argparse
import sys
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from rc_car_controller import RCCarController
from sensor_manager import SensorManager


class CalibrationTools:
    """Calibration tools for the RC car system."""
    
    def __init__(self):
        """Initialize calibration tools."""
        self.car_controller = RCCarController()
        self.sensor_manager = SensorManager()
        
        print("Calibration Tools initialized")
    
    def calibrate_motors(self):
        """Calibrate motor speed and direction."""
        print("=== Motor Calibration ===")
        print("This will test motor speeds and directions")
        print("Make sure the car is elevated and wheels can spin freely")
        print("Press Enter to continue...")
        input()
        
        try:
            # Test forward speeds
            print("Testing forward speeds...")
            speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            for speed in speeds:
                print(f"Testing forward speed: {speed}")
                self.car_controller.set_motor_speed(speed, 'forward')
                time.sleep(2)
                self.car_controller.stop_motor()
                time.sleep(0.5)
            
            # Test reverse speeds
            print("Testing reverse speeds...")
            for speed in speeds:
                print(f"Testing reverse speed: {speed}")
                self.car_controller.set_motor_speed(speed, 'reverse')
                time.sleep(2)
                self.car_controller.stop_motor()
                time.sleep(0.5)
            
            print("Motor calibration complete!")
            
        except KeyboardInterrupt:
            print("Motor calibration interrupted")
        finally:
            self.car_controller.stop_motor()
    
    def calibrate_servo(self):
        """Calibrate servo steering."""
        print("=== Servo Calibration ===")
        print("This will test servo steering angles")
        print("Make sure the steering mechanism is free to move")
        print("Press Enter to continue...")
        input()
        
        try:
            # Test center position
            print("Testing center position...")
            self.car_controller.center_steering()
            time.sleep(2)
            
            # Test left turns
            print("Testing left turns...")
            left_angles = [-0.2, -0.4, -0.6, -0.8, -1.0]
            for angle in left_angles:
                print(f"Testing left angle: {angle}")
                self.car_controller.set_steering(angle)
                time.sleep(2)
            
            # Test right turns
            print("Testing right turns...")
            right_angles = [0.2, 0.4, 0.6, 0.8, 1.0]
            for angle in right_angles:
                print(f"Testing right angle: {angle}")
                self.car_controller.set_steering(angle)
                time.sleep(2)
            
            # Return to center
            print("Returning to center...")
            self.car_controller.center_steering()
            time.sleep(2)
            
            print("Servo calibration complete!")
            
        except KeyboardInterrupt:
            print("Servo calibration interrupted")
        finally:
            self.car_controller.center_steering()
    
    def calibrate_sensors(self):
        """Calibrate all sensors."""
        print("=== Sensor Calibration ===")
        print("This will calibrate ultrasonic and IR sensors")
        print("Follow the instructions for each sensor type")
        print("Press Enter to continue...")
        input()
        
        try:
            # Calibrate ultrasonic sensor
            print("Calibrating ultrasonic sensor...")
            print("Place objects at known distances (10cm, 20cm, 30cm, 50cm, 100cm)")
            print("Press Enter after placing each object")
            
            distances = [10, 20, 30, 50, 100]
            ultrasonic_readings = []
            
            for distance in distances:
                print(f"Place object at {distance}cm and press Enter")
                input()
                
                readings = []
                for _ in range(10):
                    reading = self.sensor_manager.read_ultrasonic()
                    if reading != float('inf'):
                        readings.append(reading)
                    time.sleep(0.1)
                
                if readings:
                    avg_reading = np.mean(readings)
                    ultrasonic_readings.append((distance, avg_reading))
                    print(f"Expected: {distance}cm, Measured: {avg_reading:.1f}cm")
            
            # Calibrate IR sensors
            print("Calibrating IR sensors...")
            print("Place white/reflective objects near each IR sensor")
            print("Press Enter after placing objects")
            input()
            
            ir_readings = {'left': [], 'right': [], 'front': []}
            
            for _ in range(20):
                readings = self.sensor_manager.read_ir_sensors()
                ir_readings['left'].append(readings['left'])
                ir_readings['right'].append(readings['right'])
                ir_readings['front'].append(readings['front'])
                time.sleep(0.1)
            
            print("IR sensor readings:")
            for sensor, readings in ir_readings.items():
                avg_reading = np.mean(readings)
                print(f"{sensor}: {avg_reading:.2f}")
            
            # Save calibration data
            calibration_data = {
                'ultrasonic': ultrasonic_readings,
                'ir_sensors': {
                    'left': np.mean(ir_readings['left']),
                    'right': np.mean(ir_readings['right']),
                    'front': np.mean(ir_readings['front'])
                },
                'timestamp': time.time()
            }
            
            with open('sensor_calibration.json', 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
            print("Sensor calibration complete!")
            print("Calibration data saved to sensor_calibration.json")
            
        except KeyboardInterrupt:
            print("Sensor calibration interrupted")
    
    def test_line_following(self):
        """Test line following calibration."""
        print("=== Line Following Test ===")
        print("This will test line following with different line positions")
        print("Place a line (tape or marker) and position the car")
        print("Press Enter to continue...")
        input()
        
        try:
            print("Testing line following...")
            print("Move the car to different positions relative to the line")
            print("Press Ctrl+C to stop")
            
            while True:
                sensor_data = self.sensor_manager.get_navigation_data()
                line_position = sensor_data['line_position']
                
                print(f"Line position: {line_position:.2f} "
                      f"(Left IR: {sensor_data['left_ir']}, Right IR: {sensor_data['right_ir']})")
                
                if abs(line_position) > 0.1:
                    # Line detected, show steering recommendation
                    steering = -line_position * 0.6
                    print(f"Recommended steering: {steering:.2f}")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("Line following test stopped")
    
    def test_obstacle_avoidance(self):
        """Test obstacle avoidance calibration."""
        print("=== Obstacle Avoidance Test ===")
        print("This will test obstacle detection and avoidance")
        print("Place obstacles at different distances")
        print("Press Enter to continue...")
        input()
        
        try:
            print("Testing obstacle avoidance...")
            print("Place obstacles at different distances and positions")
            print("Press Ctrl+C to stop")
            
            while True:
                sensor_data = self.sensor_manager.get_navigation_data()
                distance = sensor_data['distance_front']
                ir_readings = {
                    'left': sensor_data['left_ir'],
                    'right': sensor_data['right_ir'],
                    'front': sensor_data['front_ir']
                }
                
                print(f"Distance: {distance:.1f}cm, "
                      f"IR L:{ir_readings['left']} R:{ir_readings['right']} F:{ir_readings['front']}")
                
                if distance < 30:
                    print("OBSTACLE DETECTED!")
                    if ir_readings['left'] == 0 and ir_readings['right'] == 1:
                        print("Recommendation: Turn LEFT")
                    elif ir_readings['left'] == 1 and ir_readings['right'] == 0:
                        print("Recommendation: Turn RIGHT")
                    else:
                        print("Recommendation: Turn RANDOM")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("Obstacle avoidance test stopped")
    
    def generate_calibration_report(self):
        """Generate calibration report."""
        print("=== Calibration Report ===")
        
        # Check if calibration files exist
        calibration_files = [
            'rc_car_config.json',
            'sensor_config.json',
            'navigation_config.json',
            'sensor_calibration.json'
        ]
        
        print("Calibration files status:")
        for file in calibration_files:
            if os.path.exists(file):
                print(f"✓ {file} - Found")
            else:
                print(f"✗ {file} - Missing")
        
        # Test system components
        print("\nSystem component tests:")
        
        # Test motor
        try:
            self.car_controller.set_motor_speed(0.1, 'forward')
            time.sleep(0.5)
            self.car_controller.stop_motor()
            print("✓ Motor control - Working")
        except Exception as e:
            print(f"✗ Motor control - Error: {e}")
        
        # Test servo
        try:
            self.car_controller.set_steering(0.1)
            time.sleep(0.5)
            self.car_controller.center_steering()
            print("✓ Servo control - Working")
        except Exception as e:
            print(f"✗ Servo control - Error: {e}")
        
        # Test sensors
        try:
            sensor_data = self.sensor_manager.read_all_sensors()
            print("✓ Sensor reading - Working")
        except Exception as e:
            print(f"✗ Sensor reading - Error: {e}")
        
        print("\nCalibration report complete!")
    
    def cleanup(self):
        """Cleanup resources."""
        self.car_controller.cleanup()
        self.sensor_manager.cleanup()
        print("Calibration tools cleaned up")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='RC Car Calibration Tools')
    parser.add_argument('--calibrate-motors', action='store_true',
                       help='Calibrate motors')
    parser.add_argument('--calibrate-servo', action='store_true',
                       help='Calibrate servo')
    parser.add_argument('--calibrate-sensors', action='store_true',
                       help='Calibrate sensors')
    parser.add_argument('--test-line-following', action='store_true',
                       help='Test line following')
    parser.add_argument('--test-obstacle-avoidance', action='store_true',
                       help='Test obstacle avoidance')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate calibration report')
    parser.add_argument('--calibrate-all', action='store_true',
                       help='Run all calibrations')
    
    args = parser.parse_args()
    
    tools = CalibrationTools()
    
    try:
        if args.calibrate_motors:
            tools.calibrate_motors()
        elif args.calibrate_servo:
            tools.calibrate_servo()
        elif args.calibrate_sensors:
            tools.calibrate_sensors()
        elif args.test_line_following:
            tools.test_line_following()
        elif args.test_obstacle_avoidance:
            tools.test_obstacle_avoidance()
        elif args.generate_report:
            tools.generate_calibration_report()
        elif args.calibrate_all:
            print("Running all calibrations...")
            tools.calibrate_motors()
            tools.calibrate_servo()
            tools.calibrate_sensors()
            tools.generate_calibration_report()
        else:
            print("Calibration Tools ready")
            print("Use --calibrate-motors, --calibrate-servo, --calibrate-sensors, etc.")
    finally:
        tools.cleanup()


if __name__ == "__main__":
    main()
