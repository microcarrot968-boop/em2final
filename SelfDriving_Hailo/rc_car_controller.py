#!/usr/bin/env python3
"""
RC Car Controller Module
Handles motor and servo control for the self-driving RC car.
"""

import RPi.GPIO as GPIO
import time
import argparse
import sys
from typing import Tuple, Optional
import json
import os


class RCCarController:
    """Controls motors and servo for RC car."""
    
    def __init__(self, config_file: str = 'rc_car_config.json'):
        """
        Initialize RC Car Controller.
        
        Args:
            config_file (str): Path to configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # GPIO Pin Configuration
        self.MOTOR_PWM_PIN = self.config['pins']['motor_pwm']
        self.MOTOR_DIR_PIN = self.config['pins']['motor_dir']
        self.SERVO_PWM_PIN = self.config['pins']['servo_pwm']
        self.BUZZER_PIN = self.config['pins']['buzzer']
        
        # PWM Configuration
        self.MOTOR_FREQ = self.config['pwm']['motor_frequency']
        self.SERVO_FREQ = self.config['pwm']['servo_frequency']
        
        # Servo Configuration
        self.SERVO_CENTER = self.config['servo']['center_duty']
        self.SERVO_LEFT_MAX = self.config['servo']['left_max_duty']
        self.SERVO_RIGHT_MAX = self.config['servo']['right_max_duty']
        
        # Motor Configuration
        self.MOTOR_MAX_SPEED = self.config['motor']['max_speed']
        self.MOTOR_MIN_SPEED = self.config['motor']['min_speed']
        
        # Initialize GPIO
        self.setup_gpio()
        
        # Initialize PWM
        self.motor_pwm = None
        self.servo_pwm = None
        self.setup_pwm()
        
        # Current state
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.is_initialized = True
        
        print("RC Car Controller initialized successfully!")
        print(f"Motor PWM Pin: {self.MOTOR_PWM_PIN}")
        print(f"Motor Direction Pin: {self.MOTOR_DIR_PIN}")
        print(f"Servo PWM Pin: {self.SERVO_PWM_PIN}")
        print(f"Buzzer Pin: {self.BUZZER_PIN}")
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        default_config = {
            "pins": {
                "motor_pwm": 18,
                "motor_dir": 19,
                "servo_pwm": 13,
                "buzzer": 12
            },
            "pwm": {
                "motor_frequency": 1000,
                "servo_frequency": 50
            },
            "servo": {
                "center_duty": 7.5,
                "left_max_duty": 5.0,
                "right_max_duty": 10.0
            },
            "motor": {
                "max_speed": 100,
                "min_speed": 0
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
    
    def setup_gpio(self):
        """Setup GPIO pins."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.MOTOR_PWM_PIN, GPIO.OUT)
        GPIO.setup(self.MOTOR_DIR_PIN, GPIO.OUT)
        GPIO.setup(self.SERVO_PWM_PIN, GPIO.OUT)
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        
        # Initialize outputs
        GPIO.output(self.MOTOR_DIR_PIN, GPIO.LOW)
        GPIO.output(self.BUZZER_PIN, GPIO.LOW)
    
    def setup_pwm(self):
        """Setup PWM channels."""
        self.motor_pwm = GPIO.PWM(self.MOTOR_PWM_PIN, self.MOTOR_FREQ)
        self.servo_pwm = GPIO.PWM(self.SERVO_PWM_PIN, self.SERVO_FREQ)
        
        self.motor_pwm.start(0)
        self.servo_pwm.start(self.SERVO_CENTER)
    
    def set_motor_speed(self, speed: float, direction: str = 'forward'):
        """
        Set motor speed and direction.
        
        Args:
            speed (float): Speed value between 0.0 and 1.0
            direction (str): 'forward' or 'reverse'
        """
        if not self.is_initialized:
            print("Controller not initialized!")
            return
        
        # Clamp speed to valid range
        speed = max(0.0, min(1.0, speed))
        
        # Convert to duty cycle
        duty_cycle = speed * (self.MOTOR_MAX_SPEED - self.MOTOR_MIN_SPEED) + self.MOTOR_MIN_SPEED
        
        # Set direction
        if direction.lower() == 'forward':
            GPIO.output(self.MOTOR_DIR_PIN, GPIO.HIGH)
        elif direction.lower() == 'reverse':
            GPIO.output(self.MOTOR_DIR_PIN, GPIO.LOW)
        else:
            print(f"Invalid direction: {direction}")
            return
        
        # Set PWM duty cycle
        self.motor_pwm.ChangeDutyCycle(duty_cycle)
        self.current_speed = speed
        
        print(f"Motor: {direction} at {speed:.2f} speed (duty: {duty_cycle:.1f}%)")
    
    def set_steering(self, angle: float):
        """
        Set steering angle.
        
        Args:
            angle (float): Steering angle between -1.0 (left) and 1.0 (right)
        """
        if not self.is_initialized:
            print("Controller not initialized!")
            return
        
        # Clamp angle to valid range
        angle = max(-1.0, min(1.0, angle))
        
        # Convert to duty cycle
        if angle < 0:  # Left turn
            duty_cycle = self.SERVO_CENTER + (angle * (self.SERVO_CENTER - self.SERVO_LEFT_MAX))
        else:  # Right turn
            duty_cycle = self.SERVO_CENTER + (angle * (self.SERVO_RIGHT_MAX - self.SERVO_CENTER))
        
        # Set PWM duty cycle
        self.servo_pwm.ChangeDutyCycle(duty_cycle)
        self.current_steering = angle
        
        print(f"Steering: {angle:.2f} (duty: {duty_cycle:.1f}%)")
    
    def stop_motor(self):
        """Stop the motor."""
        if self.motor_pwm:
            self.motor_pwm.ChangeDutyCycle(0)
            self.current_speed = 0.0
            print("Motor stopped")
    
    def center_steering(self):
        """Center the steering."""
        if self.servo_pwm:
            self.servo_pwm.ChangeDutyCycle(self.SERVO_CENTER)
            self.current_steering = 0.0
            print("Steering centered")
    
    def beep(self, duration: float = 0.1):
        """
        Sound the buzzer.
        
        Args:
            duration (float): Duration of beep in seconds
        """
        GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.BUZZER_PIN, GPIO.LOW)
        print(f"Beep for {duration:.1f}s")
    
    def emergency_stop(self):
        """Emergency stop - stop motor and center steering."""
        print("EMERGENCY STOP!")
        self.stop_motor()
        self.center_steering()
        self.beep(0.5)  # Long beep for emergency
    
    def get_status(self) -> dict:
        """Get current controller status."""
        return {
            'speed': self.current_speed,
            'steering': self.current_steering,
            'initialized': self.is_initialized,
            'config': self.config
        }
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self.motor_pwm:
            self.motor_pwm.stop()
        if self.servo_pwm:
            self.servo_pwm.stop()
        GPIO.cleanup()
        self.is_initialized = False
        print("RC Car Controller cleaned up")


def test_motors():
    """Test motor functionality."""
    print("=== Motor Test ===")
    controller = RCCarController()
    
    try:
        # Test forward
        print("Testing forward motion...")
        controller.set_motor_speed(0.3, 'forward')
        time.sleep(2)
        
        # Stop
        controller.stop_motor()
        time.sleep(1)
        
        # Test reverse
        print("Testing reverse motion...")
        controller.set_motor_speed(0.3, 'reverse')
        time.sleep(2)
        
        # Stop
        controller.stop_motor()
        print("Motor test complete")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        controller.cleanup()


def test_servo():
    """Test servo functionality."""
    print("=== Servo Test ===")
    controller = RCCarController()
    
    try:
        # Test center
        print("Testing center position...")
        controller.center_steering()
        time.sleep(1)
        
        # Test left turn
        print("Testing left turn...")
        controller.set_steering(-0.5)
        time.sleep(1)
        
        # Test right turn
        print("Testing right turn...")
        controller.set_steering(0.5)
        time.sleep(1)
        
        # Center again
        controller.center_steering()
        print("Servo test complete")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        controller.cleanup()


def test_buzzer():
    """Test buzzer functionality."""
    print("=== Buzzer Test ===")
    controller = RCCarController()
    
    try:
        # Test short beep
        controller.beep(0.1)
        time.sleep(0.5)
        
        # Test long beep
        controller.beep(0.5)
        time.sleep(0.5)
        
        # Test multiple beeps
        for i in range(3):
            controller.beep(0.1)
            time.sleep(0.2)
        
        print("Buzzer test complete")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        controller.cleanup()


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='RC Car Controller')
    parser.add_argument('--test-motors', action='store_true',
                       help='Test motor functionality')
    parser.add_argument('--test-servo', action='store_true',
                       help='Test servo functionality')
    parser.add_argument('--test-buzzer', action='store_true',
                       help='Test buzzer functionality')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all components')
    parser.add_argument('--config', default='rc_car_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.test_motors:
        test_motors()
    elif args.test_servo:
        test_servo()
    elif args.test_buzzer:
        test_buzzer()
    elif args.test_all:
        print("=== Complete RC Car Controller Test ===")
        test_motors()
        test_servo()
        test_buzzer()
    else:
        print("RC Car Controller ready for use")
        print("Use --test-motors, --test-servo, --test-buzzer, or --test-all to test components")


if __name__ == "__main__":
    main()
