# Project 3: Self-Driving RC Car System

A comprehensive self-driving RC car implementation that combines computer vision, sensor fusion, and autonomous navigation. This project builds upon Project 2's Smart Vision System and adds motor control, servo steering, and intelligent decision-making capabilities.

## 🚗 System Overview

The self-driving RC car system consists of multiple integrated modules that work together to provide autonomous navigation capabilities:

### Hardware Components
- **Raspberry Pi 5** (64-bit Bookworm) - Main controller
- **AI Camera** - Computer vision input
- **Motor Controller** - DC motor speed control
- **Servo Motor** - Steering control
- **Ultrasonic Sensors** - Distance measurement
- **IR Sensors** - Line following and obstacle detection
- **Buzzer** - Audio feedback
- **OLED Display** - Status information

### Software Modules
- **`rc_car_controller.py`** - Motor and servo control
- **`sensor_manager.py`** - Sensor data collection and processing
- **`navigation_system.py`** - Path planning and decision making
- **`self_driving_car.py`** - Main autonomous driving system
- **`calibration_tools.py`** - System calibration utilities
- **`safety_monitor.py`** - Safety features and emergency stops

## 🔧 Hardware Setup

### Pin Configuration

| Component | GPIO Pin | Function |
|-----------|----------|----------|
| **Motor PWM** | GPIO 18 | DC Motor Speed Control |
| **Motor Direction** | GPIO 19 | Motor Direction (Forward/Reverse) |
| **Servo PWM** | GPIO 13 | Steering Control |
| **Ultrasonic Trigger** | GPIO 23 | Distance Sensor Trigger |
| **Ultrasonic Echo** | GPIO 24 | Distance Sensor Echo |
| **Left IR Sensor** | GPIO 9 | Left Line Detection |
| **Right IR Sensor** | GPIO 10 | Right Line Detection |
| **Front IR Sensor** | GPIO 25 | Front Obstacle Detection |
| **Buzzer** | GPIO 12 | Audio Feedback |
| **Emergency Button** | GPIO 21 | Emergency Stop |
| **I2C SDA** | GPIO 2 | OLED Display Data |
| **I2C SCL** | GPIO 3 | OLED Display Clock |

### Power Requirements
- **Raspberry Pi 5**: 5V/3A (USB-C)
- **Motors**: 6V-12V (separate battery pack)
- **Servo**: 5V-6V (can share Pi power)
- **Sensors**: 3.3V-5V (Pi GPIO)

## 🚀 Quick Start Guide

### Step 1: Hardware Assembly
1. Mount Raspberry Pi 5 on RC car chassis
2. Connect camera module to Pi
3. Wire motor controller to GPIO pins
4. Install servo motor for steering
5. Mount ultrasonic and IR sensors
6. Connect buzzer and OLED display
7. Install emergency stop button

### Step 2: Software Installation
```bash
# Install system dependencies
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-picamera2 rpicam-apps python3-rpi.gpio

# Enable I2C and PWM interfaces
sudo raspi-config  # Navigate to Interface Options

# Install Python packages
pip install -r requirements.txt
```

### Step 3: System Calibration
```bash
# Run calibration tools
python3 calibration_tools.py --calibrate-all
```

### Step 4: Test Individual Components
```bash
# Test motor control
python3 rc_car_controller.py --test-motors

# Test sensors
python3 sensor_manager.py --test-all

# Test safety systems
python3 safety_monitor.py --test-safety
```

### Step 5: Run Self-Driving System
```bash
# Autonomous driving mode
python3 self_driving_car.py --mode autonomous --speed 0.5 --duration 60

# Line following mode
python3 self_driving_car.py --mode line-following --speed 0.3 --duration 60

# Obstacle avoidance mode
python3 self_driving_car.py --mode obstacle-avoidance --speed 0.4 --duration 60
```

## 📊 Key Features

### 1. Object Detection & Avoidance
- Uses TensorFlow Lite models to detect and avoid obstacles
- Real-time object classification with confidence scoring
- Adaptive avoidance strategies based on object type

### 2. Line Following
- Follows colored lines using IR sensors
- Adaptive steering based on line position
- Recovery mechanisms for lost lines

### 3. Autonomous Navigation
- Combines multiple sensors for intelligent path planning
- Real-time decision making with safety constraints
- Multiple driving modes for different scenarios

### 4. Safety Systems
- Emergency stop functionality
- Collision avoidance with multiple thresholds
- Fail-safe mechanisms and error recovery
- Comprehensive safety monitoring

### 5. Real-time Control
- Low-latency motor and servo control
- 10Hz control loop for responsive navigation
- Hardware PWM for smooth operation

### 6. Data Logging
- Comprehensive logging of driving decisions
- Sensor readings and performance metrics
- Safety events and system status
- Statistical analysis capabilities

### 7. Calibration Tools
- Easy setup and tuning of all system parameters
- Automated calibration procedures
- Performance testing and validation

## 🎯 Operating Modes

### Autonomous Mode
- General-purpose autonomous driving
- Combines obstacle avoidance and path planning
- Adaptive speed control based on environment

### Line Following Mode
- Specialized for following marked paths
- Uses IR sensors for line detection
- Optimized for track-based navigation

### Obstacle Avoidance Mode
- Focused on obstacle detection and avoidance
- Reactive navigation based on sensor inputs
- Emergency maneuvers for collision prevention

## 📈 Data Collection & Analysis

The system automatically logs:
- **Driving decisions** (steering angle, speed, direction)
- **Sensor readings** (distance, IR values, camera detections)
- **Performance metrics** (speed, accuracy, response time)
- **Safety events** (emergency stops, near-collisions)

### Analysis Tools
```bash
# Analyze driving data
python3 statistical_analysis.py --driving-data

# Generate performance report
python3 generate_report.py --input driving_log.csv
```

## ⚠️ Safety Guidelines

1. **Always test in a safe, controlled environment**
2. **Start with low speeds** (0.1-0.3) for initial testing
3. **Keep emergency stop accessible** (Ctrl+C or physical button)
4. **Monitor battery levels** to prevent sudden stops
5. **Check sensor readings** before autonomous operation
6. **Have manual override** capability at all times

## 🔍 Troubleshooting

### Common Issues

**Motor not responding**
- Check wiring and power supply
- Verify GPIO pin connections
- Test with calibration tools

**Servo jittering**
- Adjust PWM frequency and power
- Check servo power supply
- Verify PWM signal quality

**Sensors giving wrong readings**
- Recalibrate sensors
- Check sensor power and connections
- Verify sensor mounting

**Camera not working**
- Check camera connection and permissions
- Test with `rpicam-hello -t 3000`
- Verify camera module compatibility

**System freezing**
- Check power supply and thermal management
- Monitor system resources with `htop`
- Verify GPIO access permissions

### Debug Commands
```bash
# Check GPIO status
python3 -c "import RPi.GPIO as GPIO; print('GPIO available')"

# Test camera
rpicam-hello -t 3000

# Check I2C devices
sudo i2cdetect -y 1

# Monitor system resources
htop

# Check logs
tail -f navigation_log.csv
```

## 📈 Performance Optimization

### Speed Optimization
- Use hardware PWM for smoother motor control
- Optimize object detection model (use quantized models)
- Implement sensor fusion for faster decision making

### Accuracy Improvement
- Calibrate sensors regularly
- Use multiple sensor readings for validation
- Implement adaptive thresholds based on lighting conditions

## 🎓 Learning Objectives

By completing this project, students will:
1. **Understand autonomous vehicle systems** and their components
2. **Learn sensor fusion** techniques for robust navigation
3. **Implement real-time control systems** with feedback loops
4. **Apply computer vision** to practical robotics applications
5. **Develop safety-critical software** with proper error handling
6. **Analyze system performance** using statistical methods

## 📚 Project Structure

```
SelfDriving_Hailo/
├── Project3_SelfDriving_RC_Car.ipynb    # Main instruction notebook
├── rc_car_controller.py                 # Motor and servo control
├── sensor_manager.py                     # Sensor data collection
├── navigation_system.py                 # Path planning and decisions
├── self_driving_car.py                  # Main autonomous system
├── calibration_tools.py                 # System calibration
├── safety_monitor.py                    # Safety features
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── config/                              # Configuration files
    ├── rc_car_config.json
    ├── sensor_config.json
    ├── navigation_config.json
    └── safety_config.json
```

## 🔧 Configuration Files

The system uses JSON configuration files for easy customization:

- **`rc_car_config.json`** - Motor and servo settings
- **`sensor_config.json`** - Sensor parameters and thresholds
- **`navigation_config.json`** - Navigation and safety parameters
- **`safety_config.json`** - Safety monitoring settings

## 📊 Output Files

- **`navigation_log.csv`** - Navigation decisions and sensor data
- **`safety_log.csv`** - Safety events and violations
- **`driving_log.csv`** - Complete driving session data
- **`sensor_calibration.json`** - Sensor calibration data

## 🚀 Advanced Features

### Custom Navigation Algorithms
- Implement your own path planning algorithms
- Add machine learning-based decision making
- Create custom sensor fusion techniques

### Integration with Project 2
- Use object detection from Project 2 for enhanced navigation
- Integrate statistical analysis for performance optimization
- Combine vision and sensor data for robust navigation

### Extensions and Modifications
- Add GPS navigation capabilities
- Implement wireless communication for remote monitoring
- Create mobile app for real-time control
- Add voice control and feedback

## 📚 Additional Resources

- [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/usage/gpio/)
- [TensorFlow Lite on Raspberry Pi](https://www.tensorflow.org/lite/guide/python)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Autonomous Vehicle Control Theory](https://en.wikipedia.org/wiki/Autonomous_car)
- [ROS (Robot Operating System)](https://www.ros.org/)

## 🤝 Contributing

This project is part of the EF Engineering Math 2 course at KENTECH. Contributions and improvements are welcome!

## 📄 License

This project is part of the EF Engineering Math 2 course at KENTECH.

---

> **Note**: This project builds upon Project 2's Smart Vision System. Make sure to complete Project 2 first to understand the object detection pipeline.

> **Safety First**: Always test in a controlled environment and have emergency stop capabilities readily available.
