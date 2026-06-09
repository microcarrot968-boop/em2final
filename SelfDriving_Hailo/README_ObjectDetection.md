# Smart Vision System - Object Detection Pipeline

This project provides a modular object detection system using TensorFlow Lite on Raspberry Pi 5, with comprehensive statistical analysis capabilities.

## Project Structure

```
├── setup_environment.py      # Environment setup and package installation
├── model_manager.py          # Model download and management
├── object_detector.py        # TensorFlow Lite object detection
├── statistical_analysis.py   # Statistical analysis (Week 6-8)
├── main_detection.py         # Main pipeline orchestration
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── Project2_SmartVision_ObjectDetection_Hailo.ipynb  # Original notebook
```

## Quick Start

### 1. Environment Setup

```bash
# Install system dependencies (run in terminal)
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-picamera2 rpicam-apps

# Test camera
rpicam-hello -t 3000

# Optional: Install Hailo AI HAT support
sudo apt install -y hailo-all
sudo reboot
```

### 2. Python Environment

```bash
# Install Python packages
python3 setup_environment.py

# Or install manually
pip install -r requirements.txt
```

### 3. Run Object Detection

#### Manual Mode (Press Enter for each frame)
```bash
python3 main_detection.py --mode manual --light-level high --distance 15.0 --ground-truth person
```

#### Automatic Mode (Timed capture)
```bash
python3 main_detection.py --mode auto --frames 100 --interval 0.2 --light-level low --distance 10.0 --ground-truth car
```

#### With Statistical Analysis
```bash
python3 main_detection.py --mode auto --frames 50 --analyze
```

## Usage Examples

### Basic Detection
```bash
# Manual detection with high lighting, 15cm distance, looking for a person
python3 main_detection.py --mode manual --light-level high --distance 15.0 --ground-truth person

# Automatic detection: 100 frames, 0.2s intervals, low lighting, looking for a car
python3 main_detection.py --mode auto --frames 100 --interval 0.2 --light-level low --ground-truth car
```

### Statistical Analysis
```bash
# Run detection and analysis together
python3 main_detection.py --mode auto --frames 200 --analyze

# Or analyze existing data
python3 statistical_analysis.py
```

### Individual Module Usage

#### Model Management
```bash
python3 model_manager.py
```

#### Environment Setup
```bash
python3 setup_environment.py
```

#### Statistical Analysis Only
```bash
python3 statistical_analysis.py
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Detection mode (manual/auto) | manual |
| `--frames` | Number of frames for auto mode | 100 |
| `--interval` | Interval between frames (seconds) | 0.2 |
| `--light-level` | Lighting condition (low/medium/high) | unknown |
| `--distance` | Distance to object (cm) | NaN |
| `--ground-truth` | Ground truth label | None |
| `--analyze` | Run statistical analysis | False |
| `--model-dir` | Models directory | models |
| `--csv-path` | CSV log file path | run_log.csv |

## Output Files

- `run_log.csv`: Raw detection data with timestamps, predictions, accuracy, etc.
- `run_log_clean.csv`: Cleaned data with quality control flags
- `models/`: Downloaded TensorFlow Lite models and labels

## Statistical Analysis Features

### Week 6: Discrete Distributions
- **Binomial**: Number of correct predictions per window
- **Geometric**: Trials until first success
- **Poisson**: Detections per time window

### Week 7: Continuous Distributions
- **Confidence**: Beta vs Normal distribution fitting
- **Inference Time**: Exponential vs Weibull fitting
- **Distance**: Gaussian vs Lognormal fitting (if available)
- **Model Selection**: AIC/BIC criteria for best fit

### Week 8: Joint Distributions
- **Correlations**: Between confidence, inference time, distance
- **Conditional Probabilities**: Success rates under different conditions
- **Joint Plots**: Scatter plots with marginal distributions

## Troubleshooting

### Camera Issues
```bash
# Test camera functionality
rpicam-hello -t 3000

# Check camera permissions
sudo usermod -a -G video $USER
```

### Package Installation Issues
```bash
# For Picamera2 issues
sudo apt install -y python3-picamera2

# For TensorFlow Lite issues
pip install --upgrade tflite-runtime
```

### Model Download Issues
```bash
# Check internet connection
ping google.com

# Manual model download
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

## Hardware Requirements

- **Raspberry Pi 5** (64-bit Bookworm)
- **AI Camera** (compatible with Picamera2)
- **Optional**: AI HAT (Hailo-8/8L) for accelerated inference
- **Storage**: At least 2GB free space for models and data
- **Memory**: 4GB+ RAM recommended

## Performance Notes

- **Inference Time**: Typically 50-200ms per frame on Pi 5
- **Memory Usage**: ~200MB for model and runtime
- **Storage**: ~50MB for models, varies for data logs
- **CPU Usage**: 20-40% during detection

## Data Format

The CSV log contains the following columns:
- `timestamp`: ISO format timestamp
- `predicted_label`: Detected object class
- `is_correct`: 1 if correct, 0 if incorrect
- `confidence`: Detection confidence (0-1)
- `inference_time_ms`: Processing time in milliseconds
- `light_level`: Lighting condition label
- `distance_cm`: Distance to object in centimeters

## License

This project is part of the EF Engineering Math 2 course at KENTECH.
