#!/usr/bin/env python3
"""
Object Detection Module
Handles TensorFlow Lite object detection and data logging.
"""

import os
import csv
import time
from datetime import datetime
from typing import Tuple, Optional, Union
import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter


class ObjectDetector:
    """Handles object detection using TensorFlow Lite models."""
    
    def __init__(self, model_path: str, labels: list, csv_path: str = 'run_log.csv'):
        """
        Initialize ObjectDetector.
        
        Args:
            model_path (str): Path to TensorFlow Lite model.
            labels (list): List of class labels.
            csv_path (str): Path to CSV log file.
        """
        self.model_path = model_path
        self.labels = labels
        self.csv_path = csv_path
        
        # Initialize TensorFlow Lite interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input dimensions
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shapes: {[out['shape'] for out in self.output_details]}")
    
    def ensure_csv(self) -> Tuple[csv.writer, object]:
        """
        Ensure CSV file exists with proper headers.
        
        Returns:
            Tuple[csv.writer, object]: CSV writer and file object.
        """
        new_file = not os.path.exists(self.csv_path)
        file_obj = open(self.csv_path, 'a', newline='')
        writer = csv.writer(file_obj)
        
        if new_file:
            writer.writerow([
                'timestamp', 'predicted_label', 'is_correct', 'confidence',
                'inference_time_ms', 'light_level', 'distance_cm'
            ])
        
        return writer, file_obj
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame (np.ndarray): Input frame.
            
        Returns:
            np.ndarray: Preprocessed frame.
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        # Add batch dimension and ensure uint8
        processed = np.expand_dims(resized, 0).astype(np.uint8)
        return processed
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[str, float, float]:
        """
        Run inference on a single frame.
        
        Args:
            frame (np.ndarray): Input frame.
            
        Returns:
            Tuple[str, float, float]: (predicted_label, confidence, inference_time_ms)
        """
        # Preprocess frame
        input_data = self.preprocess_frame(frame)
        
        # Run inference
        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000.0
        
        # Get outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0].astype(int)
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Find best detection
        if len(scores) > 0 and float(np.max(scores)) > 0:
            best_idx = int(np.argmax(scores))
            class_id = classes[best_idx]
            confidence = float(scores[best_idx])
            
            # Get label
            if class_id < len(self.labels):
                label = self.labels[class_id]
            else:
                label = f'class_{class_id}'
        else:
            label = 'none'
            confidence = 0.0
        
        return label, confidence, inference_time
    
    def run_manual_detection(self, light_level: str = 'unknown', 
                           distance_cm: float = float('nan'), 
                           ground_truth: Optional[str] = None):
        """
        Run manual detection mode (press Enter for each frame).
        
        Args:
            light_level (str): Lighting condition label.
            distance_cm (float): Distance to object in cm.
            ground_truth (Optional[str]): Ground truth label for accuracy calculation.
        """
        print("Starting manual detection mode...")
        print("Press Enter to capture frame, Ctrl+C to stop")
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={'size': (640, 480), 'format': 'RGB888'}
        )
        picam2.configure(config)
        picam2.start()
        
        writer, file_obj = self.ensure_csv()
        
        try:
            while True:
                input()  # Wait for Enter key
                frame = picam2.capture_array()
                label, confidence, inference_time = self.infer_frame(frame)
                
                # Calculate accuracy
                is_correct = int(ground_truth is not None and label == ground_truth)
                
                # Log to CSV
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    label, is_correct, round(confidence, 4),
                    round(inference_time, 2), light_level, distance_cm
                ])
                file_obj.flush()
                
                print(f'Detected: {label} (confidence: {confidence:.2f}, time: {inference_time:.1f}ms)')
                
        except KeyboardInterrupt:
            print("\nManual detection stopped by user.")
        finally:
            file_obj.close()
            picam2.stop()
            print("Camera stopped and file saved.")
    
    def run_auto_detection(self, n_frames: int = 100, interval_sec: float = 0.2,
                          light_level: str = 'unknown', distance_cm: float = float('nan'),
                          ground_truth: Optional[str] = None):
        """
        Run automatic detection mode (capture frames at intervals).
        
        Args:
            n_frames (int): Number of frames to capture.
            interval_sec (float): Interval between frames in seconds.
            light_level (str): Lighting condition label.
            distance_cm (float): Distance to object in cm.
            ground_truth (Optional[str]): Ground truth label for accuracy calculation.
        """
        print(f"Starting automatic detection mode...")
        print(f"Capturing {n_frames} frames with {interval_sec}s intervals")
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={'size': (640, 480), 'format': 'RGB888'}
        )
        picam2.configure(config)
        picam2.start()
        
        writer, file_obj = self.ensure_csv()
        
        try:
            for i in range(n_frames):
                frame = picam2.capture_array()
                label, confidence, inference_time = self.infer_frame(frame)
                
                # Calculate accuracy
                is_correct = int(ground_truth is not None and label == ground_truth)
                
                # Log to CSV
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    label, is_correct, round(confidence, 4),
                    round(inference_time, 2), light_level, distance_cm
                ])
                file_obj.flush()
                
                print(f'[{i+1}/{n_frames}] Detected: {label} (confidence: {confidence:.2f}, time: {inference_time:.1f}ms)')
                
                if i < n_frames - 1:  # Don't sleep after the last frame
                    time.sleep(interval_sec)
                    
        except KeyboardInterrupt:
            print("\nAutomatic detection stopped by user.")
        finally:
            file_obj.close()
            picam2.stop()
            print("Camera stopped and file saved.")


if __name__ == "__main__":
    print("=== Object Detector Test ===")
    
    # This would typically be run with proper model and labels
    print("ObjectDetector class ready for use.")
    print("Use with ModelManager to get model path and labels.")
