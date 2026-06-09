#!/usr/bin/env python3
"""
Model Manager Module
Handles downloading and managing TensorFlow Lite models for object detection.
"""

import os
import urllib.request
import zipfile
from pathlib import Path


class ModelManager:
    """Manages TensorFlow Lite model download and setup."""
    
    def __init__(self, models_dir='models'):
        """
        Initialize ModelManager.
        
        Args:
            models_dir (str): Directory to store models.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs and paths
        self.model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip'
        self.model_path = self.models_dir / 'detect.tflite'
        self.label_paths = [
            self.models_dir / 'labelmap.txt',
            self.models_dir / 'coco_labels.txt'
        ]
    
    def download_model(self):
        """
        Download the MobileNet-SSD TensorFlow Lite model.
        
        Returns:
            bool: True if model is available, False otherwise.
        """
        if self.model_path.exists():
            print('Model already present.')
            return True
        
        print('Downloading MobileNet-SSD model...')
        zip_path = self.models_dir / 'coco_ssd.zip'
        
        try:
            urllib.request.urlretrieve(self.model_url, zip_path)
            print('Model downloaded successfully.')
            
            # Extract the model
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.models_dir)
            print('Model extracted successfully.')
            
            # Clean up zip file
            zip_path.unlink()
            
            return self.model_path.exists()
            
        except Exception as e:
            print(f'Error downloading model: {str(e)}')
            return False
    
    def get_label_path(self):
        """
        Find the label file for the model.
        
        Returns:
            str or None: Path to label file if found, None otherwise.
        """
        for label_path in self.label_paths:
            if label_path.exists():
                return str(label_path)
        return None
    
    def load_labels(self):
        """
        Load class labels from the label file.
        
        Returns:
            list: List of class labels.
        """
        label_path = self.get_label_path()
        
        if label_path is None:
            # Fallback COCO 80 labels (corrected from 91)
            coco_labels = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            return coco_labels
        
        try:
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except Exception as e:
            print(f'Error loading labels: {str(e)}')
            return [f'class_{i}' for i in range(80)]
    
    def get_model_info(self):
        """
        Get information about the model and labels.
        
        Returns:
            dict: Model information including paths and label count.
        """
        return {
            'model_path': str(self.model_path),
            'label_path': self.get_label_path(),
            'labels': self.load_labels(),
            'label_count': len(self.load_labels()),
            'model_exists': self.model_path.exists()
        }


if __name__ == "__main__":
    print("=== Model Manager Test ===")
    
    manager = ModelManager()
    
    # Download model
    if manager.download_model():
        print("Model download successful!")
        
        # Get model info
        info = manager.get_model_info()
        print(f"Model path: {info['model_path']}")
        print(f"Label path: {info['label_path']}")
        print(f"Number of labels: {info['label_count']}")
        print(f"First 5 labels: {info['labels'][:5]}")
    else:
        print("Model download failed!")
