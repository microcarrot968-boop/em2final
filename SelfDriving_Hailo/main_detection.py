#!/usr/bin/env python3
"""
Main Detection Script
Orchestrates the complete object detection pipeline.
"""

import argparse
import sys
from model_manager import ModelManager
from object_detector import ObjectDetector
from statistical_analysis import StatisticalAnalyzer


def main():
    """Main function to run the object detection pipeline."""
    parser = argparse.ArgumentParser(description='Smart Vision System - Object Detection Pipeline')
    parser.add_argument('--mode', choices=['manual', 'auto'], default='manual',
                       help='Detection mode: manual (press Enter) or auto (timed)')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames for auto mode (default: 100)')
    parser.add_argument('--interval', type=float, default=0.2,
                       help='Interval between frames in auto mode (default: 0.2s)')
    parser.add_argument('--light-level', default='unknown',
                       help='Lighting condition (low/medium/high)')
    parser.add_argument('--distance', type=float, default=float('nan'),
                       help='Distance to object in cm')
    parser.add_argument('--ground-truth', default=None,
                       help='Ground truth label for accuracy calculation')
    parser.add_argument('--analyze', action='store_true',
                       help='Run statistical analysis after detection')
    parser.add_argument('--model-dir', default='models',
                       help='Directory for models (default: models)')
    parser.add_argument('--csv-path', default='run_log.csv',
                       help='Path for CSV log file (default: run_log.csv)')
    
    args = parser.parse_args()
    
    print("=== Smart Vision System - Object Detection Pipeline ===")
    print(f"Mode: {args.mode}")
    print(f"Light level: {args.light_level}")
    print(f"Distance: {args.distance} cm")
    print(f"Ground truth: {args.ground_truth}")
    
    try:
        # Step 1: Setup model manager
        print("\n1. Setting up model manager...")
        model_manager = ModelManager(args.model_dir)
        
        # Download model if needed
        if not model_manager.download_model():
            print("Failed to download model. Exiting.")
            return 1
        
        # Get model info
        model_info = model_manager.get_model_info()
        print(f"Model loaded: {model_info['model_path']}")
        print(f"Labels loaded: {model_info['label_count']} classes")
        
        # Step 2: Initialize object detector
        print("\n2. Initializing object detector...")
        detector = ObjectDetector(
            model_path=model_info['model_path'],
            labels=model_info['labels'],
            csv_path=args.csv_path
        )
        
        # Step 3: Run detection
        print("\n3. Starting detection...")
        if args.mode == 'manual':
            detector.run_manual_detection(
                light_level=args.light_level,
                distance_cm=args.distance,
                ground_truth=args.ground_truth
            )
        else:  # auto mode
            detector.run_auto_detection(
                n_frames=args.frames,
                interval_sec=args.interval,
                light_level=args.light_level,
                distance_cm=args.distance,
                ground_truth=args.ground_truth
            )
        
        # Step 4: Run statistical analysis if requested
        if args.analyze:
            print("\n4. Running statistical analysis...")
            analyzer = StatisticalAnalyzer(args.csv_path)
            analyzer.discrete_distribution_analysis()
            analyzer.continuous_distribution_analysis()
            analyzer.joint_distribution_analysis()
            analyzer.export_cleaned_data()
        
        print("\n=== Detection Pipeline Complete ===")
        return 0
        
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
        return 0
    except Exception as e:
        print(f"\nError in detection pipeline: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
