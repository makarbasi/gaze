#!/usr/bin/env python3
"""
Analyze landmark model input and output jitter

This script analyzes the logged landmark data to determine:
1. How much the input images vary (stability)
2. How much the output landmarks jitter

Usage:
    python analyze_landmark_jitter.py <path_to_log_directory>

Example:
    python analyze_landmark_jitter.py /path/to/landmark_logs/session_20250101_120000
"""

import os
import sys
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def load_input_images(input_images_dir):
    """Load all input images from the directory"""
    images = []
    filenames = sorted([f for f in os.listdir(input_images_dir) if f.endswith('.png')])
    
    for filename in filenames:
        img_path = os.path.join(input_images_dir, filename)
        img = np.array(Image.open(img_path))
        images.append(img)
    
    return images, filenames

def analyze_input_stability(images):
    """Analyze how stable the input images are"""
    if len(images) < 2:
        return None
    
    images_array = np.array(images)
    
    # Calculate pixel-wise variance across all frames
    pixel_variance = np.var(images_array, axis=0)
    mean_variance = np.mean(pixel_variance)
    max_variance = np.max(pixel_variance)
    
    # Calculate frame-to-frame differences
    frame_diffs = []
    for i in range(len(images) - 1):
        diff = np.abs(images[i+1].astype(float) - images[i].astype(float))
        frame_diffs.append(np.mean(diff))
    
    mean_frame_diff = np.mean(frame_diffs)
    max_frame_diff = np.max(frame_diffs)
    
    return {
        'mean_pixel_variance': mean_variance,
        'max_pixel_variance': max_variance,
        'mean_frame_diff': mean_frame_diff,
        'max_frame_diff': max_frame_diff,
        'num_frames': len(images)
    }

def load_landmark_outputs(csv_file):
    """Load landmark outputs from CSV"""
    landmarks = []
    timestamps = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = float(row['timestamp_ms'])
            landmark_values = [float(row[f'landmark_{i}']) for i in range(136)]
            landmarks.append(landmark_values)
            timestamps.append(timestamp)
    
    return np.array(landmarks), np.array(timestamps)

def analyze_landmark_jitter(landmarks):
    """Analyze how much the landmarks jitter"""
    if len(landmarks) < 2:
        return None
    
    # Calculate standard deviation for each landmark coordinate
    landmark_std = np.std(landmarks, axis=0)
    
    # Calculate frame-to-frame differences
    frame_diffs = []
    for i in range(len(landmarks) - 1):
        diff = np.abs(landmarks[i+1] - landmarks[i])
        frame_diffs.append(diff)
    
    frame_diffs = np.array(frame_diffs)
    mean_frame_diff = np.mean(frame_diffs, axis=0)
    max_frame_diff = np.max(frame_diffs, axis=0)
    
    # Calculate overall jitter metrics
    overall_std = np.mean(landmark_std)
    overall_max_std = np.max(landmark_std)
    overall_mean_diff = np.mean(mean_frame_diff)
    overall_max_diff = np.max(max_frame_diff)
    
    return {
        'overall_std': overall_std,
        'overall_max_std': overall_max_std,
        'overall_mean_frame_diff': overall_mean_diff,
        'overall_max_frame_diff': overall_max_diff,
        'per_landmark_std': landmark_std,
        'per_landmark_mean_diff': mean_frame_diff,
        'per_landmark_max_diff': max_frame_diff,
        'num_frames': len(landmarks)
    }

def plot_results(input_stats, landmark_stats, output_dir):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Input image variance heatmap
    if input_stats:
        ax = axes[0, 0]
        ax.text(0.5, 0.5, f"Input Stability:\n"
                          f"Mean Pixel Variance: {input_stats['mean_pixel_variance']:.2f}\n"
                          f"Max Pixel Variance: {input_stats['max_pixel_variance']:.2f}\n"
                          f"Mean Frame Diff: {input_stats['mean_frame_diff']:.2f}\n"
                          f"Max Frame Diff: {input_stats['max_frame_diff']:.2f}\n"
                          f"Frames: {input_stats['num_frames']}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Input Image Stability')
        ax.axis('off')
    
    # Plot 2: Landmark jitter summary
    if landmark_stats:
        ax = axes[0, 1]
        ax.text(0.5, 0.5, f"Landmark Jitter:\n"
                          f"Overall Std Dev: {landmark_stats['overall_std']:.4f}\n"
                          f"Max Std Dev: {landmark_stats['overall_max_std']:.4f}\n"
                          f"Mean Frame Diff: {landmark_stats['overall_mean_frame_diff']:.4f}\n"
                          f"Max Frame Diff: {landmark_stats['overall_max_frame_diff']:.4f}\n"
                          f"Frames: {landmark_stats['num_frames']}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_title('Landmark Output Jitter')
        ax.axis('off')
    
    # Plot 3: Landmark standard deviation per coordinate
    if landmark_stats:
        ax = axes[1, 0]
        ax.plot(landmark_stats['per_landmark_std'])
        ax.set_xlabel('Landmark Coordinate Index')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Jitter per Landmark Coordinate')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Frame-to-frame differences over time
    if landmark_stats:
        ax = axes[1, 1]
        # Calculate frame-to-frame differences
        frame_diffs = []
        landmarks = load_landmark_outputs(os.path.join(output_dir, 'landmark_outputs.csv'))[0]
        for i in range(len(landmarks) - 1):
            diff = np.linalg.norm(landmarks[i+1] - landmarks[i])
            frame_diffs.append(diff)
        
        ax.plot(frame_diffs)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Euclidean Distance')
        ax.set_title('Frame-to-Frame Landmark Change')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'jitter_analysis.png')
    plt.savefig(output_file, dpi=150)
    print(f"Analysis plot saved to: {output_file}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_landmark_jitter.py <path_to_log_directory>")
        print("Example: python analyze_landmark_jitter.py /path/to/landmark_logs/session_20250101_120000")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)
    
    input_images_dir = os.path.join(log_dir, 'input_images')
    csv_file = os.path.join(log_dir, 'landmark_outputs.csv')
    
    if not os.path.exists(input_images_dir):
        print(f"Error: Input images directory not found: {input_images_dir}")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"Error: Landmark outputs CSV not found: {csv_file}")
        sys.exit(1)
    
    print("Loading input images...")
    images, filenames = load_input_images(input_images_dir)
    print(f"Loaded {len(images)} input images")
    
    print("Analyzing input stability...")
    input_stats = analyze_input_stability(images)
    if input_stats:
        print(f"  Mean pixel variance: {input_stats['mean_pixel_variance']:.2f}")
        print(f"  Max pixel variance: {input_stats['max_pixel_variance']:.2f}")
        print(f"  Mean frame difference: {input_stats['mean_frame_diff']:.2f}")
        print(f"  Max frame difference: {input_stats['max_frame_diff']:.2f}")
    
    print("\nLoading landmark outputs...")
    landmarks, timestamps = load_landmark_outputs(csv_file)
    print(f"Loaded {len(landmarks)} landmark outputs")
    
    print("Analyzing landmark jitter...")
    landmark_stats = analyze_landmark_jitter(landmarks)
    if landmark_stats:
        print(f"  Overall std dev: {landmark_stats['overall_std']:.4f}")
        print(f"  Max std dev: {landmark_stats['overall_max_std']:.4f}")
        print(f"  Mean frame difference: {landmark_stats['overall_mean_frame_diff']:.4f}")
        print(f"  Max frame difference: {landmark_stats['overall_max_frame_diff']:.4f}")
    
    print("\nGenerating plots...")
    plot_results(input_stats, landmark_stats, log_dir)
    
    print("\n=== Summary ===")
    print(f"Input Stability: {'STABLE' if input_stats and input_stats['mean_frame_diff'] < 5.0 else 'UNSTABLE'}")
    print(f"Landmark Jitter: {'LOW' if landmark_stats and landmark_stats['overall_std'] < 1.0 else 'HIGH'}")
    print(f"\nFull analysis saved to: {log_dir}")

if __name__ == '__main__':
    main()

