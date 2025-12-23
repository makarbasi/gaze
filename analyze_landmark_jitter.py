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

def load_gaze_inputs(left_eye_dir, right_eye_dir, face_dir):
    """Load all gaze model input images from the directories"""
    left_eye_images = []
    right_eye_images = []
    face_images = []
    
    # Load left eye images
    left_filenames = sorted([f for f in os.listdir(left_eye_dir) if f.endswith('.png')])
    for filename in left_filenames:
        img_path = os.path.join(left_eye_dir, filename)
        img = np.array(Image.open(img_path))
        left_eye_images.append(img)
    
    # Load right eye images
    right_filenames = sorted([f for f in os.listdir(right_eye_dir) if f.endswith('.png')])
    for filename in right_filenames:
        img_path = os.path.join(right_eye_dir, filename)
        img = np.array(Image.open(img_path))
        right_eye_images.append(img)
    
    # Load face images
    face_filenames = sorted([f for f in os.listdir(face_dir) if f.endswith('.png')])
    for filename in face_filenames:
        img_path = os.path.join(face_dir, filename)
        img = np.array(Image.open(img_path))
        face_images.append(img)
    
    return left_eye_images, right_eye_images, face_images

def analyze_input_stability(images, region_name):
    """Analyze how stable the input images are for a specific region"""
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
        'region': region_name,
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

def plot_results(input_stats_dict, landmark_stats, output_dir):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Gaze input stability summary
    if input_stats_dict:
        ax = axes[0, 0]
        stats_text = "Gaze Model Input Stability:\n\n"
        for region_name, stats in input_stats_dict.items():
            stats_text += f"{region_name.upper()}:\n"
            stats_text += f"  Mean Pixel Variance: {stats['mean_pixel_variance']:.2f}\n"
            stats_text += f"  Max Pixel Variance: {stats['max_pixel_variance']:.2f}\n"
            stats_text += f"  Mean Frame Diff: {stats['mean_frame_diff']:.2f}\n"
            stats_text += f"  Max Frame Diff: {stats['max_frame_diff']:.2f}\n\n"
        stats_text += f"Frames: {list(input_stats_dict.values())[0]['num_frames']}"
        ax.text(0.5, 0.5, stats_text,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Gaze Model Input Stability')
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
    
    left_eye_dir = os.path.join(log_dir, 'left_eye')
    right_eye_dir = os.path.join(log_dir, 'right_eye')
    face_dir = os.path.join(log_dir, 'face')
    csv_file = os.path.join(log_dir, 'landmark_outputs.csv')
    
    if not os.path.exists(left_eye_dir):
        print(f"Error: Left eye directory not found: {left_eye_dir}")
        sys.exit(1)
    if not os.path.exists(right_eye_dir):
        print(f"Error: Right eye directory not found: {right_eye_dir}")
        sys.exit(1)
    if not os.path.exists(face_dir):
        print(f"Error: Face directory not found: {face_dir}")
        sys.exit(1)
    if not os.path.exists(csv_file):
        print(f"Error: Landmark outputs CSV not found: {csv_file}")
        sys.exit(1)
    
    print("Loading gaze model inputs...")
    left_eye_images, right_eye_images, face_images = load_gaze_inputs(left_eye_dir, right_eye_dir, face_dir)
    print(f"Loaded {len(left_eye_images)} left eye images")
    print(f"Loaded {len(right_eye_images)} right eye images")
    print(f"Loaded {len(face_images)} face images")
    
    print("\nAnalyzing input stability...")
    input_stats_dict = {}
    
    left_stats = analyze_input_stability(left_eye_images, 'left_eye')
    if left_stats:
        input_stats_dict['left_eye'] = left_stats
        print(f"\nLeft Eye:")
        print(f"  Mean pixel variance: {left_stats['mean_pixel_variance']:.2f}")
        print(f"  Max pixel variance: {left_stats['max_pixel_variance']:.2f}")
        print(f"  Mean frame difference: {left_stats['mean_frame_diff']:.2f}")
        print(f"  Max frame difference: {left_stats['max_frame_diff']:.2f}")
    
    right_stats = analyze_input_stability(right_eye_images, 'right_eye')
    if right_stats:
        input_stats_dict['right_eye'] = right_stats
        print(f"\nRight Eye:")
        print(f"  Mean pixel variance: {right_stats['mean_pixel_variance']:.2f}")
        print(f"  Max pixel variance: {right_stats['max_pixel_variance']:.2f}")
        print(f"  Mean frame difference: {right_stats['mean_frame_diff']:.2f}")
        print(f"  Max frame difference: {right_stats['max_frame_diff']:.2f}")
    
    face_stats = analyze_input_stability(face_images, 'face')
    if face_stats:
        input_stats_dict['face'] = face_stats
        print(f"\nFace:")
        print(f"  Mean pixel variance: {face_stats['mean_pixel_variance']:.2f}")
        print(f"  Max pixel variance: {face_stats['max_pixel_variance']:.2f}")
        print(f"  Mean frame difference: {face_stats['mean_frame_diff']:.2f}")
        print(f"  Max frame difference: {face_stats['max_frame_diff']:.2f}")
    
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
    plot_results(input_stats_dict, landmark_stats, log_dir)
    
    print("\n=== Summary ===")
    if input_stats_dict:
        for region_name, stats in input_stats_dict.items():
            stability = 'STABLE' if stats['mean_frame_diff'] < 5.0 else 'UNSTABLE'
            print(f"{region_name.upper()} Stability: {stability} (mean diff: {stats['mean_frame_diff']:.2f})")
    print(f"Landmark Jitter: {'LOW' if landmark_stats and landmark_stats['overall_std'] < 1.0 else 'HIGH'}")
    print(f"\nFull analysis saved to: {log_dir}")

if __name__ == '__main__':
    main()

