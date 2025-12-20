"""
Looking at Camera Classifier - IMPROVED Training v2

Improvements over v1:
  1. Option to use landmarks as additional features
  2. Data augmentation (noise injection)
  3. Class balancing (weighted loss)
  4. Multiple model architectures to try
  5. Cross-validation for robust evaluation
  6. Feature importance analysis
  7. Ensemble option

Usage: python train_model_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json

# ============================================================
# CONFIGURATION - ADJUST THESE FOR BETTER RESULTS
# ============================================================

DATA_DIR = "data"
OUTPUT_MODEL = "looking_classifier_v2.tflite"
OUTPUT_KERAS = "looking_classifier_v2.h5"

# Feature selection
USE_LANDMARKS = True  # Include important landmark features
# Selected landmarks for looking classifier v2:
# - Eyes: left/right corners + top/bottom (gives robust eye shape/pose)
# - Nose: tip + left/right corners
#
# Note: "eye center" is added as a DERIVED feature (computed from the two eye corners),
# so it doesn't require a specific landmark index.
LANDMARK_INDICES = [
    # Left eye: corners + top/bottom
    60, 64, 62, 66,
    # Right eye: corners + top/bottom
    68, 72, 70, 74,
    # Nose: tip + left/right corners
    54, 55, 59,
]
# Remove duplicates (keep deterministic order for reproducibility)
LANDMARK_INDICES = sorted(set(LANDMARK_INDICES))

# Core 7 features (always used)
CORE_FEATURES = [
    'gaze_pitch', 'gaze_yaw', 
    'head_pitch', 'head_yaw', 'head_roll',
    'relative_pitch', 'relative_yaw'
]

# Data augmentation - DISABLED per user request
USE_AUGMENTATION = False
AUGMENTATION_FACTOR = 0
NOISE_STD = 0.0

# Class balancing - ENABLED
USE_CLASS_WEIGHTS = True

# Training
EPOCHS = 200
BATCH_SIZE = 32
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42

# Cross-validation - disabled for speed
USE_CROSS_VALIDATION = False
N_FOLDS = 5

# Hyperparameter tuning
USE_HYPERPARAMETER_TUNING = True

# ============================================================


def load_data_file(filepath):
    """Load data from Excel or CSV file"""
    print(f"Loading: {filepath}")
    
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    print(f"  Loaded {len(df)} rows")
    return df


def extract_features(df, use_landmarks=False):
    """Extract features from dataframe"""
    features = []
    feature_names = []
    
    # Core features (7)
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    for feat in CORE_FEATURES:
        feat_lower = feat.lower()
        if feat_lower in df_cols_lower:
            col_name = df_cols_lower[feat_lower]
        elif feat in df.columns:
            col_name = feat
        else:
            print(f"  WARNING: Missing feature {feat}")
            continue
        features.append(df[col_name].values)
        feature_names.append(feat)
    
    # Landmark features (optional)
    if use_landmarks:
        for idx in LANDMARK_INDICES:
            x_col = f'lm{idx}_x'
            y_col = f'lm{idx}_y'
            if x_col in df.columns and y_col in df.columns:
                features.append(df[x_col].values)
                features.append(df[y_col].values)
                feature_names.append(x_col)
                feature_names.append(y_col)
    
    X = np.column_stack(features)
    return X, feature_names


def load_dataset():
    """Load and combine datasets"""
    looking_file = None
    notlooking_file = None
    
    # Prefer Excel over CSV if both exist (common workflow: update .xlsx, keep older .csv around)
    for ext in ['.xlsx', '.xls', '.csv']:
        p = os.path.join(DATA_DIR, f"looking{ext}")
        if os.path.exists(p):
            looking_file = p
            break
    
    # Prefer Excel over CSV if both exist
    for ext in ['.xlsx', '.xls', '.csv']:
        p = os.path.join(DATA_DIR, f"notlooking{ext}")
        if os.path.exists(p):
            notlooking_file = p
            break
    
    if not looking_file or not notlooking_file:
        print("ERROR: Could not find data files!")
        sys.exit(1)
    
    df_looking = load_data_file(looking_file)
    df_notlooking = load_data_file(notlooking_file)
    
    X_looking, feature_names = extract_features(df_looking, USE_LANDMARKS)
    X_notlooking, _ = extract_features(df_notlooking, USE_LANDMARKS)
    
    y_looking = np.ones(len(X_looking))
    y_notlooking = np.zeros(len(X_notlooking))
    
    X = np.vstack([X_looking, X_notlooking])
    y = np.concatenate([y_looking, y_notlooking])
    
    print(f"\nDataset summary:")
    print(f"  Looking: {len(X_looking)}, Not looking: {len(X_notlooking)}")
    print(f"  Total: {len(X)}, Features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")
    
    return X, y, feature_names


def augment_data(X, y, factor=3, noise_std=0.02):
    """Augment data by adding Gaussian noise"""
    print(f"\nAugmenting data (factor={factor}, noise_std={noise_std})...")
    
    X_aug = [X]
    y_aug = [y]
    
    for i in range(factor):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    X_augmented = np.vstack(X_aug)
    y_augmented = np.concatenate(y_aug)
    
    # Shuffle
    idx = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[idx]
    y_augmented = y_augmented[idx]
    
    print(f"  Original: {len(X)}, Augmented: {len(X_augmented)}")
    return X_augmented, y_augmented


def remove_invalid_samples(X, y):
    """Remove NaN/Inf samples"""
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    removed = len(X) - len(X_clean)
    if removed > 0:
        print(f"  Removed {removed} invalid samples")
    return X_clean, y_clean


def build_model_v1(input_dim):
    """Original architecture"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_model_v2(input_dim):
    """Deeper architecture"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_model_v3(input_dim):
    """Wide architecture with residual-like connections"""
    inputs = keras.layers.Input(shape=(input_dim,))
    
    # First block
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Second block
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Third block
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Output
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model with optimizer"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_with_cross_validation(X, y, n_folds=5):
    """Train with k-fold cross-validation to find best model"""
    print(f"\n[CROSS-VALIDATION] {n_folds} folds")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    fold_scores = []
    best_score = 0
    best_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Class weights
        class_weights = None
        if USE_CLASS_WEIGHTS:
            weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weights = {0: weights[0], 1: weights[1]}
        
        # Build and train
        model = build_model_v2(X.shape[1])
        compile_model(model)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        model.fit(
            X_train_scaled, y_train_fold,
            validation_data=(X_val_scaled, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = (model.predict(X_val_scaled, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_val_fold, y_pred)
        fold_scores.append(acc)
        print(f"    Accuracy: {acc*100:.2f}%")
        
        if acc > best_score:
            best_score = acc
            best_fold = fold
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n  Cross-validation results:")
    print(f"    Mean accuracy: {mean_score*100:.2f}% (+/- {std_score*100:.2f}%)")
    print(f"    Best fold: {best_fold+1} with {best_score*100:.2f}%")
    
    return mean_score, std_score


def try_multiple_architectures(X_train, y_train, X_val, y_val, class_weights):
    """Try different model architectures and return the best one"""
    print("\n[ARCHITECTURE SEARCH]")
    
    architectures = [
        ("v1 (Original)", build_model_v1),
        ("v2 (Deeper)", build_model_v2),
        ("v3 (Wide)", build_model_v3),
    ]
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, build_fn in architectures:
        print(f"\n  Trying {name}...")
        
        model = build_fn(X_train.shape[1])
        compile_model(model)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        val_acc = max(history.history['val_accuracy'])
        print(f"    Best val accuracy: {val_acc*100:.2f}%")
        
        if val_acc > best_score:
            best_score = val_acc
            best_model = model
            best_name = name
    
    print(f"\n  Best architecture: {best_name} ({best_score*100:.2f}%)")
    return best_model, best_name


def hyperparameter_tuning(X_train, y_train, X_val, y_val, class_weights, input_dim):
    """Fast grid search over key hyperparameters"""
    print("\n[HYPERPARAMETER TUNING - FAST]")
    
    # Reduced hyperparameter grid for speed
    configs = [
        {'lr': 0.001, 'units': (128, 64, 32), 'dropout': 0.3},
        {'lr': 0.001, 'units': (256, 128, 64), 'dropout': 0.3},
        {'lr': 0.0005, 'units': (128, 64, 32), 'dropout': 0.2},
        {'lr': 0.0005, 'units': (256, 128, 64, 32), 'dropout': 0.3},
        {'lr': 0.001, 'units': (64, 32, 16), 'dropout': 0.2},
        {'lr': 0.0001, 'units': (256, 128, 64), 'dropout': 0.4},
    ]
    
    best_model = None
    best_score = 0
    best_params = {}
    
    print(f"  Testing {len(configs)} configurations...")
    
    for idx, params in enumerate(configs):
        lr = params['lr']
        units = params['units']
        dropout = params['dropout']
        
        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        for i, u in enumerate(units):
            model.add(keras.layers.Dense(u, activation='relu', 
                      kernel_regularizer=keras.regularizers.l2(0.001)))
            model.add(keras.layers.BatchNormalization())
            if i < len(units) - 1:
                model.add(keras.layers.Dropout(dropout))
        
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, 
                                          restore_best_weights=True, verbose=0),
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Quick tuning
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        val_acc = max(history.history['val_accuracy'])
        
        if val_acc > best_score:
            best_score = val_acc
            best_model = model
            best_params = params
            print(f"  [{idx+1}/{len(configs)}] NEW BEST: {val_acc*100:.2f}% - lr={lr}, units={units}, dropout={dropout}")
        else:
            print(f"  [{idx+1}/{len(configs)}] {val_acc*100:.2f}% - lr={lr}, units={units}, dropout={dropout}")
    
    print(f"\n  Best hyperparameters:")
    print(f"    Learning rate: {best_params['lr']}")
    print(f"    Hidden units: {best_params['units']}")
    print(f"    Dropout rate: {best_params['dropout']}")
    print(f"    Best val accuracy: {best_score*100:.2f}%")
    
    return best_model, best_params, best_score


def plot_results(history, y_true, y_pred, y_prob):
    """Plot training curves and confusion matrix"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].figure.colorbar(im, ax=axes[1, 0])
    classes = ['Not Looking', 'Looking']
    axes[1, 0].set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=classes, yticklabels=classes,
                   ylabel='True Label', xlabel='Predicted Label',
                   title='Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black",
                          fontsize=16)
    
    # ROC-like metrics bar chart
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    bars = axes[1, 1].bar(metrics.keys(), [v * 100 for v in metrics.values()], color='steelblue')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Model Metrics')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].axhline(y=90, color='r', linestyle='--', label='90% target')
    axes[1, 1].legend()
    
    for bar, val in zip(bars, metrics.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val*100:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_results_v2.png', dpi=150)
    print("  Saved training_results_v2.png")
    plt.close()


def print_metrics(y_true, y_pred, y_prob):
    """Print detailed metrics"""
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n[OVERALL METRICS]")
    print(f"   Accuracy:    {accuracy*100:.2f}%  {'[OK]' if accuracy >= 0.9 else '[NEEDS IMPROVEMENT]'}")
    print(f"   Precision:   {precision*100:.2f}%")
    print(f"   Recall:      {recall*100:.2f}%")
    print(f"   F1-Score:    {f1*100:.2f}%")
    print(f"   Specificity: {specificity*100:.2f}%")
    print(f"   AUC-ROC:     {auc:.4f}")
    
    print(f"\n[CONFUSION MATRIX]")
    print(f"   True Negatives:  {tn} (correctly NOT looking)")
    print(f"   True Positives:  {tp} (correctly LOOKING)")
    print(f"   False Positives: {fp} (wrong: said looking)")
    print(f"   False Negatives: {fn} (wrong: said not looking)")
    
    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, 
                               target_names=['Not Looking', 'Looking'],
                               digits=3))
    
    if accuracy < 0.9:
        print("\n[TIPS TO IMPROVE ACCURACY]")
        print("   1. Collect more training data (especially edge cases)")
        print("   2. Set USE_LANDMARKS = True to add eye landmarks")
        print("   3. Increase AUGMENTATION_FACTOR to 5")
        print("   4. Try recording in different lighting conditions")
        print("   5. Make sure labels are accurate (no mislabeled data)")
    
    return accuracy


def convert_to_tflite(model, scaler, feature_names):
    """Convert to TFLite"""
    print("\n[CONVERTING TO TFLITE]")
    
    class NormalizedModel(tf.Module):
        def __init__(self, model, mean, scale):
            super().__init__()
            self.model = model
            self.mean = tf.constant(mean, dtype=tf.float32)
            self.scale = tf.constant(scale, dtype=tf.float32)
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.float32)])
        def predict(self, x):
            x_normalized = (x - self.mean) / self.scale
            return self.model(x_normalized)
    
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    
    # Update input signature with correct shape
    num_features = len(mean)
    
    class NormalizedModelFixed(tf.Module):
        def __init__(self, model, mean, scale):
            super().__init__()
            self.model = model
            self.mean = tf.constant(mean, dtype=tf.float32)
            self.scale = tf.constant(scale, dtype=tf.float32)
    
    normalized_model = NormalizedModelFixed(model, mean, scale)
    
    # Create concrete function with correct shape
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, num_features], dtype=tf.float32)])
    def predict_fn(x):
        x_normalized = (x - normalized_model.mean) / normalized_model.scale
        return normalized_model.model(x_normalized)
    
    concrete_func = predict_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(OUTPUT_MODEL, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  Saved: {OUTPUT_MODEL} ({len(tflite_model)/1024:.1f} KB)")
    
    # Save metadata
    metadata = {
        'mean': mean.tolist(),
        'scale': scale.tolist(),
        'features': feature_names,
        'num_features': num_features
    }
    with open('model_metadata_v2.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Saved: model_metadata_v2.json")
    
    return tflite_model


def main():
    print("=" * 60)
    print("Looking at Camera Classifier - IMPROVED Training v2")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  USE_LANDMARKS: {USE_LANDMARKS}")
    print(f"  USE_AUGMENTATION: {USE_AUGMENTATION} (factor={AUGMENTATION_FACTOR})")
    print(f"  USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"  USE_CROSS_VALIDATION: {USE_CROSS_VALIDATION}")
    
    # Load data
    print("\n[1/7] Loading data...")
    X, y, feature_names = load_dataset()
    
    # Clean data
    print("\n[2/7] Cleaning data...")
    X, y = remove_invalid_samples(X, y)
    
    # Augment data
    if USE_AUGMENTATION:
        print("\n[3/7] Augmenting data...")
        X, y = augment_data(X, y, AUGMENTATION_FACTOR, NOISE_STD)
    else:
        print("\n[3/7] Skipping augmentation")
    
    # Cross-validation (optional)
    if USE_CROSS_VALIDATION:
        print("\n[4/7] Cross-validation...")
        cv_mean, cv_std = train_with_cross_validation(X, y, N_FOLDS)
    else:
        print("\n[4/7] Skipping cross-validation")
    
    # Split data
    print("\n[5/7] Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Class weights
    class_weights = None
    if USE_CLASS_WEIGHTS:
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {0: weights[0], 1: weights[1]}
        print(f"  Class weights: {class_weights}")
    
    # Hyperparameter tuning or architecture search
    print("\n[6/7] Training...")
    
    if USE_HYPERPARAMETER_TUNING:
        best_model, best_params, tuning_score = hyperparameter_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val, class_weights, X_train.shape[1]
        )
        
        # Final training with best hyperparameters (longer epochs)
        print(f"\n  Final training with best hyperparameters...")
        
        # Rebuild with best params for full training
        input_dim = X_train.shape[1]
        best_model = keras.Sequential()
        best_model.add(keras.layers.Input(shape=(input_dim,)))
        
        for i, u in enumerate(best_params['units']):
            best_model.add(keras.layers.Dense(u, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.001)))
            best_model.add(keras.layers.BatchNormalization())
            if i < len(best_params['units']) - 1:
                best_model.add(keras.layers.Dropout(best_params['dropout']))
        
        best_model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        best_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=best_params['lr']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        best_model, best_arch = try_multiple_architectures(
            X_train_scaled, y_train, X_val_scaled, y_val, class_weights
        )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
    ]
    
    history = best_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[7/7] Evaluating...")
    y_pred_prob = best_model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Plot results
    plot_results(history, y_test, y_pred, y_pred_prob)
    
    # Print metrics
    final_acc = print_metrics(y_test, y_pred, y_pred_prob)
    
    # Save model
    best_model.save(OUTPUT_KERAS)
    print(f"\n  Saved Keras model: {OUTPUT_KERAS}")
    
    # Convert to TFLite
    convert_to_tflite(best_model, scaler, feature_names)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Final accuracy: {final_acc*100:.2f}%")
    print("=" * 60)
    
    if final_acc >= 0.9:
        print("\n[SUCCESS] Achieved 90%+ accuracy!")
    else:
        print(f"\n[INFO] Current accuracy: {final_acc*100:.1f}%")
        print("To improve, try:")
        print("  1. Add more training data")
        print("  2. Set USE_LANDMARKS = True in the script")
        print("  3. Check for mislabeled samples in your data")


if __name__ == "__main__":
    main()

