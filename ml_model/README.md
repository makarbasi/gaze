# Looking at Camera Classifier - ML Model Training

This folder contains everything needed to train a TFLite model that classifies whether a person is looking at the camera.

## Models in this folder

This repo contains **two** generations of the "looking at camera" classifier:

- **v1 (7 features)**: `train_model.py` → `looking_classifier.tflite` (+ `scaler_params.json`)
- **v2 (41 features)**: `train_model_v2.py` → `looking_classifier_v2.tflite` (+ `model_metadata_v2.json`)

The Android app supports **v2** and will fall back to v1 if needed.

## Features Used (v1 = 7 total)

| Feature | Description |
|---------|-------------|
| `gaze_pitch` | Vertical gaze angle (radians) |
| `gaze_yaw` | Horizontal gaze angle (radians) |
| `head_pitch` | Head tilt up/down (radians) |
| `head_yaw` | Head turn left/right (radians) |
| `head_roll` | Head tilt sideways (radians) |
| `relative_pitch` | gaze_pitch - head_pitch |
| `relative_yaw` | gaze_yaw - head_yaw |

## Quick Start

### 1. Install Dependencies

```bash
cd ml_model
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your Excel or CSV files in the `data/` folder:
- `data/looking.xlsx` (or `.csv`) - samples where person IS looking at camera
- `data/notlooking.xlsx` (or `.csv`) - samples where person is NOT looking

Your files must have these columns:
```
gaze_pitch, gaze_yaw, head_pitch, head_yaw, head_roll, relative_pitch, relative_yaw
```

### 3. Train the Model

```bash
python train_model.py
```

### 4. Output Files (v1)

After training, you'll get:
- `looking_classifier.tflite` - Use this in your Android app
- `looking_classifier.h5` - Keras model (for further training)
- `scaler_params.json` - Normalization parameters
- `training_history.png` - Training curves plot

## Features Used (v2 = 41 total)

v2 uses:

- **7 core features** (same as v1): `gaze_pitch`, `gaze_yaw`, `head_pitch`, `head_yaw`, `head_roll`, `relative_pitch`, `relative_yaw`
- **34 landmark features**: selected eye/nose landmarks as `(x,y)` pairs (see `model_metadata_v2.json` for the exact list + order)

### Output Files (v2)

After training v2, you'll get:

- `looking_classifier_v2.tflite`
- `looking_classifier_v2.h5`
- `model_metadata_v2.json` (includes `features` order + mean/scale used when exporting)

## Data Format

Your Excel/CSV files should look like this:

| gaze_pitch | gaze_yaw | head_pitch | head_yaw | head_roll | relative_pitch | relative_yaw |
|------------|----------|------------|----------|-----------|----------------|--------------|
| 0.05 | -0.02 | 0.1 | 0.05 | 0.01 | -0.05 | -0.07 |
| 0.08 | 0.01 | 0.12 | 0.03 | -0.02 | -0.04 | -0.02 |
| ... | ... | ... | ... | ... | ... | ... |

**Note:** Values are in radians. The columns from your app recording should work directly!

## Model Architecture

```
Input (7 features)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(16) + ReLU
    ↓
Dense(1) + Sigmoid → Output (0-1 probability)
```

## Tips for Better Results

1. **Balanced data**: Try to have roughly equal samples of looking/not-looking
2. **Diverse conditions**: Record in different lighting, angles, distances
3. **More data is better**: Aim for 500+ samples per class
4. **Clean data**: Remove samples where face detection failed

## Using the Model in Android

### v2 (recommended)

Copy:

- `looking_classifier_v2.tflite` → `app/src/main/assets/looking_classifier_v2.tflite`
- `model_metadata_v2.json` → `app/src/main/assets/model_metadata_v2.json`

Android will read `model_metadata_v2.json["features"]` to build the input vector in the correct order.

### v1 (legacy)

Copy `looking_classifier.tflite` to:
```
app/src/main/assets/looking_classifier.tflite
```

Then load and run inference:
```java
// Load model
Interpreter tflite = new Interpreter(loadModelFile("looking_classifier.tflite"));

// Prepare input (7 features)
float[][] input = new float[1][7];
input[0][0] = gaze_pitch;
input[0][1] = gaze_yaw;
input[0][2] = head_pitch;
input[0][3] = head_yaw;
input[0][4] = head_roll;
input[0][5] = relative_pitch;
input[0][6] = relative_yaw;

// Run inference
float[][] output = new float[1][1];
tflite.run(input, output);

// Result: output[0][0] > 0.5 means "looking at camera"
boolean isLooking = output[0][0] > 0.5f;
```

