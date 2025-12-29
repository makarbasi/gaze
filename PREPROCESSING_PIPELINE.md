# Gaze App – Preprocessing & Model Pipeline (Code-Backed)

This doc explains **where and how input preprocessing happens** in the Android app, and shows the **actual code** used at each stage so it can be reviewed/audited.

## High-level flow (per camera frame)

1. **Camera frame (RGB)** → `rgbFrameBitmap`
2. **ROI crop + rotate** (from `/data/local/tmp/display_config.json`) → `croppedBitmap` (480×480)
3. **Face detection**
   - **TFLite**: `TFLiteFaceDetector` (internally resizes to 640×480 + converts to grayscale)
   - **Fallback**: QNN face detector (scales + packs RGB floats)
4. For each detected face:
   - **Landmark preprocess**: face crop → 112×112 float RGB
   - **Landmark postprocess**: landmarks mapped back to image coordinates
   - **Gaze preprocess**: head pose + normalized eye/face crops
   - **Gaze postprocess**: degrees → radians → camera-space pitch/yaw
5. **Looking classifier** (TFLite): consumes gaze + head pose + selected landmark coords → probability “looking at camera”

The main per-frame orchestration lives in:
- `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/ClassifierActivity.java`

## Runtime config that affects preprocessing

Most preprocessing parameters are loaded at runtime from:
- **`/data/local/tmp/display_config.json`** (parsed by `DisplayConfig`)

Key fields used by preprocessing:
- `crop_side` / `crop_offset_x` / `crop_offset_y` / `crop_scale`
- `img_orientation`
- `fisheye_enabled`, `fisheye_strength`, `fisheye_zoom`
- `face_detection_threshold`, `min_face_size`
- `debug_logs` (toggles spammy per-frame logs)

Implementation:
- `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/DisplayConfig.java`

## 1) ROI crop + rotate (frame → 480×480 crop)

Implemented in `ClassifierActivity.processImage()` using `DisplayConfig` + `ImageUtils.getTransformationMatrix(...)`.

```java
// ClassifierActivity.java (inside processImage)
rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

DisplayConfig cropConfig = DisplayConfig.getInstance();
cropConfig.checkAndReload();

RegionResult region = extractRegionWithInfo(
    rgbFrameBitmap, cropConfig.cropOffsetX, cropConfig.cropOffsetY, cropConfig.cropScale);
Bitmap regionBitmap = region.bitmap;

final Canvas canvas = new Canvas(croppedBitmap);
Matrix regionToCropTransform = ImageUtils.getTransformationMatrix(
    regionBitmap.getWidth(),
    regionBitmap.getHeight(),
    DemoConfig.crop_W,
    DemoConfig.crop_H,
    cropConfig.imgOrientation,
    MAINTAIN_ASPECT);
canvas.drawBitmap(regionBitmap, regionToCropTransform, null);

if (DemoConfig.USE_FRONT_CAM) {
  Mat mm = bitmap2mat(croppedBitmap);
  Core.flip(mm, mm, 0);
  Utils.matToBitmap(mm, croppedBitmap);
}
```

Notes:
- The crop result is **`croppedBitmap`** (usually 480×480).
- If front camera is enabled, it applies an OpenCV `flip(...)` on the crop.

## 2) Face detection preprocessing

### 2.1 TFLite face detector (resizes + grayscale)

The app passes the **480×480 crop** into the detector, but **the detector internally rescales to 640×480** (model input) and converts to grayscale bytes:

File: `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/TFLiteFaceDetector.java`

```java
public float[][] detectFaces(Bitmap bitmap) {
  if (!isInitialized || interpreter == null) {
    Log.e(TAG, "Face detector not initialized!");
    return new float[0][];
  }

  // Scale bitmap to 640x480
  Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, FACE_DET_WIDTH, FACE_DET_HEIGHT, true);

  // Store scale factors to map back to original coordinates
  float scaleX = (float) bitmap.getWidth() / FACE_DET_WIDTH;
  float scaleY = (float) bitmap.getHeight() / FACE_DET_HEIGHT;

  // Create input buffer - GRAYSCALE (1 channel), UINT8 quantized
  ByteBuffer inputBuffer = ByteBuffer.allocateDirect(FACE_DET_WIDTH * FACE_DET_HEIGHT);
  inputBuffer.order(ByteOrder.nativeOrder());

  int[] intValues = new int[FACE_DET_WIDTH * FACE_DET_HEIGHT];
  resizedBitmap.getPixels(intValues, 0, FACE_DET_WIDTH, 0, 0, FACE_DET_WIDTH, FACE_DET_HEIGHT);

  // Convert RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
  for (int pixelValue : intValues) {
    int r = (pixelValue >> 16) & 0xFF;
    int g = (pixelValue >> 8) & 0xFF;
    int b = pixelValue & 0xFF;
    byte gray = (byte) (0.299f * r + 0.587f * g + 0.114f * b);
    inputBuffer.put(gray);
  }
  inputBuffer.rewind();

  // Run inference (multiple outputs)
  interpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer},
      new java.util.HashMap<Integer, Object>() {{
        put(0, outputArray0);
        put(1, outputArray1);
        put(2, outputArray2);
      }});

  // Parse + map coordinates back using scaleX/scaleY
  List<float[]> faceBoxes = parseFaceDetectionOutputs(heatmapOutput, bboxOutput,
      FACE_DET_WIDTH, FACE_DET_HEIGHT, scaleX, scaleY);

  return faceBoxes.toArray(new float[0][]);
}
```

### 2.2 QNN face detection fallback (scale + pack RGB floats)

If TFLite face detection is disabled/unavailable, the app uses:
- `DetectionUtils.scale(...)` (bitmap scaling)
- `DetectionUtils.preprocessing(...)` (packs RGB to a float array)

File: `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/DetectionUtils.java`

```java
public static void preprocessing(Bitmap bitmap, float[] input) {
  int width = bitmap.getWidth();
  int height = bitmap.getHeight();

  int[] pixels = new int[width * height];
  bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
  for (int i = 0; i < pixels.length; i++) {
    int pixel = pixels[i];
    float R = Color.red(pixel);
    float G = Color.green(pixel);
    float B = Color.blue(pixel);
    input[i * 3] = R;
    input[i * 3 + 1] = G;
    input[i * 3 + 2] = B;
  }
}
```

## 3) Landmark preprocessing (face crop → 112×112 float RGB)

Landmark crop/resize happens in `ProcessFactory.landmark_preprocess(...)`.

File: `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/ProcessFactory.java`
(`ProcessFactory` is in package `com.example.gazedemo1` and imported statically by the activities.)

```java
public static LandmarkPreprocessResult landmark_preprocess(Mat img, float[] face) {
  int width = img.width();
  int height = img.height();
  int x1 = (int) face[0];
  int y1 = (int) face[1];
  int x2 = (int) face[2];
  int y2 = (int) face[3];
  int w = x2 - x1 + 1;
  int h = y2 - y1 + 1;
  int cx = x1 + w / 2;
  int cy = y1 + h / 2;

  int size = (int) (Math.max(w, h) * 1.15);
  x1 = cx - size / 2;
  x2 = x1 + size;
  y1 = cy - size / 2;
  y2 = y1 + size;

  x1 = Math.max(0, x1);
  y1 = Math.max(0, y1);
  x2 = Math.min(width, x2);
  y2 = Math.min(height, y2);

  int edx1 = Math.max(0, -x1);
  int edy1 = Math.max(0, -y1);
  int edx2 = Math.max(0, x2 - width);
  int edy2 = Math.max(0, y2 - height);

  Mat cropped = crop(img, x1, y1, x2, y2);
  if (edx1 > 0 || edy1 > 0 || edx2 > 0 || edy2 > 0) {
    Scalar border_value = new Scalar(0);
    Core.copyMakeBorder(cropped, cropped, edy1, edy2, edx1, edx2, Core.BORDER_CONSTANT, border_value);
  }
  Imgproc.resize(cropped, cropped, new Size(112, 112));

  LandmarkPreprocessResult result = new LandmarkPreprocessResult();
  result.input = mat2array(cropped); // RGB floats in [0..1]
  result.edx1 = edx1;
  result.edy1 = edy1;
  result.x1 = x1;
  result.y1 = y1;
  result.size = size;
  return result;
}
```

And the landmark output is mapped back into image coordinates:

```java
public static float[] landmark_postprocess(LandmarkPreprocessResult result, float[] output) {
  for (int i = 0; i < output.length; i++) {
    output[i] *= (float) result.size;
    if (i % 2 == 0) {
      output[i] = output[i] - result.edx1 + result.x1;
    } else {
      output[i] = output[i] - result.edy1 + result.y1;
    }
  }
  return output;
}
```

## 4) Gaze preprocessing (head pose + normalized crops)

### 4.1 Head pose estimate (solvePnP)

File: `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/GazeEstimationUtils.java`

```java
public static Mat get_camera_matrix(float cam_w, float cam_h) {
  float c_x = cam_w / 2;
  float c_y = cam_h / 2;
  float f_x = c_x / (float) Math.tan(60.0 / 2.0 * Math.PI / 180.0);
  float f_y = f_x;
  float[] matrix = {f_x, 0.0f, c_x, 0.0f, f_y, c_y, 0.0f, 0.0f, 1.0f};
  MatOfFloat mat = new MatOfFloat();
  mat.fromArray(matrix);
  return mat.reshape(1, new int[]{3, 3});
}

public static void estimateHeadPose(float[] landmarks, Mat rvec, Mat tvec, Mat camera_matrix) {
  Point[] imagePoints = new MatOfPoint2f(extract_critical_landmarks(landmarks)).toArray();
  MatOfDouble distCoeffs = new MatOfDouble();
  distCoeffs.fromArray(0, 0, 0, 0, 0);
  Calib3d.solvePnP(face_model, new MatOfPoint2f(imagePoints), camera_matrix, distCoeffs, rvec, tvec);
}
```

### 4.2 Normalize data for gaze model (eye crops + face crop)

File: `GazeEstimationUtils.normalizeDataForInference(...)` (builds normalized crops).
This returns:
- left eye crop (`60×60`)
- right eye crop (`60×60`)
- face crop (`120×120`)
- plus a rotation matrix `R` used in postprocessing

`ProcessFactory.gaze_preprocess(...)` calls into it:

```java
public static GazePreprocessResult gaze_preprocess(Mat img, float[] landmark) {
  Mat rvec = new Mat();
  Mat tvec = new Mat();
  Mat camera_matrix = GazeEstimationUtils.get_camera_matrix(DemoConfig.crop_W, DemoConfig.crop_H);
  GazeEstimationUtils.estimateHeadPose(landmark, rvec, tvec, camera_matrix);

  List data = GazeEstimationUtils.normalizeDataForInference(img, rvec, tvec, camera_matrix);
  float[] leye_image = mat2array((Mat) data.get(0));
  float[] reye_image = mat2array((Mat) data.get(1));
  float[] face_image = mat2array((Mat) data.get(2));
  Mat R = (Mat) data.get(3);

  GazePreprocessResult result = new GazePreprocessResult();
  result.face = face_image;
  result.leye = leye_image;
  result.reye = reye_image;
  result.R = R;
  result.tvec = tvec;
  result.rvec = rvec;
  result.camera_matrix = camera_matrix;
  return result;
}
```

### 4.3 Gaze postprocess (degrees → radians → camera space)

File: `ProcessFactory.gaze_postprocess(...)`

```java
public static float[] gaze_postprocess(float[] pred_pitchyaw_aligned, Mat R) {
  pred_pitchyaw_aligned = deg2rad(pred_pitchyaw_aligned);
  Mat pred_vec_aligned = GazeEstimationUtils.euler_to_vec(pred_pitchyaw_aligned[0], pred_pitchyaw_aligned[1]);
  Mat pred_vec_cam = R.inv().matMul(pred_vec_aligned);
  Core.divide(pred_vec_cam, new Scalar(Core.norm(pred_vec_cam)), pred_vec_cam);
  return GazeEstimationUtils.vec_to_euler(pred_vec_cam);
}
```

## 5) Looking classifier (TFLite) – feature assembly

The looking classifier builds its input vector using the **metadata feature list** (from `model_metadata_v2.json`).
This guarantees Android feature order matches training.

File: `app/src/main/java/org/tensorflow/lite/examples/gaze_estimation/LookingClassifier.java`

```java
// Core features derived from gaze + head pose
float gaze_pitch = gazePitchYaw[0];
float gaze_yaw = gazePitchYaw[1];
float[] headPose = extractHeadPose(rvec);
float head_pitch = headPose[0];
float head_yaw = headPose[1];
float head_roll = headPose[2];
float relative_pitch = gaze_pitch - head_pitch;
float relative_yaw = gaze_yaw - head_yaw;

// Feature vector in metadata order (RAW values; model contains normalization)
for (int i = 0; i < numFeatures; i++) {
  FeatureSpec spec = featureSpecs[i];
  float v;
  switch (spec.kind) {
    case GAZE_PITCH: v = gaze_pitch; break;
    case GAZE_YAW: v = gaze_yaw; break;
    case HEAD_PITCH: v = head_pitch; break;
    case HEAD_YAW: v = head_yaw; break;
    case HEAD_ROLL: v = head_roll; break;
    case REL_PITCH: v = relative_pitch; break;
    case REL_YAW: v = relative_yaw; break;
    case LANDMARK_X: v = getLandmarkCoord(landmarks, spec.landmarkIndex, true); break;
    case LANDMARK_Y: v = getLandmarkCoord(landmarks, spec.landmarkIndex, false); break;
    default: v = 0f;
  }
  inputArray[0][i] = v;
}

interpreter.run(inputArray, outputArray);
lastProbability = outputArray[0][0];
```

## Logging control (to keep Logcat clean)

Set in `/data/local/tmp/display_config.json`:
- `"debug_logs": false` → quiet
- `"debug_logs": true` → enables debug prints, including gated native QNN `LOGD(...)`

The app gates common per-frame spam (face candidates, smooth matching, native QNN inference logs) behind this flag.




