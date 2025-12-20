/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.gaze_estimation;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.widget.SwitchCompat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Vector;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.gaze_estimation.env.BorderedText;
import org.tensorflow.lite.examples.gaze_estimation.env.ImageUtils;
import org.tensorflow.lite.examples.gaze_estimation.env.Logger;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier.Device;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier.Model;

import com.example.gazedemo1.ProcessFactory;
// PlatformValidator removed - causes native crashes on modern Android due to OpenCL restrictions
// import com.qualcomm.qti.platformvalidator.PlatformValidator;
// import com.qualcomm.qti.platformvalidator.PlatformValidatorUtil;

// QNN Model Runner for DLC files (replaces SNPE)
import org.tensorflow.lite.examples.gaze_estimation.QNNModelRunner;

import static com.example.gazedemo1.ProcessFactory.bitmap2mat;
import static com.example.gazedemo1.ProcessFactory.gaze_postprocess;
import static com.example.gazedemo1.ProcessFactory.gaze_preprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_postprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_preprocess;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.preprocessing;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.scale;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.transpose;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.postprocessing;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawbox;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawgaze;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawheadpose;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawlandmark;

// TFLite Face Detector from HotGaze (more accurate model)
import org.tensorflow.lite.examples.gaze_estimation.TFLiteFaceDetector;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(DemoConfig.Preview_H, DemoConfig.Preview_W);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private BorderedText borderedText;
  // QNN Model Runners (replaces SNPE NeuralNetwork)
  private QNNModelRunner face_detection_qnn = null;
  private QNNModelRunner landmark_detection_qnn = null;
  private QNNModelRunner gaze_estimation_qnn = null;
  
  private float[] face_detection_input = new float[DemoConfig.getFaceDetectionInputsize()];
  private float[] NNoutput = null;
  Bitmap detection = null;
  Bitmap faceCropBitmap = null;  // High-res face crop for display
  SmootherList smoother_list = null;
  
  // TFLite Face Detector (from HotGaze - more accurate)
  private TFLiteFaceDetector tfliteFaceDetector = null;
  
  // Looking Classifier - determines if user is looking at camera
  private LookingClassifier lookingClassifier = null;
  private volatile boolean isLookingAtCamera = false;
  private volatile float lookingProbability = 0f;

  // Latest head pose (pitch/yaw/roll) for UI (radians). Updated from gaze_preprocess_result.rvec.
  private volatile float latestHeadPitch = 0f;
  private volatile float latestHeadYaw = 0f;
  private volatile float latestHeadRoll = 0f;
  private volatile boolean hasHeadPose = false;
  
  // Processed FPS (how many frames actually run through processImage() per second).
  private long fpsWindowStartMs = 0L;
  private int fpsWindowFrames = 0;
  private volatile float processedFps = 0f;
  
  // Minimal overlay mode (UI)
  private volatile boolean minimalModeEnabled = false;
  private SwitchCompat minimalModeSwitch;
  
  // Fisheye toggle (UI). When OFF, fisheye correction is never executed.
  private volatile boolean fisheyeUiEnabled = true;
  private SwitchCompat fisheyeSwitch;
  private View cameraContainer;
  private View regionPreviewContainer;
  private ImageView faceImageView;
  private View regionLabel;
  private View faceLabel;
  private TextView headPosePitchText;
  private TextView headPoseYawText;
  private TextView headPoseRollText;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    
    // Load gaze configuration from /data/local/tmp/gaze_config.txt
    gazeConfig = GazeConfig.getInstance();
    Log.i("GazeConfig", "Loaded config: threshold_base=" + Math.toDegrees(gazeConfig.thresholdBase) + 
        "°, threshold_max=" + Math.toDegrees(gazeConfig.thresholdMax) + "°");
    
    // Initialize gaze calibration UI elements
    gazeOverlay = findViewById(R.id.gaze_overlay);
    gazeStatusText = findViewById(R.id.gaze_status_text);
    calibrateButton = findViewById(R.id.calibrate_button);
    lookingProbText = findViewById(R.id.looking_prob_text);
    headPosePitchText = findViewById(R.id.head_pose_pitch);
    headPoseYawText = findViewById(R.id.head_pose_yaw);
    headPoseRollText = findViewById(R.id.head_pose_roll);
    
    // Minimal overlay mode UI elements
    minimalModeSwitch = findViewById(R.id.switch_minimal_mode);
    fisheyeSwitch = findViewById(R.id.switch_fisheye);
    cameraContainer = findViewById(R.id.container); // Camera preview host (TextureView fragment)
    regionPreviewContainer = findViewById(R.id.region_preview_container);
    faceImageView = findViewById(R.id.faceImageView);
    regionLabel = findViewById(R.id.region_label);
    faceLabel = findViewById(R.id.face_label);
    
    if (minimalModeSwitch != null) {
      minimalModeSwitch.setChecked(minimalModeEnabled);
      minimalModeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
        minimalModeEnabled = isChecked;
        applyMinimalModeUi(isChecked);
      });
    }
    if (fisheyeSwitch != null) {
      fisheyeSwitch.setChecked(fisheyeUiEnabled);
      fisheyeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
        fisheyeUiEnabled = isChecked;
      });
    }
    applyMinimalModeUi(minimalModeEnabled);
    
    // Apply saved calibration if auto_calibrate is enabled
    if (gazeConfig.autoCalibrate) {
      referencePitch = gazeConfig.referencePitch;
      referenceYaw = gazeConfig.referenceYaw;
      isCalibrated = true;
      smoothedPitch = referencePitch;
      smoothedYaw = referenceYaw;
      lookingAtCameraCount = gazeConfig.smoothingFrames;
      isLookingAtCamera = true;
      
      // Show overlay since we're auto-calibrated
      if (gazeOverlay != null) {
        gazeOverlay.setVisibility(View.VISIBLE);
      }
      if (gazeStatusText != null) {
        gazeStatusText.setVisibility(View.VISIBLE);
      }
      
      Log.i("GazeConfig", "Auto-calibrated from config: pitch=" + Math.toDegrees(referencePitch) + 
          "°, yaw=" + Math.toDegrees(referenceYaw) + "°");
      Toast.makeText(this, "Auto-calibrated from saved config", Toast.LENGTH_SHORT).show();
    }
    
    if (calibrateButton != null) {
      calibrateButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          // Check if face is detected using TFLite face detection output
          // This is the ONLY check - we don't care if user is "looking at camera" or not
          Log.i("GazeCalib", "Calibrate clicked. hasFace=" + hasFaceDetected + 
              ", hasGaze=" + hasGazeData + 
              ", pitch=" + Math.toDegrees(latestGazePitch) + 
              ", yaw=" + Math.toDegrees(latestGazeYaw));
          
          if (hasFaceDetected) {
            // Face is detected - save current gaze values as reference
            // (regardless of whether user is "looking at camera" or not)
            referencePitch = latestGazePitch;
            referenceYaw = latestGazeYaw;
            isCalibrated = true;
            
            // Reset smoothing values
            smoothedPitch = latestGazePitch;
            smoothedYaw = latestGazeYaw;
            lookingAtCameraCount = gazeConfig.smoothingFrames;  // Start as "looking"
            notLookingAtCameraCount = 0;
            isLookingAtCamera = true;
            
            // Save to config file for persistence
            gazeConfig.updateReference(referencePitch, referenceYaw);
            
            Toast.makeText(ClassifierActivity.this, 
                String.format("✓ Calibrated! Pitch=%.1f°, Yaw=%.1f° (saved)", 
                    Math.toDegrees(referencePitch), Math.toDegrees(referenceYaw)),
                Toast.LENGTH_SHORT).show();
            
            // Show overlay
            if (gazeOverlay != null) {
              gazeOverlay.setVisibility(View.VISIBLE);
            }
            if (gazeStatusText != null) {
              gazeStatusText.setVisibility(View.VISIBLE);
            }
            
            Log.i("GazeCalib", "✓ Calibrated at Pitch=" + Math.toDegrees(referencePitch) + 
                "°, Yaw=" + Math.toDegrees(referenceYaw) + "°");
          } else {
            // No face detected by TFLite face detection
            Toast.makeText(ClassifierActivity.this, 
                "No face detected. Look at the camera and try again.", 
                Toast.LENGTH_SHORT).show();
            Log.w("GazeCalib", "No face detected by TFLite. hasFace=" + hasFaceDetected);
          }
        }
      });
    }
    
    // Initialize recording UI elements
    recordButton = findViewById(R.id.record_button);
    isLookingCheckbox = findViewById(R.id.is_looking_checkbox);
    
    if (recordButton != null) {
      recordButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          if (!isRecording) {
            // Start recording
            startRecording();
          } else {
            // Stop recording
            stopRecording();
          }
        }
      });
    }
  }
  
  /**
   * Minimal overlay mode:
   * - no camera preview
   * - no region/face preview bitmaps
   * - overlay stays visible and turns green/red
   *
   * Important: keep the TextureView surface alive (INVISIBLE not GONE) for Camera2.
   */
  private void applyMinimalModeUi(boolean enabled) {
    if (cameraContainer != null) {
      cameraContainer.setVisibility(enabled ? View.INVISIBLE : View.VISIBLE);
    }
    if (regionPreviewContainer != null) {
      regionPreviewContainer.setVisibility(enabled ? View.GONE : View.VISIBLE);
    }
    if (faceImageView != null) {
      faceImageView.setVisibility(enabled ? View.GONE : View.VISIBLE);
    }
    if (regionLabel != null) {
      regionLabel.setVisibility(enabled ? View.GONE : View.VISIBLE);
    }
    if (faceLabel != null) {
      faceLabel.setVisibility(enabled ? View.GONE : View.VISIBLE);
    }
    // Ensure overlay UI is visible when minimal mode is on
    if (enabled) {
      if (gazeOverlay != null) gazeOverlay.setVisibility(View.VISIBLE);
      if (gazeStatusText != null) gazeStatusText.setVisibility(View.VISIBLE);
    }
  }
  
  /**
   * Start recording landmarks and gaze data to a file
   * Files are saved to app's external files directory:
   *   /storage/emulated/0/Android/data/org.tensorflow.lite.examples.gaze_estimation/files/
   * 
   * To pull files via ADB:
   *   adb pull /storage/emulated/0/Android/data/org.tensorflow.lite.examples.gaze_estimation/files/
   */
  private void startRecording() {
    try {
      // Create filename with timestamp
      SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
      String timestamp = sdf.format(new Date());
      boolean isLooking = isLookingCheckbox != null && isLookingCheckbox.isChecked();
      String suffix = isLooking ? "_looking" : "_notlooking";
      String filename = timestamp + suffix + ".csv";
      
      // Use app's external files directory (accessible via ADB without root)
      File dir = getExternalFilesDir(null);
      if (dir == null) {
        throw new IOException("External files directory not available");
      }
      File file = new File(dir, filename);
      currentRecordingFile = file.getAbsolutePath();
      
      // Create file and writer
      recordingWriter = new BufferedWriter(new FileWriter(file));
      
      // Write CSV header
      StringBuilder header = new StringBuilder();
      header.append("timestamp,isLooking");
      // Landmark columns (98 landmarks * 2 = 196 values)
      for (int i = 0; i < 98; i++) {
        header.append(",lm").append(i).append("_x");
        header.append(",lm").append(i).append("_y");
      }
      // Gaze and head pose columns (7 features for ML model)
      header.append(",gaze_pitch,gaze_yaw");
      header.append(",head_pitch,head_yaw,head_roll");
      header.append(",relative_pitch,relative_yaw");
      header.append("\n");
      recordingWriter.write(header.toString());
      
      isRecording = true;
      recordButton.setText("⏹ Stop");
      recordButton.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xDDFF0000));
      
      Toast.makeText(this, "Recording started: " + currentRecordingFile, Toast.LENGTH_SHORT).show();
      Log.i("Recording", "Started recording to: " + currentRecordingFile);
      
    } catch (IOException e) {
      Log.e("Recording", "Failed to start recording: " + e.getMessage());
      Toast.makeText(this, "Failed to start recording: " + e.getMessage(), Toast.LENGTH_SHORT).show();
    }
  }
  
  /**
   * Stop recording and close the file
   */
  private void stopRecording() {
    if (recordingWriter != null) {
      try {
        recordingWriter.close();
        recordingWriter = null;
        
        Toast.makeText(this, "Recording saved: " + currentRecordingFile, Toast.LENGTH_SHORT).show();
        Log.i("Recording", "Stopped recording, saved to: " + currentRecordingFile);
        
      } catch (IOException e) {
        Log.e("Recording", "Error closing recording file: " + e.getMessage());
      }
    }
    
    isRecording = false;
    recordButton.setText("⏺ Record");
    recordButton.setBackgroundTintList(android.content.res.ColorStateList.valueOf(0xDDFF5722));
    currentRecordingFile = null;
  }
  
  /**
   * Extract head pose Euler angles (pitch, yaw, roll) from rotation vector
   * @param rvec Rotation vector from head pose estimation
   * @return float[3] = {head_pitch, head_yaw, head_roll} in radians
   */
  private float[] extractHeadPose(Mat rvec) {
    if (rvec == null || rvec.empty()) {
      return new float[]{0, 0, 0};
    }
    
    try {
      // Convert rotation vector to rotation matrix
      Mat rotMat = new Mat();
      org.opencv.calib3d.Calib3d.Rodrigues(rvec, rotMat);
      
      // Extract Euler angles from rotation matrix
      // Using the convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
      double sy = Math.sqrt(rotMat.get(0, 0)[0] * rotMat.get(0, 0)[0] + rotMat.get(1, 0)[0] * rotMat.get(1, 0)[0]);
      boolean singular = sy < 1e-6;
      
      double pitch, yaw, roll;
      if (!singular) {
        roll = Math.atan2(rotMat.get(2, 1)[0], rotMat.get(2, 2)[0]);
        pitch = Math.atan2(-rotMat.get(2, 0)[0], sy);
        yaw = Math.atan2(rotMat.get(1, 0)[0], rotMat.get(0, 0)[0]);
      } else {
        roll = Math.atan2(-rotMat.get(1, 2)[0], rotMat.get(1, 1)[0]);
        pitch = Math.atan2(-rotMat.get(2, 0)[0], sy);
        yaw = 0;
      }
      
      rotMat.release();
      return new float[]{(float)pitch, (float)yaw, (float)roll};
    } catch (Exception e) {
      Log.e("Recording", "Error extracting head pose: " + e.getMessage());
      return new float[]{0, 0, 0};
    }
  }
  
  /**
   * Record a single frame with gaze and head pose data
   * Records: gaze_pitch, gaze_yaw, head_pitch, head_yaw, head_roll, relative_pitch, relative_yaw
   */
  private void recordFrame(float[] landmarks, float[] gaze, Mat rvec) {
    if (!isRecording || recordingWriter == null) {
      return;
    }
    
    try {
      StringBuilder line = new StringBuilder();
      
      // Timestamp
      line.append(System.currentTimeMillis());
      
      // isLooking status
      boolean isLooking = isLookingCheckbox != null && isLookingCheckbox.isChecked();
      line.append(",").append(isLooking ? "1" : "0");
      
      // Landmarks (196 values = 98 landmarks * 2)
      if (landmarks != null && landmarks.length >= 196) {
        for (int i = 0; i < 196; i++) {
          line.append(",").append(landmarks[i]);
        }
      } else {
        for (int i = 0; i < 196; i++) {
          line.append(",0");
        }
      }
      
      // Extract head pose from rvec
      float[] headPose = extractHeadPose(rvec);
      float head_pitch = headPose[0];
      float head_yaw = headPose[1];
      float head_roll = headPose[2];
      
      // Gaze (pitch, yaw)
      float gaze_pitch = (gaze != null && gaze.length >= 2) ? gaze[0] : 0;
      float gaze_yaw = (gaze != null && gaze.length >= 2) ? gaze[1] : 0;
      
      // Calculate relative gaze (gaze relative to head orientation)
      float relative_pitch = gaze_pitch - head_pitch;
      float relative_yaw = gaze_yaw - head_yaw;
      
      // Write all 7 features
      line.append(",").append(gaze_pitch);
      line.append(",").append(gaze_yaw);
      line.append(",").append(head_pitch);
      line.append(",").append(head_yaw);
      line.append(",").append(head_roll);
      line.append(",").append(relative_pitch);
      line.append(",").append(relative_yaw);
      
      line.append("\n");
      recordingWriter.write(line.toString());
      
    } catch (IOException e) {
      Log.e("Recording", "Error writing frame: " + e.getMessage());
    }
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  /**
   * Setup QNN network for DLC model
   * @param model_path Path to the DLC file
   * @param model_op_out Output layer name
   * @param preferredBackend Preferred backend (GPU, CPU - HTP is excluded)
   * @return QNNModelRunner instance or null on failure
   */
  private QNNModelRunner setupQNNNetwork(String model_path, String model_op_out, QNNModelRunner.Backend preferredBackend) {
    // Try backends in order of preference: GPU and CPU only (no HTP)
    QNNModelRunner.Backend[] backendsToTry = {
        preferredBackend,
        QNNModelRunner.Backend.GPU,
        QNNModelRunner.Backend.CPU
    };
    
    for (QNNModelRunner.Backend backend : backendsToTry) {
      try {
        Log.i("QNN", "Trying to load " + model_path + " with backend: " + backend);
        QNNModelRunner runner = new QNNModelRunner();
        if (runner.loadModel(model_path, model_op_out, backend)) {
          Log.i("QNN", "✓ Successfully loaded " + model_path + " with backend: " + runner.getBackendInfo());
          return runner;
        }
        runner.close();
      } catch (Exception e) {
        Log.w("QNN", "Failed to load " + model_path + " with " + backend + ": " + e.getMessage());
      }
    }
    
    Log.e("QNN", "Failed to load " + model_path + " with any backend");
    return null;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap =
        Bitmap.createBitmap(
                DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);

    // Use img_orientation from display config (can be modified at runtime via /data/local/tmp/display_config.json)
    DisplayConfig displayConfig = DisplayConfig.getInstance();
    LOGGER.i("Using img_orientation from config: %d", displayConfig.imgOrientation);
    
    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            DemoConfig.crop_W,
            DemoConfig.crop_H,
            displayConfig.imgOrientation,
            MAINTAIN_ASPECT);


    if(DemoConfig.USE_VERTICAL)
      detection = Bitmap.createBitmap(DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);
    else
      detection = Bitmap.createBitmap(DemoConfig.crop_H, DemoConfig.crop_W, Config.ARGB_8888);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    // Initialize TFLite Face Detector (from HotGaze - more accurate model)
    if (DemoConfig.USE_TFLITE_FACE_DETECTION) {
      try {
        tfliteFaceDetector = new TFLiteFaceDetector(this);
        tfliteFaceDetector.initialize();
        tfliteFaceDetector.setFaceDetectionThreshold(DemoConfig.tflite_face_detection_threshold);
        tfliteFaceDetector.setMinFaceSize(DemoConfig.tflite_min_face_size);
        tfliteFaceDetector.setNmsIouThreshold(DemoConfig.tflite_nms_iou_threshold);
        Log.i("TFLiteFace", "✓ TFLite Face Detector initialized: " + tfliteFaceDetector.getAcceleratorType());
      } catch (Exception e) {
        Log.e("TFLiteFace", "TFLite Face Detector initialization failed, falling back to SNPE", e);
        tfliteFaceDetector = null;
        Toast.makeText(this, "TFLite face detection failed, using SNPE fallback", Toast.LENGTH_LONG).show();
      }
    }
    
    // Initialize Looking Classifier (determines if user is looking at camera)
    try {
      lookingClassifier = new LookingClassifier();
      boolean ok = lookingClassifier.initialize(this);
      if (ok) {
        Log.i("LookingClassifier", "✓ Looking Classifier initialized successfully");
      } else {
        Log.e("LookingClassifier", "Looking Classifier initialization failed: " + lookingClassifier.getLastInitError());
        // Keep the instance for debugging (isInitialized() will remain false)
      }
    } catch (Exception e) {
      Log.e("LookingClassifier", "Looking Classifier error: " + e.getMessage(), e);
      // Keep null on hard crash here
      lookingClassifier = null;
    }

    // Create QNN networks for landmark and gaze estimation (and face detection fallback)
    // QNN replaces SNPE for running DLC files
    try {
      Log.i("QNN", "════════════════════════════════════════");
      Log.i("QNN", "Initializing QNN networks for DLC models...");
      Log.i("QNN", "════════════════════════════════════════");
      
      // Check if QNN is available
      if (!QNNModelRunner.isQnnAvailable()) {
        Log.e("QNN", "QNN is not available on this device. Please ensure QNN libraries are installed.");
        Toast.makeText(this, "QNN not available. Please install QNN SDK libraries.", Toast.LENGTH_LONG).show();
        return;
      }
      
      // Only initialize QNN face detection if TFLite is not being used or failed
      if (!DemoConfig.USE_TFLITE_FACE_DETECTION || tfliteFaceDetector == null) {
        // Use face_detection_model_out ("bbox_det") as the output tensor name
        // Use GPU backend (not HTP) for better compatibility
        face_detection_qnn = setupQNNNetwork(DemoConfig.face_detection_model_path, DemoConfig.face_detection_model_out, QNNModelRunner.Backend.GPU);
        if (face_detection_qnn != null) {
          Log.i("QNN", "✓ QNN Face Detection network loaded: " + face_detection_qnn.getBackendInfo());
        }
      } else {
        Log.i("QNN", "Using TFLite for face detection, skipping QNN face detection");
      }
      
      // Load landmark detection DLC with QNN
      // Use landmark_detection_model_out ("facial_landmark") as the output tensor name, NOT landmark_detection_model_op_out ("Gemm_137")
      // Use GPU backend (not HTP) for better compatibility
      landmark_detection_qnn = setupQNNNetwork(DemoConfig.landmark_detection_model_path, DemoConfig.landmark_detection_model_out, QNNModelRunner.Backend.GPU);
      if (landmark_detection_qnn != null) {
        Log.i("QNN", "✓ QNN Landmark Detection network loaded: " + landmark_detection_qnn.getBackendInfo());
      }
      
      // Load gaze estimation DLC with QNN
      // Use gaze_estimation_model_out ("gaze_pitchyaw") as the output tensor name, NOT gaze_estimation_model_op_out ("Gemm_602")
      // Use GPU backend (not HTP) for better compatibility
      gaze_estimation_qnn = setupQNNNetwork(DemoConfig.gaze_estimation_model_path, DemoConfig.gaze_estimation_model_out, QNNModelRunner.Backend.GPU);
      if (gaze_estimation_qnn != null) {
        Log.i("QNN", "✓ QNN Gaze Estimation network loaded: " + gaze_estimation_qnn.getBackendInfo());
      }

      if ((!DemoConfig.USE_TFLITE_FACE_DETECTION && face_detection_qnn == null) || 
          landmark_detection_qnn == null || gaze_estimation_qnn == null) {
        Log.e("QNN", "Failed to initialize one or more QNN networks");
        Toast.makeText(this, "QNN initialization failed. Some features may not work.", Toast.LENGTH_LONG).show();
      } else {
        Log.i("QNN", "════════════════════════════════════════");
        Log.i("QNN", "✓ All QNN networks initialized successfully");
        Log.i("QNN", "════════════════════════════════════════");
      }
    } catch (Exception e) {
      Log.e("QNN", "QNN initialization failed: " + e.getMessage(), e);
      Toast.makeText(this, "QNN unavailable: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }

    smoother_list = new SmootherList();
  }

  long inferencetime;
  long latency;
  int bitmapset = 0;
  int[] pixs = new int[DemoConfig.crop_H*DemoConfig.crop_W];
  int[] pixs_out = new int[DemoConfig.crop_H*DemoConfig.crop_W];
  
  // Store latest gaze pitch/yaw for display
  // These values PERSIST between frames (not reset each frame) so button can read them
  private volatile float latestGazePitch = 0f;
  private volatile float latestGazeYaw = 0f;
  private volatile boolean hasGazeData = false;
  private volatile boolean hasFaceDetected = false;  // Set when face detection finds a face
  
  // Gaze calibration with head movement compensation
  private boolean isCalibrated = false;
  private float referencePitch = 0f;
  private float referenceYaw = 0f;
  
  // Gaze config - loaded from /data/local/tmp/gaze_config.txt
  private GazeConfig gazeConfig;
  
  // Temporal smoothing for stable detection
  private int lookingAtCameraCount = 0;
  private int notLookingAtCameraCount = 0;
  // Note: isLookingAtCamera is defined above with lookingClassifier fields
  
  // Running average for gaze (reduces jitter)
  private float smoothedPitch = 0f;
  private float smoothedYaw = 0f;
  
  // UI elements for gaze tracking
  private View gazeOverlay;
  private TextView gazeStatusText;
  private TextView lookingProbText;
  private Button calibrateButton;
  
  // Recording UI and state
  private Button recordButton;
  private CheckBox isLookingCheckbox;
  private boolean isRecording = false;
  private BufferedWriter recordingWriter = null;
  private String currentRecordingFile = null;
  @Override

  protected Bitmap processImage() {
    inferencetime = 0;
    // NOTE: We do NOT reset hasFaceDetected and hasGazeData here anymore
    // They persist between frames so the calibrate button can read them reliably
    
    // Check if OpenCV is ready before using Mat operations
    if (!isOpenCVReady) {
      Log.e("OpenCV", "OpenCV not initialized, skipping frame processing");
      return rgbFrameBitmap;
    }
    
    // Update processed FPS stats (counts only processed frames, not camera frames).
    final long nowMs = SystemClock.uptimeMillis();
    if (fpsWindowStartMs == 0L) {
      fpsWindowStartMs = nowMs;
      fpsWindowFrames = 0;
      processedFps = 0f;
    }
    fpsWindowFrames++;
    final long elapsedMs = nowMs - fpsWindowStartMs;
    if (elapsedMs >= 1000L) {
      processedFps = (fpsWindowFrames * 1000f) / Math.max(1L, elapsedMs);
      fpsWindowStartMs = nowMs;
      fpsWindowFrames = 0;
    }
    
    final boolean minimalMode = minimalModeEnabled;
    
    // Check if required networks are initialized (using QNN now)
    boolean hasFaceDetection = (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) || 
                               (face_detection_qnn != null && face_detection_qnn.isReady());
    boolean hasLandmark = landmark_detection_qnn != null && landmark_detection_qnn.isReady();
    boolean hasGaze = gaze_estimation_qnn != null && gaze_estimation_qnn.isReady();
    
    if (!hasFaceDetection || !hasLandmark || !hasGaze) {
      Log.e("QNN", "Networks not initialized, skipping frame processing");
      return rgbFrameBitmap;
    }
    
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    // Get crop region settings from config
    DisplayConfig cropConfig = DisplayConfig.getInstance();
    cropConfig.checkAndReload();
    
    // Apply region-of-interest cropping for wide FOV cameras
    // crop_offset_x: 0.0=left, 0.25=left-quarter, 0.5=center, 0.75=right-quarter, 1.0=right
    // crop_scale: 0.5=use half width (2x zoom), 1.0=use full width
    RegionResult region = extractRegionWithInfo(
        rgbFrameBitmap, cropConfig.cropOffsetX, cropConfig.cropOffsetY, cropConfig.cropScale);
    Bitmap regionBitmap = region.bitmap;
    
    //transform and crop the frame using ImageUtils with proper rotation
    final Canvas canvas = new Canvas(croppedBitmap);
    
    // Create transformation matrix with rotation from config (img_orientation)
    Matrix regionToCropTransform = ImageUtils.getTransformationMatrix(
        regionBitmap.getWidth(),
        regionBitmap.getHeight(),
        DemoConfig.crop_W,
        DemoConfig.crop_H,
        cropConfig.imgOrientation,
        MAINTAIN_ASPECT);

    // Inverse transform for mapping crop-space landmarks back into full preview frame coordinates.
    Matrix cropToRegionTransform = new Matrix();
    regionToCropTransform.invert(cropToRegionTransform);
    
    canvas.drawBitmap(regionBitmap, regionToCropTransform, null);

    if (DemoConfig.USE_FRONT_CAM) {
      // flip the camera image
      Mat mm = bitmap2mat(croppedBitmap);
      Core.flip(mm, mm, 0);
      Utils.matToBitmap(mm, croppedBitmap);
    }

/*
    // DEBUG IMAGE START
    Mat mm = bitmap2mat(BitmapFactory.decodeResource(getResources(), R.drawable.sample10));
    Imgproc.resize(mm, mm, new org.opencv.core.Size(480, 480));
    croppedBitmap = Bitmap.createBitmap(DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);
    Utils.matToBitmap(mm, croppedBitmap);
    Log.d("BITMAP_SIZE", croppedBitmap.getWidth() + " " + croppedBitmap.getHeight());
    // DEBUG IMAGE END
*/

    //convert to floating point
    long startTime = SystemClock.uptimeMillis();
    final double CURRENT_TIMESTAMP = (double)System.currentTimeMillis() / 1000.0;
    long inferStartTime;  // Declare at method scope for reuse in landmark/gaze detection

    float[][] boxes = null;
    
    // ==== APPLY FISHEYE CORRECTION AND UPDATE DETECTION SETTINGS ====
    DisplayConfig displayConfig = DisplayConfig.getInstance();
    displayConfig.checkAndReload();
    
    // Update smoother parameters from config (allows runtime tuning)
    smoother_list.updateSmoothingParams();
    
    // NOTE: Fisheye correction is now applied AFTER face detection, only to the face region
    // This is more efficient: ~112x112 pixels vs 480x480 (18x fewer pixels!)
    Bitmap processedBitmap = croppedBitmap;
    
    // Update face detection settings from config
    if (tfliteFaceDetector != null) {
      tfliteFaceDetector.setFaceDetectionThreshold(displayConfig.faceDetectionThreshold);
      tfliteFaceDetector.setMinFaceSize(displayConfig.minFaceSize);
    }
    
    // ==== FACE DETECTION ====
    if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
      // Use TFLite Face Detector (from HotGaze - more accurate model)
      inferStartTime = SystemClock.uptimeMillis();
      boxes = tfliteFaceDetector.detectFaces(processedBitmap);
      inferencetime += SystemClock.uptimeMillis() - inferStartTime;
      if (!minimalMode) {
        Log.d("TFLiteFace", "TFLite face detection: " + (boxes != null ? boxes.length : 0) + " faces, threshold=" + displayConfig.faceDetectionThreshold);
      }
    } else {
      // Fallback to QNN face detection (DLC model)
      DetectionUtils.scaleResult scale_result = scale(croppedBitmap, DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_H);
      cropCopyBitmap = scale_result.scaledBitmap;
      float ratioX = scale_result.ratioX;
      float ratioY = scale_result.ratioY;

      // Face detection start (QNN)
      preprocessing(cropCopyBitmap, face_detection_input);
      
      // Run inference with QNN
      int[] inputDims = {1, DemoConfig.face_detection_input_H, DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_C};
      inferStartTime = SystemClock.uptimeMillis();
      NNoutput = face_detection_qnn.runInference("full_image", face_detection_input, inputDims);
      inferencetime += SystemClock.uptimeMillis() - inferStartTime;

      if (NNoutput != null) {
        if (!minimalMode) {
          Log.d("QNN", "Face detection output size: " + NNoutput.length);
        }
        boxes = postprocessing(NNoutput, ratioX, ratioY);
      } else {
        Log.e("QNN", "Face detection inference failed");
      }
    }
    // face detection end

    // Use the same image for all processing (corrected if fisheye enabled)
    // This ensures bounding boxes and landmarks align correctly
    Mat img = bitmap2mat(processedBitmap);
    // Log.d("DEBUG_IMG_SIZE", img.size() + " " + processedBitmap.getWidth() + "x"+processedBitmap.getHeight());

    Vector<float[]> landmarks = new Vector<float[]>();
    Vector<float[]> gazes = new Vector<float[]>();
    Vector<Mat> tvecs = new Vector<Mat>();
    Vector<Mat> rvecs = new Vector<Mat>();
    Mat camera_matrix = null;
    
    // Face crop for display (created from first detected face)
    Bitmap currentFaceCrop = null;
    Mat faceCropMat = null;
    
    // Update face detection status based on TFLite/QNN output
    if (boxes != null && boxes.length != 0) {
      hasFaceDetected = true;  // Face detected by TFLite/QNN
      if (!minimalMode) {
        Log.d("BOXES_SIZE", String.valueOf(boxes.length));
      }
    } else {
      hasFaceDetected = false;  // No face detected
      hasGazeData = false;      // No gaze data if no face
      hasHeadPose = false;      // No head pose if no face
    }
    
    // landmark detection start
    if (boxes != null && boxes.length != 0) {

      smoother_list.autoclean();
      smoother_list.match(boxes);

      // smooth face bbox
      for (int b=0;b<boxes.length;b++) {
        double[] bbox = new double[4];
        for (int ii=0;ii<4;ii++)
          bbox[ii] = (double)boxes[b][ii];
        // Log.d("SMOOTH_DEBUG_BEFORE", bbox[0] + " " + bbox[1] + " " + bbox[2] + " " + bbox[3]);
        bbox = smoother_list.smooth(bbox, b, CURRENT_TIMESTAMP);
        // Log.d("SMOOTH_DEBUG_AFTER", bbox[0] + " " + bbox[1] + " " + bbox[2] + " " + bbox[3]);
        for (int ii=0;ii<4;ii++)
          boxes[b][ii] = (float)bbox[ii];
      }

      for (int b=0;b<boxes.length;b++) {
        float[] box = boxes[b];
        
        // Create high-resolution face crop with padding for the first face
        if (!minimalMode && b == 0) {
          float padding = displayConfig.faceCropPadding;
          int displaySize = displayConfig.faceCropDisplaySize;
          currentFaceCrop = extractFaceCrop(processedBitmap, box, padding, displaySize);
          if (currentFaceCrop != null) {
            faceCropMat = bitmap2mat(currentFaceCrop);
          }
        }
        
        // Preprocess face for landmark detection
        // If fisheye enabled, correction is applied to the 112x112 face crop (much faster than 480x480!)
        ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
        
        // Apply fisheye correction to the face crop if enabled
        if (fisheyeUiEnabled && displayConfig.fisheyeEnabled) {
          // The landmark input is a 112x112x3 float array, convert to bitmap, correct, convert back
          Bitmap faceCropForCorrection = floatArrayToBitmap(landmark_preprocess_result.input, 112, 112);
          Bitmap correctedFaceCrop = applyFisheyeCorrection(faceCropForCorrection, displayConfig.fisheyeStrength, displayConfig.fisheyeZoom);
          landmark_preprocess_result.input = bitmapToFloatArray(correctedFaceCrop);
        }
        
        float[] landmark_detection_input = landmark_preprocess_result.input;
        
        // ==== LANDMARK DETECTION with QNN ====
        int[] landmarkInputDims = {1, DemoConfig.landmark_detection_input_H, DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
        inferStartTime = SystemClock.uptimeMillis();
        NNoutput = landmark_detection_qnn.runInference("face_image", landmark_detection_input, landmarkInputDims);
        inferencetime += SystemClock.uptimeMillis() - inferStartTime;
        
        if (NNoutput != null) {
            if (!minimalMode) {
              Log.d("QNN", "Landmark detection output size: " + NNoutput.length);
            }
            float[] landmark = landmark_postprocess(landmark_preprocess_result, NNoutput);

            double[] landmark_post = new double[landmark.length];
            for (int ii=0;ii<landmark.length;ii++)
              landmark_post[ii] = (double)landmark[ii];
            landmark_post = smoother_list.smooth(landmark_post, b, CURRENT_TIMESTAMP);
            for (int ii=0;ii<landmark.length;ii++)
              landmark[ii] = (float)landmark_post[ii];

            landmarks.addElement(landmark);

            // ==== GAZE ESTIMATION with QNN ====
            ProcessFactory.GazePreprocessResult gaze_preprocess_result = gaze_preprocess(img, landmark);
            rvecs.addElement(gaze_preprocess_result.rvec);
            tvecs.addElement(gaze_preprocess_result.tvec);
            camera_matrix = gaze_preprocess_result.camera_matrix;
            
            // Prepare multi-input for gaze estimation
            String[] inputNames = {"left_eye", "right_eye", "face"};
            float[][] inputDataArrays = {
                gaze_preprocess_result.leye,
                gaze_preprocess_result.reye,
                gaze_preprocess_result.face
            };
            int[][] inputDimsArrays = {
                {1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C},
                {1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C},
                {1, DemoConfig.gaze_estimation_face_input_H, DemoConfig.gaze_estimation_face_input_W, DemoConfig.gaze_estimation_face_input_C}
            };
            
            inferStartTime = SystemClock.uptimeMillis();
            float[] gazeOutput = gaze_estimation_qnn.runMultiInputInference(inputNames, inputDataArrays, inputDimsArrays);
            inferencetime += SystemClock.uptimeMillis() - inferStartTime;
            
            if (gazeOutput != null) {
                // Log.d("QNN", "Gaze estimation output: " + gazeOutput[0] + " " + gazeOutput[1]);
                float[] gaze_pitchyaw = gaze_postprocess(gazeOutput, gaze_preprocess_result.R);

                double[] gaze_pitchyaw_post = new double[gaze_pitchyaw.length];
                for (int ii=0;ii<gaze_pitchyaw_post.length;ii++)
                  gaze_pitchyaw_post[ii] = (double)gaze_pitchyaw[ii];
                gaze_pitchyaw_post = smoother_list.smooth(gaze_pitchyaw_post, b, CURRENT_TIMESTAMP);
                for (int ii=0;ii<gaze_pitchyaw_post.length;ii++)
                  gaze_pitchyaw[ii] = (float)gaze_pitchyaw_post[ii];

                gazes.addElement(gaze_pitchyaw);
                
                // Store latest gaze for display (use first detected face)
                if (b == 0) {
                  latestGazePitch = gaze_pitchyaw[0];
                  latestGazeYaw = gaze_pitchyaw[1];
                  hasGazeData = true;

                  // Store latest head pose for display (use first detected face)
                  float[] headPose = extractHeadPose(gaze_preprocess_result.rvec);
                  latestHeadPitch = headPose[0];
                  latestHeadYaw = headPose[1];
                  latestHeadRoll = headPose[2];
                  hasHeadPose = true;
                  
                  // Run Looking Classifier to determine if user is looking at camera
                  if (lookingClassifier != null && lookingClassifier.isInitialized()) {
                    if (!minimalMode) {
                      Log.d("LookingClassifier", "Calling predict with gaze: " + gaze_pitchyaw[0] + ", " + gaze_pitchyaw[1]);
                    }
                    // IMPORTANT:
                    // The v2 model was trained on the SAME raw features we record (see recordFrame()):
                    // - gaze_pitch/yaw from gaze_postprocess (radians)
                    // - head pose from gaze_preprocess_result.rvec (same camera_matrix convention)
                    // - landmarks from landmark_postprocess() (image coordinates of the SAME processed frame)
                    //
                    // So we must feed the classifier the *exact same* raw values without any extra remapping.
                    isLookingAtCamera = lookingClassifier.predict(gaze_pitchyaw, gaze_preprocess_result.rvec, landmark);
                    lookingProbability = lookingClassifier.getLastProbability();
                    if (!minimalMode) {
                      Log.d("LookingClassifier", "Result: looking=" + isLookingAtCamera + ", prob=" + lookingProbability);
                    }
                  } else {
                    if (!minimalMode) {
                      Log.w("LookingClassifier", "Classifier not available: null=" + (lookingClassifier == null));
                    }
                  }
                  
                  // Record frame if recording is active (now includes head pose)
                  recordFrame(landmark, gaze_pitchyaw, gaze_preprocess_result.rvec);
                }
            } else {
                Log.e("QNN", "Gaze estimation inference failed");
            }
        } else {
            Log.e("QNN", "Landmark detection inference failed");
        }
      }
      // landamrk detection end
      if (!minimalMode) {
        for (int i=0;i<gazes.size();i++) {
          drawgaze(img, gazes.elementAt(i), landmarks.elementAt(i));
          drawheadpose(img, rvecs.elementAt(i), tvecs.elementAt(i), camera_matrix);
        }
        Utils.matToBitmap(img, processedBitmap);
        processedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
        drawbox(pixs, boxes, DemoConfig.crop_H, DemoConfig.crop_W);
        for (float[] landmark : landmarks) {
          drawlandmark(pixs, landmark, DemoConfig.crop_H, DemoConfig.crop_W);
        }
      }
      
      // Draw on face crop for display (landmarks relative to face crop)
      if (!minimalMode && currentFaceCrop != null && faceCropMat != null && gazes.size() > 0) {
        // Draw gaze and landmarks on face crop
        // We need to transform landmark coordinates to face crop space
        float[] box = boxes[0];
        float padding = displayConfig.faceCropPadding;
        float boxW = box[2] - box[0];
        float boxH = box[3] - box[1];
        float cropSize = Math.max(boxW, boxH) * padding;
        float cropX = box[0] + boxW/2 - cropSize/2;
        float cropY = box[1] + boxH/2 - cropSize/2;
        float scale = (float)displayConfig.faceCropDisplaySize / cropSize;
        
        // Draw gaze on face crop mat
        if (landmarks.size() > 0 && gazes.size() > 0) {
          // Transform landmarks to face crop coordinates
          float[] origLandmarks = landmarks.elementAt(0);
          float[] cropLandmarks = new float[origLandmarks.length];
          for (int i = 0; i < origLandmarks.length / 2; i++) {
            cropLandmarks[i*2] = (origLandmarks[i*2] - cropX) * scale;
            cropLandmarks[i*2+1] = (origLandmarks[i*2+1] - cropY) * scale;
          }
          drawgaze(faceCropMat, gazes.elementAt(0), cropLandmarks);
        }
        Utils.matToBitmap(faceCropMat, currentFaceCrop);
        faceCropBitmap = currentFaceCrop;
      }
    } else {
      if (!minimalMode) {
        processedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
      }
      faceCropBitmap = null;  // No face detected
    }

    //transpose
    if (!minimalMode) {
      if(!DemoConfig.USE_VERTICAL) {
        transpose(pixs, pixs_out, DemoConfig.crop_H, DemoConfig.crop_W);
        detection.setPixels(pixs_out, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
      }
      else{
        detection.setPixels(pixs, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
      }
    }
    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    latency=lastProcessingTimeMs;

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            if (classifier != null) {
              final long startTime = SystemClock.uptimeMillis();
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

              if (bitmapset == 0){
                // Check if config file has been modified and reload
                DisplayConfig displayConfig = DisplayConfig.getInstance();
                displayConfig.checkAndReload();
                final float detectionRotation = displayConfig.detectionRotation;
                final float detectionScaleX = displayConfig.detectionScaleX;
                final float detectionScaleY = displayConfig.detectionScaleY;
                
                // Capture face crop for display
                final Bitmap faceCropToDisplay = faceCropBitmap;
                
                // Log every 30 frames to avoid spam
                if (System.currentTimeMillis() % 1000 < 50) {
                    Log.i("DisplayConfig", "APPLYING detection: rotation=" + detectionRotation + 
                        ", scaleX=" + detectionScaleX + ", scaleY=" + detectionScaleY +
                        ", faceCrop=" + (faceCropToDisplay != null ? "yes" : "no"));
                }
                
                runOnUiThread(
                        new Runnable() {
                          @Override
                          public void run() {
                            if (!minimalModeEnabled) {
                              // Region preview (left half)
                              ImageView imageView = (ImageView) findViewById(R.id.imageView2);
                              if (imageView != null) {
                                imageView.setRotation(detectionRotation);
                                imageView.setScaleX(detectionScaleX);
                                imageView.setScaleY(detectionScaleY);
                                imageView.setImageBitmap(detection);
                              }
                              
                              // Face crop preview
                              ImageView faceView = (ImageView) findViewById(R.id.faceImageView);
                              if (faceView != null) {
                                if (faceCropToDisplay != null) {
                                  faceView.setImageBitmap(faceCropToDisplay);
                                } else {
                                  // No face detected - show placeholder or clear
                                  faceView.setImageBitmap(null);
                                }
                              }
                            }
                          }
                        });
//                bitmapset = 1;
              }
              // Capture gaze data for UI thread
              final boolean displayGaze = hasGazeData;
              final float displayPitch = latestGazePitch;
              final float displayYaw = latestGazeYaw;
              final boolean calibrated = isCalibrated;
              final float refPitch = referencePitch;
              final float refYaw = referenceYaw;
              
              // Apply exponential smoothing to gaze values (reduces jitter from head movement)
              // Apply display smoothing (exponential moving average)
              // Uses display_smoothing_alpha from DisplayConfig for additional stability
              if (displayGaze) {
                // Use the lower of the two alphas for maximum smoothing
                DisplayConfig smoothingConfig = DisplayConfig.getInstance();
                float alpha = Math.min(gazeConfig.smoothingAlpha, smoothingConfig.displaySmoothingAlpha);
                smoothedPitch = alpha * displayPitch + (1 - alpha) * smoothedPitch;
                smoothedYaw = alpha * displayYaw + (1 - alpha) * smoothedYaw;
              }
              
              // Check if looking at camera
              // Get thresholds from DisplayConfig (can be modified at runtime)
              DisplayConfig gazeThresholds = DisplayConfig.getInstance();
              boolean currentFrameLooking = false;
              boolean useMLClassifierForDetection = lookingClassifier != null && lookingClassifier.isInitialized();
              
              // ML Classifier takes priority - use its result directly
              if (useMLClassifierForDetection && displayGaze) {
                // ML classifier already computed isLookingAtCamera in processImage() at line ~882
                currentFrameLooking = isLookingAtCamera;
                
                // Log ML result (every ~1 second)
                if (System.currentTimeMillis() % 1000 < 50) {
                  Log.i("GazeML", String.format(
                    "ML Classifier: Looking=%b, Probability=%.1f%%",
                    isLookingAtCamera, lookingProbability * 100));
                }
              } else if (calibrated && displayGaze) {
                // Fallback to threshold-based detection (only if calibrated)
                float pitchDiff = Math.abs(smoothedPitch - refPitch);
                float yawDiff = Math.abs(smoothedYaw - refYaw);
                
                float pitchThreshold = gazeThresholds.gazePitchThreshold;
                float yawThreshold = gazeThresholds.gazeYawThreshold;
                boolean pitchOnly = gazeThresholds.gazePitchOnly;
                
                // Log for debugging (every ~1 second)
                if (System.currentTimeMillis() % 1000 < 50) {
                  Log.i("GazeThreshold", String.format(
                    "Pitch: %.1f° (diff: %.1f°, thr: %.1f°) | Yaw: %.1f° (diff: %.1f°, thr: %.1f°) | PitchOnly: %s",
                    Math.toDegrees(smoothedPitch), Math.toDegrees(pitchDiff), Math.toDegrees(pitchThreshold),
                    Math.toDegrees(smoothedYaw), Math.toDegrees(yawDiff), Math.toDegrees(yawThreshold),
                    pitchOnly ? "YES" : "NO"));
                }
                
                if (pitchOnly) {
                  currentFrameLooking = (pitchDiff < pitchThreshold);
                } else {
                  currentFrameLooking = (pitchDiff < pitchThreshold) && (yawDiff < yawThreshold);
                }
              }
              
              // Temporal smoothing - require consecutive frames to change state
              int smoothingFrames = gazeThresholds.gazeConsecutiveFrames;
              if (useMLClassifierForDetection) {
                // ML classifier: minimal smoothing for stability (2 frames)
                if (currentFrameLooking) {
                  lookingAtCameraCount++;
                  notLookingAtCameraCount = 0;
                  if (lookingAtCameraCount >= 2) {
                    isLookingAtCamera = true;
                  }
                } else {
                  notLookingAtCameraCount++;
                  lookingAtCameraCount = 0;
                  if (notLookingAtCameraCount >= 2) {
                    isLookingAtCamera = false;
                  }
                }
              } else {
                // Threshold-based: use configured smoothing frames
                if (currentFrameLooking) {
                  lookingAtCameraCount++;
                  notLookingAtCameraCount = 0;
                  if (lookingAtCameraCount >= smoothingFrames) {
                    isLookingAtCamera = true;
                  }
                } else {
                  notLookingAtCameraCount++;
                  lookingAtCameraCount = 0;
                  if (notLookingAtCameraCount >= smoothingFrames) {
                    isLookingAtCamera = false;
                  }
                }
              }
              
              final boolean lookingAtCam = isLookingAtCamera;
              
              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      showFrameInfo(String.format(Locale.US, "%.1f FPS", processedFps));
                      showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                      showCameraResolution(canvas.getWidth() + "x" + canvas.getHeight());
                      showRotationInfo(String.valueOf(sensorOrientation));
                      showInference(latency + "/" + inferencetime + "ms");
                      
                      // Display pitch and yaw values
                      if (displayGaze) {
                        showGazePitchYaw(displayPitch, displayYaw);
                      } else {
                        clearGazePitchYaw();
                      }

                      // Display head pose (pitch/yaw/roll)
                      if (headPosePitchText != null && headPoseYawText != null && headPoseRollText != null) {
                        if (hasHeadPose) {
                          headPosePitchText.setText(String.format("Head P: %.1f°", Math.toDegrees(latestHeadPitch)));
                          headPoseYawText.setText(String.format("Head Y: %.1f°", Math.toDegrees(latestHeadYaw)));
                          headPoseRollText.setText(String.format("Head R: %.1f°", Math.toDegrees(latestHeadRoll)));
                        } else {
                          headPosePitchText.setText("Head P: --");
                          headPoseYawText.setText("Head Y: --");
                          headPoseRollText.setText("Head R: --");
                        }
                      }
                      
                      // Update gaze overlay - ML classifier takes priority
                      boolean useMLClassifier = lookingClassifier != null && lookingClassifier.isInitialized();
                      
                      // DEBUG: Log ML classifier status
                      if (!minimalModeEnabled) {
                        Log.d("GazeUI", "useMLClassifier=" + useMLClassifier + 
                            ", calibrated=" + calibrated + 
                            ", displayGaze=" + displayGaze +
                            ", lookingAtCam=" + lookingAtCam +
                            ", prob=" + lookingProbability);
                      }
                      
                      if (gazeOverlay != null && gazeStatusText != null) {
                        // Always show overlay when ML classifier is active OR when calibrated
                        if (useMLClassifier || calibrated) {
                          if (displayGaze) {
                            // Always show both percentages (per frame)
                            float pLooking = Math.max(0f, Math.min(1f, lookingProbability));
                            float pNotLooking = 1f - pLooking;
                            String pctText = String.format("Looking %.0f%% | Not %.0f%%", pLooking * 100f, pNotLooking * 100f);

                            if (lookingAtCam) {
                              // Looking at camera - green overlay
                              gazeOverlay.setBackgroundColor(0x6000FF00);  // Semi-transparent green
                              gazeStatusText.setText("✓ " + pctText);
                              gazeStatusText.setTextColor(0xFF00FF00);
                            } else {
                              // Not looking at camera - red overlay
                              gazeOverlay.setBackgroundColor(0x60FF0000);  // Semi-transparent red
                              gazeStatusText.setText("✗ " + pctText);
                              gazeStatusText.setTextColor(0xFFFF0000);
                            }

                            // Also show it under the calibrate button
                            if (lookingProbText != null) {
                              final String headText =
                                  hasHeadPose
                                      ? String.format(
                                          Locale.US,
                                          "Head P %.0f° | Y %.0f° | R %.0f°",
                                          Math.toDegrees(latestHeadPitch),
                                          Math.toDegrees(latestHeadYaw),
                                          Math.toDegrees(latestHeadRoll))
                                      : "Head P -- | Y -- | R --";
                              lookingProbText.setText(pctText + "\n" + headText);
                            }

                            // Logcat (per processed frame)
                            if (!minimalModeEnabled) {
                              Log.d("LookingProb", pctText + " (looking=" + lookingAtCam + ")");
                            }
                          } else {
                            // No gaze yet (or no face) - yellow overlay
                            gazeOverlay.setBackgroundColor(0x60FFFF00);  // Semi-transparent yellow

                            final String statusText =
                                hasFaceDetected
                                    ? "Face detected, waiting for gaze/classifier..."
                                    : "No Face Detected";

                            gazeStatusText.setText(statusText);
                            gazeStatusText.setTextColor(0xFFFFFF00);

                            if (lookingProbText != null) {
                              final String headText =
                                  hasHeadPose
                                      ? String.format(
                                          Locale.US,
                                          "Head P %.0f° | Y %.0f° | R %.0f°",
                                          Math.toDegrees(latestHeadPitch),
                                          Math.toDegrees(latestHeadYaw),
                                          Math.toDegrees(latestHeadRoll))
                                      : "Head P -- | Y -- | R --";
                              lookingProbText.setText(statusText + "\n" + headText);
                            }

                            // Logcat (per processed frame)
                            if (!minimalModeEnabled) {
                              Log.d("LookingProb", statusText + " (hasFace=" + hasFaceDetected + ")");
                            }
                          }
                          // ALWAYS set visibility when ML or calibrated
                          gazeOverlay.setVisibility(View.VISIBLE);
                          gazeStatusText.setVisibility(View.VISIBLE);
                        }
                      } else {
                        Log.e("GazeUI", "gazeOverlay or gazeStatusText is NULL!");
                      }
                    }
                  });
            }
            readyForNextImage();
          }
        });
    return croppedBitmap;
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (croppedBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("ClassifierActivity onDestroy");
    
    // Stop recording if active
    if (isRecording) {
      stopRecording();
    }
    
    // Cleanup TFLite Face Detector
    if (tfliteFaceDetector != null) {
      tfliteFaceDetector.close();
      tfliteFaceDetector = null;
      LOGGER.d("TFLite Face Detector closed");
    }
    
    // Cleanup Looking Classifier
    if (lookingClassifier != null) {
      lookingClassifier.close();
      lookingClassifier = null;
      LOGGER.d("Looking Classifier closed");
    }
    
    // Cleanup QNN Model Runners
    if (face_detection_qnn != null) {
      face_detection_qnn.close();
      face_detection_qnn = null;
      LOGGER.d("QNN Face Detection closed");
    }
    if (landmark_detection_qnn != null) {
      landmark_detection_qnn.close();
      landmark_detection_qnn = null;
      LOGGER.d("QNN Landmark Detection closed");
    }
    if (gaze_estimation_qnn != null) {
      gaze_estimation_qnn.close();
      gaze_estimation_qnn = null;
      LOGGER.d("QNN Gaze Estimation closed");
    }
    
    super.onDestroy();
  }
  
  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU && model == Model.QUANTIZED) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                .show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }
  
  /**
   * Extract a face crop from the source bitmap with padding
   * 
   * @param source Source bitmap
   * @param box Face bounding box [x1, y1, x2, y2]
   * @param padding Padding multiplier (2.0 = 2x the box size)
   * @param displaySize Target size for the crop
   * @return Face crop bitmap scaled to displaySize
   */
  private Bitmap extractFaceCrop(Bitmap source, float[] box, float padding, int displaySize) {
    try {
      int srcW = source.getWidth();
      int srcH = source.getHeight();
      
      // Calculate box dimensions
      float boxX1 = box[0];
      float boxY1 = box[1];
      float boxX2 = box[2];
      float boxY2 = box[3];
      float boxW = boxX2 - boxX1;
      float boxH = boxY2 - boxY1;
      
      // Calculate crop size with padding (square crop)
      float cropSize = Math.max(boxW, boxH) * padding;
      
      // Calculate crop center (center of face box)
      float centerX = boxX1 + boxW / 2;
      float centerY = boxY1 + boxH / 2;
      
      // Calculate crop region
      int cropX = (int)(centerX - cropSize / 2);
      int cropY = (int)(centerY - cropSize / 2);
      int cropW = (int)cropSize;
      int cropH = (int)cropSize;
      
      // Clamp to image bounds
      cropX = Math.max(0, cropX);
      cropY = Math.max(0, cropY);
      if (cropX + cropW > srcW) cropW = srcW - cropX;
      if (cropY + cropH > srcH) cropH = srcH - cropY;
      
      // Ensure minimum size
      if (cropW < 10 || cropH < 10) {
        return null;
      }
      
      // Extract the crop
      Bitmap crop = Bitmap.createBitmap(source, cropX, cropY, cropW, cropH);
      
      // Scale to display size
      Bitmap scaled = Bitmap.createScaledBitmap(crop, displaySize, displaySize, true);
      
      return scaled;
    } catch (Exception e) {
      LOGGER.e(e, "Failed to extract face crop");
      return null;
    }
  }
  
  /**
   * Extract a region of interest from the source bitmap
   * Used for wide FOV cameras where only a portion of the image is needed
   * 
   * @param source Source bitmap
   * @param offsetX Horizontal offset (0.0 = left edge, 0.5 = center, 1.0 = right edge)
   * @param offsetY Vertical offset (0.0 = top, 0.5 = center, 1.0 = bottom)
   * @param scale Scale factor (0.5 = use half width, 1.0 = use full width)
   * @return Cropped region bitmap
   */
  private Bitmap extractRegion(Bitmap source, float offsetX, float offsetY, float scale) {
    int srcW = source.getWidth();
    int srcH = source.getHeight();
    
    // Calculate region size based on scale
    int regionW = (int) (srcW * scale);
    int regionH = (int) (srcH * scale);
    
    // Ensure minimum size
    regionW = Math.max(regionW, 100);
    regionH = Math.max(regionH, 100);
    
    // Calculate region position based on offset
    // offset 0.0 = region starts at left/top edge
    // offset 0.5 = region is centered
    // offset 1.0 = region ends at right/bottom edge
    int x = (int) ((srcW - regionW) * offsetX);
    int y = (int) ((srcH - regionH) * offsetY);
    
    // Clamp to valid bounds
    x = Math.max(0, Math.min(x, srcW - regionW));
    y = Math.max(0, Math.min(y, srcH - regionH));
    
    try {
      return Bitmap.createBitmap(source, x, y, regionW, regionH);
    } catch (Exception e) {
      LOGGER.e(e, "Failed to extract region, using full image");
      return source;
    }
  }

  private static final class RegionResult {
    final Bitmap bitmap;
    final int x;
    final int y;
    final int w;
    final int h;
    RegionResult(Bitmap bitmap, int x, int y, int w, int h) {
      this.bitmap = bitmap;
      this.x = x;
      this.y = y;
      this.w = w;
      this.h = h;
    }
  }

  /**
   * Same as extractRegion(), but also returns the region origin/size in the source bitmap.
   */
  private RegionResult extractRegionWithInfo(Bitmap source, float offsetX, float offsetY, float scale) {
    int srcW = source.getWidth();
    int srcH = source.getHeight();

    int regionW = (int) (srcW * scale);
    int regionH = (int) (srcH * scale);
    regionW = Math.max(regionW, 100);
    regionH = Math.max(regionH, 100);

    int x = (int) ((srcW - regionW) * offsetX);
    int y = (int) ((srcH - regionH) * offsetY);
    x = Math.max(0, Math.min(x, srcW - regionW));
    y = Math.max(0, Math.min(y, srcH - regionH));

    try {
      Bitmap region = Bitmap.createBitmap(source, x, y, regionW, regionH);
      return new RegionResult(region, x, y, regionW, regionH);
    } catch (Exception e) {
      LOGGER.e(e, "Failed to extract region, using full image");
      return new RegionResult(source, 0, 0, srcW, srcH);
    }
  }

  /**
   * Map landmarks from 480x480 crop coordinates back into full preview-frame coordinates.
   *
   * @param cropLandmarks 196 floats (98 points x,y) in crop space
   * @param cropToRegionTransform inverse matrix of regionToCropTransform
   * @param regionX x origin of the region inside the full preview frame
   * @param regionY y origin of the region inside the full preview frame
   */
  private float[] mapCropLandmarksToFullFrame(
      float[] cropLandmarks, Matrix cropToRegionTransform, int regionX, int regionY) {
    float[] out = new float[cropLandmarks.length];
    float[] pt = new float[2];
    for (int i = 0; i < cropLandmarks.length / 2; i++) {
      pt[0] = cropLandmarks[i * 2];
      pt[1] = cropLandmarks[i * 2 + 1];
      cropToRegionTransform.mapPoints(pt);
      out[i * 2] = pt[0] + regionX;
      out[i * 2 + 1] = pt[1] + regionY;
    }
    return out;
  }
  
  /**
   * Convert float array (RGB normalized 0-1) to Bitmap
   * Used for applying image processing to model input
   */
  private Bitmap floatArrayToBitmap(float[] data, int width, int height) {
    int[] pixels = new int[width * height];
    for (int i = 0; i < pixels.length; i++) {
      int r = (int) (data[i * 3] * 255);
      int g = (int) (data[i * 3 + 1] * 255);
      int b = (int) (data[i * 3 + 2] * 255);
      r = Math.max(0, Math.min(255, r));
      g = Math.max(0, Math.min(255, g));
      b = Math.max(0, Math.min(255, b));
      pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
    }
    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
    return bitmap;
  }
  
  /**
   * Convert Bitmap to float array (RGB normalized 0-1)
   * Used for converting processed images back to model input format
   */
  private float[] bitmapToFloatArray(Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int[] pixels = new int[width * height];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
    float[] data = new float[width * height * 3];
    for (int i = 0; i < pixels.length; i++) {
      int pixel = pixels[i];
      data[i * 3] = ((pixel >> 16) & 0xFF) / 255.0f;
      data[i * 3 + 1] = ((pixel >> 8) & 0xFF) / 255.0f;
      data[i * 3 + 2] = (pixel & 0xFF) / 255.0f;
    }
    return data;
  }
  
  /**
   * Apply fisheye lens correction to the image
   * This helps with landmark/gaze accuracy on wide-angle/fisheye cameras like OX05B1S
   * Now applied to face crop (112x112) instead of full image (480x480) for efficiency!
   * 
   * @param input Input bitmap with fisheye distortion
   * @param strength Correction strength (0.0 = no correction, 1.0 = full correction)
   * @param zoom Zoom factor after correction (1.0 = no zoom, >1.0 = zoom in)
   * @return Corrected bitmap
   */
  private Bitmap applyFisheyeCorrection(Bitmap input, float strength, float zoom) {
    if (strength <= 0) {
      return input;
    }
    
    try {
      Mat src = new Mat();
      Utils.bitmapToMat(input, src);
      
      int width = src.cols();
      int height = src.rows();
      float cx = width / 2.0f;
      float cy = height / 2.0f;
      
      // Create camera matrix (approximate for fisheye)
      float focalLength = Math.min(width, height) * 0.8f;
      Mat cameraMatrix = Mat.eye(3, 3, org.opencv.core.CvType.CV_64F);
      cameraMatrix.put(0, 0, focalLength);
      cameraMatrix.put(1, 1, focalLength);
      cameraMatrix.put(0, 2, cx);
      cameraMatrix.put(1, 2, cy);
      
      // Create distortion coefficients (barrel distortion for fisheye)
      // k1, k2, p1, p2, k3 - negative values for barrel distortion
      Mat distCoeffs = Mat.zeros(4, 1, org.opencv.core.CvType.CV_64F);
      float k = -strength * 0.3f;  // Scale strength to reasonable distortion values
      distCoeffs.put(0, 0, k);     // k1
      distCoeffs.put(1, 0, k * 0.5f);  // k2
      distCoeffs.put(2, 0, 0);     // p1
      distCoeffs.put(3, 0, 0);     // p2
      
      // Create new camera matrix with zoom
      Mat newCameraMatrix = cameraMatrix.clone();
      newCameraMatrix.put(0, 0, focalLength * zoom);
      newCameraMatrix.put(1, 1, focalLength * zoom);
      
      // Apply undistortion
      Mat dst = new Mat();
      org.opencv.calib3d.Calib3d.undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix);
      
      // Convert back to bitmap
      Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
      Utils.matToBitmap(dst, result);
      
      // Cleanup
      src.release();
      dst.release();
      cameraMatrix.release();
      distCoeffs.release();
      newCameraMatrix.release();
      
      return result;
    } catch (Exception e) {
      LOGGER.e(e, "Fisheye correction failed");
      return input;
    }
  }
}
