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
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
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
  private boolean isLookingAtCamera = false;
  
  // Running average for gaze (reduces jitter)
  private float smoothedPitch = 0f;
  private float smoothedYaw = 0f;
  
  // UI elements for gaze tracking
  private View gazeOverlay;
  private TextView gazeStatusText;
  private Button calibrateButton;
  @Override

  protected Bitmap processImage() {
    inferencetime = 0;
    // NOTE: We do NOT reset hasFaceDetected and hasGazeData here anymore
    // They persist between frames so the calibrate button can read them reliably
    
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
    Bitmap regionBitmap = extractRegion(rgbFrameBitmap, 
        cropConfig.cropOffsetX, cropConfig.cropOffsetY, cropConfig.cropScale);
    
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
    
    // Apply fisheye undistortion if enabled
    Bitmap processedBitmap = croppedBitmap;
    if (displayConfig.fisheyeEnabled) {
      processedBitmap = applyFisheyeCorrection(croppedBitmap, displayConfig.fisheyeStrength, displayConfig.fisheyeZoom);
    }
    
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
      Log.d("TFLiteFace", "TFLite face detection: " + (boxes != null ? boxes.length : 0) + " faces, threshold=" + displayConfig.faceDetectionThreshold);
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
        Log.d("QNN", "Face detection output size: " + NNoutput.length);
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
      Log.d("BOXES_SIZE", String.valueOf(boxes.length));
    } else {
      hasFaceDetected = false;  // No face detected
      hasGazeData = false;      // No gaze data if no face
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
        if (b == 0) {
          float padding = displayConfig.faceCropPadding;
          int displaySize = displayConfig.faceCropDisplaySize;
          currentFaceCrop = extractFaceCrop(processedBitmap, box, padding, displaySize);
          if (currentFaceCrop != null) {
            faceCropMat = bitmap2mat(currentFaceCrop);
          }
        }
        
        ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
        float[] landmark_detection_input = landmark_preprocess_result.input;
        
        // ==== LANDMARK DETECTION with QNN ====
        int[] landmarkInputDims = {1, DemoConfig.landmark_detection_input_H, DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
        inferStartTime = SystemClock.uptimeMillis();
        NNoutput = landmark_detection_qnn.runInference("face_image", landmark_detection_input, landmarkInputDims);
        inferencetime += SystemClock.uptimeMillis() - inferStartTime;
        
        if (NNoutput != null) {
            Log.d("QNN", "Landmark detection output size: " + NNoutput.length);
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
                }
            } else {
                Log.e("QNN", "Gaze estimation inference failed");
            }
        } else {
            Log.e("QNN", "Landmark detection inference failed");
        }
      }
      // landamrk detection end
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
      
      // Draw on face crop for display (landmarks relative to face crop)
      if (currentFaceCrop != null && faceCropMat != null && gazes.size() > 0) {
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
      processedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
      faceCropBitmap = null;  // No face detected
    }

    //transpose
    if(!DemoConfig.USE_VERTICAL) {
      transpose(pixs, pixs_out, DemoConfig.crop_H, DemoConfig.crop_W);
      detection.setPixels(pixs_out, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
    }
    else{
      detection.setPixels(pixs, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
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
                            // Region preview (left half)
                            ImageView imageView = (ImageView) findViewById(R.id.imageView2);
                            imageView.setRotation(detectionRotation);
                            imageView.setScaleX(detectionScaleX);
                            imageView.setScaleY(detectionScaleY);
                            imageView.setImageBitmap(detection);
                            
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
              // Uses smoothing_alpha from config file
              if (displayGaze) {
                float alpha = gazeConfig.smoothingAlpha;
                smoothedPitch = alpha * displayPitch + (1 - alpha) * smoothedPitch;
                smoothedYaw = alpha * displayYaw + (1 - alpha) * smoothedYaw;
              }
              
              // Check if looking at camera with adaptive threshold
              // Uses threshold values from config file
              boolean currentFrameLooking = false;
              if (calibrated && displayGaze) {
                float pitchDiff = Math.abs(smoothedPitch - refPitch);
                float yawDiff = Math.abs(smoothedYaw - refYaw);
                
                // Use adaptive threshold from config - more tolerant overall to handle head movement
                float thresholdBase = gazeConfig.thresholdBase;
                float thresholdMax = gazeConfig.thresholdMax;
                float threshold = thresholdBase;
                
                // If one axis is close, be more lenient on the other (natural compensation)
                if (pitchDiff < thresholdBase || yawDiff < thresholdBase) {
                  threshold = thresholdMax;
                }
                
                currentFrameLooking = (pitchDiff < threshold) && (yawDiff < threshold);
              }
              
              // Temporal smoothing - require consecutive frames to change state
              // Uses smoothing_frames from config file
              int smoothingFrames = gazeConfig.smoothingFrames;
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
              
              final boolean lookingAtCam = isLookingAtCamera;
              
              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      showFrameInfo(rgbFrameBitmap.getWidth() + "x" + rgbFrameBitmap.getHeight());
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
                      
                      // Update gaze overlay if calibrated
                      if (calibrated && gazeOverlay != null && gazeStatusText != null) {
                        if (displayGaze) {
                          if (lookingAtCam) {
                            // Looking at camera - green overlay
                            gazeOverlay.setBackgroundColor(0x6000FF00);  // Semi-transparent green
                            gazeStatusText.setText("✓ Looking at Camera");
                            gazeStatusText.setTextColor(0xFF00FF00);
                          } else {
                            // Not looking at camera - red overlay
                            gazeOverlay.setBackgroundColor(0x60FF0000);  // Semi-transparent red
                            gazeStatusText.setText("✗ Not Looking at Camera");
                            gazeStatusText.setTextColor(0xFFFF0000);
                          }
                          gazeOverlay.setVisibility(View.VISIBLE);
                          gazeStatusText.setVisibility(View.VISIBLE);
                        } else {
                          // No face detected
                          gazeOverlay.setBackgroundColor(0x60FFFF00);  // Semi-transparent yellow
                          gazeStatusText.setText("No Face Detected");
                          gazeStatusText.setTextColor(0xFFFFFF00);
                        }
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
    
    // Cleanup TFLite Face Detector
    if (tfliteFaceDetector != null) {
      tfliteFaceDetector.close();
      tfliteFaceDetector = null;
      LOGGER.d("TFLite Face Detector closed");
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
  
  /**
   * Apply fisheye lens correction to the image
   * This helps with face detection accuracy on wide-angle/fisheye cameras like OX05B1S
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
