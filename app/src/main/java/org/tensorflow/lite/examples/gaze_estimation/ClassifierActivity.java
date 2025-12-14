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

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            DemoConfig.crop_W,
            DemoConfig.crop_H,
            DemoConfig.img_orientation,
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

    //transform and crop the frame
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

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
    
    // ==== FACE DETECTION ====
    if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
      // Use TFLite Face Detector (from HotGaze - more accurate model)
      inferStartTime = SystemClock.uptimeMillis();
      boxes = tfliteFaceDetector.detectFaces(croppedBitmap);
      inferencetime += SystemClock.uptimeMillis() - inferStartTime;
      Log.d("TFLiteFace", "TFLite face detection: " + (boxes != null ? boxes.length : 0) + " faces");
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

    Mat img = bitmap2mat(croppedBitmap);
    // Log.d("DEBUG_IMG_SIZE", img.size() + " " + croppedBitmap.getWidth() + "x"+croppedBitmap.getHeight() + " " + cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());

    Vector<float[]> landmarks = new Vector<float[]>();
    Vector<float[]> gazes = new Vector<float[]>();
    Vector<Mat> tvecs = new Vector<Mat>();
    Vector<Mat> rvecs = new Vector<Mat>();
    Mat camera_matrix = null;
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
      Utils.matToBitmap(img, croppedBitmap);
      croppedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
      drawbox(pixs, boxes, DemoConfig.crop_H, DemoConfig.crop_W);
      for (float[] landmark : landmarks) {
        drawlandmark(pixs, landmark, DemoConfig.crop_H, DemoConfig.crop_W);
      }
    } else {
      croppedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
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
                runOnUiThread(
                        new Runnable() {
                          @Override
                          public void run() {
                            ImageView imageView = (ImageView) findViewById(R.id.imageView2);
                            imageView.setImageBitmap(detection);
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
}
