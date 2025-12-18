package org.tensorflow.lite.examples.gaze_estimation;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.example.gazedemo1.ProcessFactory.bitmap2mat;
import static com.example.gazedemo1.ProcessFactory.gaze_postprocess;
import static com.example.gazedemo1.ProcessFactory.gaze_preprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_postprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_preprocess;

import com.example.gazedemo1.ProcessFactory;

/**
 * Activity for processing a single user-selected image through gaze estimation pipeline.
 * - Select an image from device gallery
 * - Run face detection, landmark detection, and gaze estimation
 * - Display the results overlaid on the image
 */
public class SingleImageActivity extends AppCompatActivity {
    private static final String TAG = "SingleImage";
    
    // UI Elements
    private TextView statusText;
    private TextView resultsText;
    private ImageView previewImage;
    private ImageView processedImage;
    private Button selectImageButton;
    private ProgressBar progressBar;
    private LinearLayout resultsPanel;
    
    // Models
    private TFLiteFaceDetector tfliteFaceDetector = null;
    private QNNModelRunner landmark_detection_qnn = null;
    private QNNModelRunner gaze_estimation_qnn = null;
    
    // Threading
    private ExecutorService executorService;
    private Handler mainHandler;
    
    // Image picker request code
    private static final int REQUEST_IMAGE_PICKER = 2001;
    
    // OpenCV initialization state
    private volatile boolean isOpenCVReady = false;
    private volatile boolean modelsReady = false;
    
    // Current selected image
    private Uri selectedImageUri = null;
    private Bitmap originalBitmap = null;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Initialize OpenCV
        initLoadOpenCV();
        
        mainHandler = new Handler(Looper.getMainLooper());
        executorService = Executors.newSingleThreadExecutor();
        
        createUI();
        initializeModels();
    }
    
    private void initLoadOpenCV() {
        // Try OpenCVLoader.initDebug() first
        boolean success = OpenCVLoader.initDebug();
        if (success) {
            Log.d(TAG, "OpenCV load success via initDebug()");
            // Verify it actually works by creating a test Mat
            if (testOpenCVNative()) {
                isOpenCVReady = true;
                return;
            }
        }
        
        // Fallback: try loading the native library directly
        Log.w(TAG, "OpenCVLoader.initDebug() failed - trying System.loadLibrary()");
        try {
            System.loadLibrary("opencv_java4");
            Log.d(TAG, "OpenCV load success via System.loadLibrary()");
            if (testOpenCVNative()) {
                isOpenCVReady = true;
            } else {
                Log.e(TAG, "OpenCV native test failed even after library load");
                isOpenCVReady = false;
            }
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load opencv_java4: " + e.getMessage());
            isOpenCVReady = false;
        }
    }
    
    /**
     * Test if OpenCV native library is actually working by creating a Mat
     */
    private boolean testOpenCVNative() {
        try {
            Mat testMat = new Mat();
            testMat.release();
            Log.d(TAG, "OpenCV native test passed");
            return true;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "OpenCV native test failed: " + e.getMessage());
            return false;
        } catch (Exception e) {
            Log.e(TAG, "OpenCV native test exception: " + e.getMessage());
            return false;
        }
    }
    
    private void createUI() {
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 48, 32, 48);
        
        // Create gradient background
        GradientDrawable gradient = new GradientDrawable(
            GradientDrawable.Orientation.TOP_BOTTOM,
            new int[]{0xFF1A1A2E, 0xFF16213E, 0xFF0F3460}
        );
        layout.setBackground(gradient);
        
        // Title
        TextView title = new TextView(this);
        title.setText("🖼 Single Image Analysis");
        title.setTextSize(26);
        title.setTextColor(0xFFE0E1DD);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setGravity(Gravity.CENTER);
        title.setPadding(0, 0, 0, 8);
        layout.addView(title);
        
        // Subtitle
        TextView subtitle = new TextView(this);
        subtitle.setText("Select an image to analyze gaze direction");
        subtitle.setTextSize(14);
        subtitle.setTextColor(0xFF778DA9);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setPadding(0, 0, 0, 24);
        layout.addView(subtitle);
        
        // Status text
        statusText = new TextView(this);
        statusText.setText("Loading models...");
        statusText.setTextSize(14);
        statusText.setTextColor(0xFFFFD700);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(0, 0, 0, 16);
        layout.addView(statusText);
        
        // Progress bar (for loading)
        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleLarge);
        progressBar.setVisibility(View.VISIBLE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        progressParams.gravity = Gravity.CENTER;
        progressParams.setMargins(0, 0, 0, 24);
        progressBar.setLayoutParams(progressParams);
        layout.addView(progressBar);
        
        // Select image button
        selectImageButton = new Button(this);
        selectImageButton.setText("📁 Select Image from Gallery");
        selectImageButton.setTextSize(16);
        selectImageButton.setTextColor(0xFFFFFFFF);
        selectImageButton.setPadding(48, 32, 48, 32);
        selectImageButton.setAllCaps(false);
        selectImageButton.setEnabled(false);
        
        GradientDrawable buttonGradient = new GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            new int[]{0xFF7B2CBF, 0xFF5A189A}
        );
        buttonGradient.setCornerRadius(20);
        selectImageButton.setBackground(buttonGradient);
        selectImageButton.setOnClickListener(v -> selectImage());
        
        LinearLayout.LayoutParams selectParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        selectParams.setMargins(0, 0, 0, 24);
        selectImageButton.setLayoutParams(selectParams);
        layout.addView(selectImageButton);
        
        // Preview image container
        LinearLayout previewContainer = new LinearLayout(this);
        previewContainer.setOrientation(LinearLayout.VERTICAL);
        previewContainer.setGravity(Gravity.CENTER);
        previewContainer.setPadding(16, 16, 16, 16);
        
        GradientDrawable previewBg = new GradientDrawable();
        previewBg.setColor(0xFF252545);
        previewBg.setCornerRadius(16);
        previewContainer.setBackground(previewBg);
        
        // Original image label
        TextView originalLabel = new TextView(this);
        originalLabel.setText("Original Image");
        originalLabel.setTextSize(12);
        originalLabel.setTextColor(0xFF778DA9);
        originalLabel.setGravity(Gravity.CENTER);
        originalLabel.setPadding(0, 0, 0, 8);
        previewContainer.addView(originalLabel);
        
        // Preview image
        previewImage = new ImageView(this);
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        previewImage.setBackgroundColor(0xFF333355);
        previewImage.setAdjustViewBounds(true);
        LinearLayout.LayoutParams imageParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            350
        );
        previewImage.setLayoutParams(imageParams);
        previewContainer.addView(previewImage);
        
        layout.addView(previewContainer);
        
        // Spacer
        View spacer = new View(this);
        spacer.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 20));
        layout.addView(spacer);
        
        // Processed image container
        LinearLayout processedContainer = new LinearLayout(this);
        processedContainer.setOrientation(LinearLayout.VERTICAL);
        processedContainer.setGravity(Gravity.CENTER);
        processedContainer.setPadding(16, 16, 16, 16);
        processedContainer.setVisibility(View.GONE);
        
        GradientDrawable processedBg = new GradientDrawable();
        processedBg.setColor(0xFF1E3A5F);
        processedBg.setCornerRadius(16);
        processedContainer.setBackground(processedBg);
        
        // Processed image label
        TextView processedLabel = new TextView(this);
        processedLabel.setText("Analysis Result");
        processedLabel.setTextSize(12);
        processedLabel.setTextColor(0xFF00B4D8);
        processedLabel.setGravity(Gravity.CENTER);
        processedLabel.setPadding(0, 0, 0, 8);
        processedContainer.addView(processedLabel);
        
        // Processed image
        processedImage = new ImageView(this);
        processedImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        processedImage.setBackgroundColor(0xFF1E3A5F);
        processedImage.setAdjustViewBounds(true);
        LinearLayout.LayoutParams processedImageParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            350
        );
        processedImage.setLayoutParams(processedImageParams);
        processedContainer.addView(processedImage);
        
        layout.addView(processedContainer);
        
        // Results panel
        resultsPanel = new LinearLayout(this);
        resultsPanel.setOrientation(LinearLayout.VERTICAL);
        resultsPanel.setGravity(Gravity.CENTER);
        resultsPanel.setPadding(24, 24, 24, 24);
        resultsPanel.setVisibility(View.GONE);
        
        GradientDrawable resultsBg = new GradientDrawable();
        resultsBg.setColor(0xFF1A3A3A);
        resultsBg.setCornerRadius(16);
        resultsBg.setStroke(2, 0xFF00B4D8);
        resultsPanel.setBackground(resultsBg);
        
        LinearLayout.LayoutParams resultsParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        resultsParams.setMargins(0, 20, 0, 0);
        resultsPanel.setLayoutParams(resultsParams);
        
        // Results title
        TextView resultsTitle = new TextView(this);
        resultsTitle.setText("📊 Detection Results");
        resultsTitle.setTextSize(18);
        resultsTitle.setTextColor(0xFF00B4D8);
        resultsTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        resultsTitle.setGravity(Gravity.CENTER);
        resultsTitle.setPadding(0, 0, 0, 12);
        resultsPanel.addView(resultsTitle);
        
        // Results text
        resultsText = new TextView(this);
        resultsText.setTextSize(14);
        resultsText.setTextColor(0xFFE0E1DD);
        resultsText.setGravity(Gravity.CENTER);
        resultsPanel.addView(resultsText);
        
        layout.addView(resultsPanel);
        
        // Back button
        Button backButton = new Button(this);
        backButton.setText("← Back to Menu");
        backButton.setTextSize(14);
        backButton.setTextColor(0xFFFFFFFF);
        backButton.setPadding(32, 24, 32, 24);
        backButton.setAllCaps(false);
        
        GradientDrawable backGradient = new GradientDrawable();
        backGradient.setColor(0xFF415A77);
        backGradient.setCornerRadius(16);
        backButton.setBackground(backGradient);
        
        LinearLayout.LayoutParams backParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        backParams.setMargins(0, 32, 0, 0);
        backButton.setLayoutParams(backParams);
        backButton.setOnClickListener(v -> {
            Intent intent = new Intent(this, LauncherActivity.class);
            startActivity(intent);
            finish();
        });
        layout.addView(backButton);
        
        scrollView.addView(layout);
        setContentView(scrollView);
    }
    
    private void initializeModels() {
        statusText.setText("Loading AI models...");
        
        executorService.execute(() -> {
            try {
                // Initialize TFLite Face Detector
                if (DemoConfig.USE_TFLITE_FACE_DETECTION) {
                    tfliteFaceDetector = new TFLiteFaceDetector(this);
                    tfliteFaceDetector.initialize();
                    Log.i(TAG, "TFLite Face Detector initialized");
                }
                
                // Initialize QNN Landmark Detection
                landmark_detection_qnn = new QNNModelRunner();
                boolean landmarkOk = landmark_detection_qnn.loadModel(
                    DemoConfig.landmark_detection_model_path,
                    DemoConfig.landmark_detection_model_out,
                    QNNModelRunner.Backend.GPU
                );
                if (!landmarkOk) {
                    throw new Exception("Failed to initialize landmark detection model");
                }
                Log.i(TAG, "Landmark detection model initialized");
                
                // Initialize QNN Gaze Estimation
                gaze_estimation_qnn = new QNNModelRunner();
                boolean gazeOk = gaze_estimation_qnn.loadModel(
                    DemoConfig.gaze_estimation_model_path,
                    DemoConfig.gaze_estimation_model_out,
                    QNNModelRunner.Backend.GPU
                );
                if (!gazeOk) {
                    throw new Exception("Failed to initialize gaze estimation model");
                }
                Log.i(TAG, "Gaze estimation model initialized");
                
                modelsReady = true;
                
                mainHandler.post(() -> {
                    statusText.setText("✓ Ready! Select an image to analyze");
                    statusText.setTextColor(0xFF00FF7F);
                    progressBar.setVisibility(View.GONE);
                    selectImageButton.setEnabled(true);
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize models: " + e.getMessage(), e);
                mainHandler.post(() -> {
                    statusText.setText("❌ Failed to load models: " + e.getMessage());
                    statusText.setTextColor(0xFFFF6B6B);
                    progressBar.setVisibility(View.GONE);
                    Toast.makeText(this, "Model loading failed", Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_IMAGE_PICKER);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (requestCode == REQUEST_IMAGE_PICKER && resultCode == Activity.RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                selectedImageUri = uri;
                loadAndProcessImage(uri);
            }
        }
    }
    
    private void loadAndProcessImage(Uri imageUri) {
        // Pre-check: verify OpenCV is actually working
        if (!isOpenCVReady) {
            // Try reinitializing OpenCV
            initLoadOpenCV();
            if (!isOpenCVReady) {
                statusText.setText("❌ OpenCV not available. Please restart the app.");
                statusText.setTextColor(0xFFFF6B6B);
                Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show();
                return;
            }
        }
        
        // Pre-check: verify models are loaded
        if (!modelsReady) {
            statusText.setText("⏳ Models still loading. Please wait...");
            statusText.setTextColor(0xFFFFD700);
            Toast.makeText(this, "Please wait for models to load", Toast.LENGTH_SHORT).show();
            return;
        }
        
        statusText.setText("Processing image...");
        statusText.setTextColor(0xFFFFD700);
        progressBar.setVisibility(View.VISIBLE);
        selectImageButton.setEnabled(false);
        
        // Hide results panel until processing is complete
        resultsPanel.setVisibility(View.GONE);
        
        executorService.execute(() -> {
            try {
                // Load image
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                if (inputStream == null) {
                    throw new Exception("Failed to open image");
                }
                
                Bitmap loadedBitmap = BitmapFactory.decodeStream(inputStream);
                inputStream.close();
                
                if (loadedBitmap == null) {
                    throw new Exception("Failed to decode image");
                }
                
                originalBitmap = loadedBitmap.copy(Bitmap.Config.ARGB_8888, true);
                
                // Show original preview
                mainHandler.post(() -> {
                    previewImage.setImageBitmap(originalBitmap);
                });
                
                // Process image
                ProcessingResult result = processImage(originalBitmap);
                
                mainHandler.post(() -> {
                    displayResults(result);
                    statusText.setText("✓ Analysis complete!");
                    statusText.setTextColor(0xFF00FF7F);
                    progressBar.setVisibility(View.GONE);
                    selectImageButton.setEnabled(true);
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Error processing image: " + e.getMessage(), e);
                mainHandler.post(() -> {
                    statusText.setText("❌ Error: " + e.getMessage());
                    statusText.setTextColor(0xFFFF6B6B);
                    progressBar.setVisibility(View.GONE);
                    selectImageButton.setEnabled(true);
                    Toast.makeText(this, "Processing failed", Toast.LENGTH_SHORT).show();
                });
            }
        });
    }
    
    private static class ProcessingResult {
        boolean faceDetected = false;
        float[] boundingBox = null;
        float[] landmarks = null;
        float[] gaze = null;  // pitch, yaw
        Bitmap processedBitmap = null;
        String errorMessage = null;
    }
    
    private ProcessingResult processImage(Bitmap bitmap) {
        ProcessingResult result = new ProcessingResult();
        // Always set a default processedBitmap to avoid null issues
        result.processedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        
        if (!isOpenCVReady) {
            result.errorMessage = "OpenCV not initialized. Please restart the app.";
            Log.e(TAG, "OpenCV not ready when trying to process image");
            return result;
        }
        
        if (!modelsReady) {
            result.errorMessage = "Models not loaded. Please wait and try again.";
            return result;
        }
        
        Mat img = null;
        try {
            // Convert to Mat - wrap in try-catch for OpenCV native errors
            img = bitmap2mat(bitmap);
            if (img == null || img.empty()) {
                result.errorMessage = "Failed to convert image";
                return result;
            }
            int imgWidth = img.width();
            int imgHeight = img.height();
            
            // === FACE DETECTION ===
            float[][] boxes = null;
            
            if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
                // Resize image for face detection
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 
                    DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_H, true);
                
                boxes = tfliteFaceDetector.detectFaces(resizedBitmap);
                
                // Scale boxes back to original image size
                if (boxes != null) {
                    float scaleX = (float) imgWidth / DemoConfig.face_detection_input_W;
                    float scaleY = (float) imgHeight / DemoConfig.face_detection_input_H;
                    for (float[] box : boxes) {
                        box[0] *= scaleX;
                        box[1] *= scaleY;
                        box[2] *= scaleX;
                        box[3] *= scaleY;
                    }
                }
            }
            
            if (boxes == null || boxes.length == 0) {
                Log.d(TAG, "No face detected in image");
                result.faceDetected = false;
                result.processedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                
                // Draw "No face detected" on image
                Canvas canvas = new Canvas(result.processedBitmap);
                Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setTextSize(48);
                paint.setTextAlign(Paint.Align.CENTER);
                canvas.drawText("No Face Detected", imgWidth / 2f, imgHeight / 2f, paint);
                
                return result;
            }
            
            result.faceDetected = true;
            result.boundingBox = boxes[0];
            float[] box = boxes[0];
            
            Log.d(TAG, "Face detected at: " + box[0] + ", " + box[1] + ", " + box[2] + ", " + box[3]);
            
            // === LANDMARK DETECTION ===
            ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
            float[] landmark_detection_input = landmark_preprocess_result.input;
            
            int[] landmarkInputDims = {1, DemoConfig.landmark_detection_input_H, 
                DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
            float[] landmarkOutput = landmark_detection_qnn.runInference(
                "face_image", landmark_detection_input, landmarkInputDims);
            
            if (landmarkOutput == null) {
                result.errorMessage = "Landmark detection failed";
                result.processedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                return result;
            }
            
            float[] landmark = landmark_postprocess(landmark_preprocess_result, landmarkOutput);
            result.landmarks = landmark;
            
            Log.d(TAG, "Landmarks detected: " + landmark.length / 2 + " points");
            
            // === GAZE ESTIMATION ===
            ProcessFactory.GazePreprocessResult gaze_preprocess_result = gaze_preprocess(img, landmark);
            
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
            
            float[] gazeOutput = gaze_estimation_qnn.runMultiInputInference(
                inputNames, inputDataArrays, inputDimsArrays);
            
            if (gazeOutput != null) {
                result.gaze = gaze_postprocess(gazeOutput, gaze_preprocess_result.R);
                Log.d(TAG, "Gaze estimated: pitch=" + Math.toDegrees(result.gaze[0]) + 
                    "°, yaw=" + Math.toDegrees(result.gaze[1]) + "°");
            }
            
            // === DRAW RESULTS ON IMAGE ===
            Bitmap processedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            
            // Draw on the image using OpenCV utilities
            if (result.gaze != null && result.landmarks != null) {
                DrawUtils.drawgaze(img, result.gaze, result.landmarks);
            }
            
            // Draw bounding box
            int[] pixels = new int[imgWidth * imgHeight];
            Utils.matToBitmap(img, processedBitmap);
            processedBitmap.getPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);
            
            DrawUtils.drawbox(pixels, boxes, imgHeight, imgWidth);
            
            if (result.landmarks != null) {
                DrawUtils.drawlandmark(pixels, result.landmarks, imgHeight, imgWidth);
            }
            
            processedBitmap.setPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);
            result.processedBitmap = processedBitmap;
            
        } catch (UnsatisfiedLinkError e) {
            // OpenCV native library error
            Log.e(TAG, "OpenCV native error in processImage: " + e.getMessage(), e);
            result.errorMessage = "OpenCV native library error. Please restart the app.";
            isOpenCVReady = false;  // Mark OpenCV as not ready
        } catch (Exception e) {
            Log.e(TAG, "Error in processImage: " + e.getMessage(), e);
            result.errorMessage = e.getMessage();
        } finally {
            // Ensure processedBitmap is always set
            if (result.processedBitmap == null) {
                result.processedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            }
        }
        
        return result;
    }
    
    private void displayResults(ProcessingResult result) {
        // Show processed image
        if (result.processedBitmap != null) {
            processedImage.setImageBitmap(result.processedBitmap);
            ((View) processedImage.getParent()).setVisibility(View.VISIBLE);
        }
        
        // Build results text
        StringBuilder sb = new StringBuilder();
        
        if (result.errorMessage != null) {
            sb.append("❌ Error: ").append(result.errorMessage);
            resultsText.setTextColor(0xFFFF6B6B);
        } else if (!result.faceDetected) {
            sb.append("No face detected in the image.\n");
            sb.append("Please select an image with a visible face.");
            resultsText.setTextColor(0xFFFFD93D);
        } else {
            resultsText.setTextColor(0xFFE0E1DD);
            
            sb.append("✓ Face Detected\n\n");
            
            if (result.boundingBox != null) {
                sb.append("📦 Bounding Box:\n");
                sb.append(String.format("   Position: (%.0f, %.0f) to (%.0f, %.0f)\n\n",
                    result.boundingBox[0], result.boundingBox[1],
                    result.boundingBox[2], result.boundingBox[3]));
            }
            
            if (result.landmarks != null) {
                sb.append("🔵 Landmarks: ").append(result.landmarks.length / 2).append(" points detected\n\n");
            }
            
            if (result.gaze != null) {
                float pitchDegrees = (float) Math.toDegrees(result.gaze[0]);
                float yawDegrees = (float) Math.toDegrees(result.gaze[1]);
                
                sb.append("👁 Gaze Direction:\n");
                sb.append(String.format("   Pitch: %.1f°", pitchDegrees));
                if (pitchDegrees > 5) {
                    sb.append(" (looking up)");
                } else if (pitchDegrees < -5) {
                    sb.append(" (looking down)");
                } else {
                    sb.append(" (level)");
                }
                sb.append("\n");
                
                sb.append(String.format("   Yaw: %.1f°", yawDegrees));
                if (yawDegrees > 10) {
                    sb.append(" (looking right)");
                } else if (yawDegrees < -10) {
                    sb.append(" (looking left)");
                } else {
                    sb.append(" (forward)");
                }
            }
        }
        
        resultsText.setText(sb.toString());
        resultsPanel.setVisibility(View.VISIBLE);
    }
    
    @Override
    protected void onDestroy() {
        if (tfliteFaceDetector != null) {
            tfliteFaceDetector.close();
            tfliteFaceDetector = null;
        }
        
        if (landmark_detection_qnn != null) {
            landmark_detection_qnn.close();
            landmark_detection_qnn = null;
        }
        
        if (gaze_estimation_qnn != null) {
            gaze_estimation_qnn.close();
            gaze_estimation_qnn = null;
        }
        
        if (executorService != null) {
            executorService.shutdown();
        }
        
        super.onDestroy();
    }
}

