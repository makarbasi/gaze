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
 * Uses the SAME business logic as ImageProcessingActivity (batch mode) and ClassifierActivity (camera mode).
 */
public class SingleImageActivity extends AppCompatActivity {
    private static final String TAG = "SingleImage";
    
    // UI Elements
    private TextView statusText;
    private TextView resultsText;
    private ImageView previewImage;
    private ImageView processedImageView;
    private Button selectImageButton;
    private ProgressBar progressBar;
    private LinearLayout resultsPanel;
    private LinearLayout processedContainer;
    
    // Models - same as ImageProcessingActivity
    private TFLiteFaceDetector tfliteFaceDetector = null;
    private QNNModelRunner landmark_detection_qnn = null;
    private QNNModelRunner gaze_estimation_qnn = null;
    
    // Threading
    private ExecutorService executorService;
    private Handler mainHandler;
    
    // Image picker request code
    private static final int REQUEST_IMAGE_PICKER = 2001;
    
    // State flags
    private volatile boolean isOpenCVReady = false;
    private volatile boolean modelsReady = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Initialize OpenCV - SAME pattern as ImageProcessingActivity
        initLoadOpenCV();
        
        mainHandler = new Handler(Looper.getMainLooper());
        executorService = Executors.newSingleThreadExecutor();
        
        createUI();
        initializeModels();
    }
    
    /**
     * Initialize OpenCV - same pattern as ImageProcessingActivity
     */
    private void initLoadOpenCV() {
        boolean success = OpenCVLoader.initDebug();
        if (success) {
            Log.d(TAG, "OpenCV load success via initDebug()");
            isOpenCVReady = true;
            return;
        }
        
        Log.w(TAG, "OpenCVLoader.initDebug() failed - trying System.loadLibrary()");
        try {
            System.loadLibrary("opencv_java4");
            Log.d(TAG, "OpenCV load success via System.loadLibrary()");
            isOpenCVReady = true;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load opencv_java4: " + e.getMessage());
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show();
            isOpenCVReady = false;
        }
    }
    
    private void createUI() {
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 48, 32, 48);
        
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
        
        // Status text
        statusText = new TextView(this);
        statusText.setText("Loading models...");
        statusText.setTextSize(14);
        statusText.setTextColor(0xFFFFD700);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(0, 8, 0, 16);
        layout.addView(statusText);
        
        // Progress bar
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
        selectImageButton.setText("📁 Select Image");
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
        
        // Preview image
        previewImage = new ImageView(this);
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        previewImage.setBackgroundColor(0xFF333355);
        previewImage.setAdjustViewBounds(true);
        LinearLayout.LayoutParams imageParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        );
        imageParams.setMargins(0, 0, 0, 16);
        previewImage.setLayoutParams(imageParams);
        layout.addView(previewImage);
        
        // Processed image container (hidden initially)
        processedContainer = new LinearLayout(this);
        processedContainer.setOrientation(LinearLayout.VERTICAL);
        processedContainer.setVisibility(View.GONE);
        
        TextView processedLabel = new TextView(this);
        processedLabel.setText("Analysis Result:");
        processedLabel.setTextSize(14);
        processedLabel.setTextColor(0xFF00B4D8);
        processedLabel.setPadding(0, 0, 0, 8);
        processedContainer.addView(processedLabel);
        
        processedImageView = new ImageView(this);
        processedImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        processedImageView.setBackgroundColor(0xFF1E3A5F);
        processedImageView.setAdjustViewBounds(true);
        LinearLayout.LayoutParams processedParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        );
        processedImageView.setLayoutParams(processedParams);
        processedContainer.addView(processedImageView);
        
        layout.addView(processedContainer);
        
        // Results panel
        resultsPanel = new LinearLayout(this);
        resultsPanel.setOrientation(LinearLayout.VERTICAL);
        resultsPanel.setPadding(24, 24, 24, 24);
        resultsPanel.setVisibility(View.GONE);
        
        GradientDrawable resultsBg = new GradientDrawable();
        resultsBg.setColor(0xFF1A3A3A);
        resultsBg.setCornerRadius(16);
        resultsPanel.setBackground(resultsBg);
        
        LinearLayout.LayoutParams resultsParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        resultsParams.setMargins(0, 20, 0, 0);
        resultsPanel.setLayoutParams(resultsParams);
        
        resultsText = new TextView(this);
        resultsText.setTextSize(14);
        resultsText.setTextColor(0xFFE0E1DD);
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
    
    /**
     * Initialize models - SAME as ImageProcessingActivity
     */
    private void initializeModels() {
        statusText.setText("Loading AI models...");
        
        executorService.execute(() -> {
            try {
                // Initialize TFLite Face Detector - same as ImageProcessingActivity
                if (DemoConfig.USE_TFLITE_FACE_DETECTION) {
                    tfliteFaceDetector = new TFLiteFaceDetector(this);
                    tfliteFaceDetector.initialize();
                    Log.i(TAG, "TFLite Face Detector initialized");
                }
                
                // Initialize QNN Landmark Detection - same as ImageProcessingActivity
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
                
                // Initialize QNN Gaze Estimation - same as ImageProcessingActivity
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
                processSelectedImage(uri);
            }
        }
    }
    
    /**
     * Process the selected image - uses SAME logic as ImageProcessingActivity.processImage()
     */
    private void processSelectedImage(Uri imageUri) {
        statusText.setText("Processing image...");
        statusText.setTextColor(0xFFFFD700);
        progressBar.setVisibility(View.VISIBLE);
        selectImageButton.setEnabled(false);
        resultsPanel.setVisibility(View.GONE);
        processedContainer.setVisibility(View.GONE);
        
        executorService.execute(() -> {
            try {
                // ========== EXACT SAME LOGIC AS ImageProcessingActivity.processImage() ==========
                
                // Load image
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                if (inputStream == null) {
                    Log.e(TAG, "Failed to open image: " + imageUri);
                    showError("Failed to open image");
                    return;
                }
                
                Bitmap originalBitmap = BitmapFactory.decodeStream(inputStream);
                inputStream.close();
                
                if (originalBitmap == null) {
                    Log.e(TAG, "Failed to decode image: " + imageUri);
                    showError("Failed to decode image");
                    return;
                }
                
                // Show preview
                final Bitmap previewBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, false);
                mainHandler.post(() -> previewImage.setImageBitmap(previewBitmap));
                
                // Convert to Mat - SAME as ImageProcessingActivity
                Mat img = bitmap2mat(originalBitmap);
                int imgWidth = img.width();
                int imgHeight = img.height();
                
                // === FACE DETECTION === (same as ImageProcessingActivity)
                float[][] boxes = null;
                
                if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 
                        DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_H, true);
                    
                    boxes = tfliteFaceDetector.detectFaces(resizedBitmap);
                    
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
                    Log.d(TAG, "No face detected");
                    showResult(originalBitmap, null, null, null);
                    return;
                }
                
                float[] box = boxes[0];
                Log.d(TAG, "Face detected at: " + box[0] + ", " + box[1] + ", " + box[2] + ", " + box[3]);
                
                // === LANDMARK DETECTION === (same as ImageProcessingActivity)
                ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
                float[] landmark_detection_input = landmark_preprocess_result.input;
                
                int[] landmarkInputDims = {1, DemoConfig.landmark_detection_input_H, 
                    DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
                float[] landmarkOutput = landmark_detection_qnn.runInference(
                    "face_image", landmark_detection_input, landmarkInputDims);
                
                if (landmarkOutput == null) {
                    Log.e(TAG, "Landmark detection failed");
                    showResult(originalBitmap, box, null, null);
                    return;
                }
                
                float[] landmark = landmark_postprocess(landmark_preprocess_result, landmarkOutput);
                
                // === GAZE ESTIMATION === (same as ImageProcessingActivity)
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
                
                float[] gaze = null;
                if (gazeOutput != null) {
                    gaze = gaze_postprocess(gazeOutput, gaze_preprocess_result.R);
                    Log.d(TAG, "Gaze: pitch=" + Math.toDegrees(gaze[0]) + "°, yaw=" + Math.toDegrees(gaze[1]) + "°");
                }
                
                // Draw results on image
                Bitmap resultBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
                
                if (gaze != null && landmark != null) {
                    DrawUtils.drawgaze(img, gaze, landmark);
                }
                
                Utils.matToBitmap(img, resultBitmap);
                int[] pixels = new int[imgWidth * imgHeight];
                resultBitmap.getPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);
                
                DrawUtils.drawbox(pixels, boxes, imgHeight, imgWidth);
                DrawUtils.drawlandmark(pixels, landmark, imgHeight, imgWidth);
                
                resultBitmap.setPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);
                
                showResult(resultBitmap, box, landmark, gaze);
                
            } catch (Exception e) {
                Log.e(TAG, "Error processing image: " + e.getMessage(), e);
                showError("Error: " + e.getMessage());
            }
        });
    }
    
    private void showError(String message) {
        mainHandler.post(() -> {
            statusText.setText("❌ " + message);
            statusText.setTextColor(0xFFFF6B6B);
            progressBar.setVisibility(View.GONE);
            selectImageButton.setEnabled(true);
        });
    }
    
    private void showResult(Bitmap processedBitmap, float[] box, float[] landmarks, float[] gaze) {
        mainHandler.post(() -> {
            progressBar.setVisibility(View.GONE);
            selectImageButton.setEnabled(true);
            
            // Show processed image
            if (processedBitmap != null) {
                processedImageView.setImageBitmap(processedBitmap);
                processedContainer.setVisibility(View.VISIBLE);
            }
            
            // Build results text
            StringBuilder sb = new StringBuilder();
            
            if (box == null) {
                sb.append("⚠️ No face detected in the image.\n");
                sb.append("Please select an image with a visible face.");
                statusText.setText("No face detected");
                statusText.setTextColor(0xFFFFD93D);
            } else {
                statusText.setText("✓ Analysis complete!");
                statusText.setTextColor(0xFF00FF7F);
                
                sb.append("✓ Face Detected\n\n");
                sb.append(String.format("📦 Box: (%.0f, %.0f) to (%.0f, %.0f)\n", box[0], box[1], box[2], box[3]));
                
                if (landmarks != null) {
                    sb.append("🔵 Landmarks: ").append(landmarks.length / 2).append(" points\n");
                }
                
                if (gaze != null) {
                    float pitch = (float) Math.toDegrees(gaze[0]);
                    float yaw = (float) Math.toDegrees(gaze[1]);
                    sb.append(String.format("\n👁 Gaze:\n   Pitch: %.1f°\n   Yaw: %.1f°", pitch, yaw));
                }
            }
            
            resultsText.setText(sb.toString());
            resultsPanel.setVisibility(View.VISIBLE);
        });
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
