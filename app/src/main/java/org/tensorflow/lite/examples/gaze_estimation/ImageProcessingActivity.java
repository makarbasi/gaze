package org.tensorflow.lite.examples.gaze_estimation;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.documentfile.provider.DocumentFile;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.example.gazedemo1.ProcessFactory.bitmap2mat;
import static com.example.gazedemo1.ProcessFactory.gaze_postprocess;
import static com.example.gazedemo1.ProcessFactory.gaze_preprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_postprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_preprocess;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.postprocessing;

import com.example.gazedemo1.ProcessFactory;

/**
 * Activity for batch processing images through gaze estimation pipeline.
 * - Select a folder with images
 * - Process all images (including subfolders)
 * - Run face detection, landmark detection, and gaze estimation
 * - Save results to CSV
 */
public class ImageProcessingActivity extends AppCompatActivity {
    private static final String TAG = "ImageProcessing";
    
    // UI Elements
    private TextView statusText;
    private TextView progressText;
    private ProgressBar progressBar;
    private ImageView previewImage;
    private Button selectFolderButton;
    private Button startProcessingButton;
    private CheckBox isLookingCheckbox;
    
    // Processing state
    private Uri selectedFolderUri = null;
    private List<Uri> imageUris = new ArrayList<>();
    private boolean isProcessing = false;
    private int processedCount = 0;
    private int totalCount = 0;
    
    // Models
    private TFLiteFaceDetector tfliteFaceDetector = null;
    private QNNModelRunner landmark_detection_qnn = null;
    private QNNModelRunner gaze_estimation_qnn = null;
    
    // Recording
    private BufferedWriter recordingWriter = null;
    private String currentRecordingFile = null;
    
    // Threading
    private ExecutorService executorService;
    private Handler mainHandler;
    
    // Folder picker request code
    private static final int REQUEST_FOLDER_PICKER = 1001;
    
    // OpenCV initialization state
    private volatile boolean isOpenCVReady = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Initialize OpenCV (same pattern as CameraActivity - don't crash on failure)
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
            isOpenCVReady = true;
            return;
        }
        
        // Fallback: try loading the native library directly
        Log.w(TAG, "OpenCVLoader.initDebug() failed - trying System.loadLibrary()");
        try {
            System.loadLibrary("opencv_java4");
            Log.d(TAG, "OpenCV load success via System.loadLibrary()");
            isOpenCVReady = true;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load opencv_java4: " + e.getMessage());
            Toast.makeText(this, "OpenCV failed to load: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (requestCode == REQUEST_FOLDER_PICKER && resultCode == Activity.RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                // Take persistent permission
                getContentResolver().takePersistableUriPermission(uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION);
                selectedFolderUri = uri;
                scanForImages(uri);
            }
        }
    }
    
    private void createUI() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 32, 32, 32);
        layout.setBackgroundColor(0xFF1A1A2E);
        
        // Title
        TextView title = new TextView(this);
        title.setText("Batch Image Processing");
        title.setTextSize(24);
        title.setTextColor(0xFFFFFFFF);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setPadding(0, 0, 0, 24);
        layout.addView(title);
        
        // Status text
        statusText = new TextView(this);
        statusText.setText("Select a folder with images to process");
        statusText.setTextSize(14);
        statusText.setTextColor(0xFFCCCCCC);
        statusText.setPadding(0, 0, 0, 16);
        layout.addView(statusText);
        
        // Select folder button
        selectFolderButton = new Button(this);
        selectFolderButton.setText("📁 Select Folder");
        selectFolderButton.setTextSize(16);
        selectFolderButton.setTextColor(0xFFFFFFFF);
        selectFolderButton.setBackgroundColor(0xFF2196F3);
        selectFolderButton.setOnClickListener(v -> selectFolder());
        layout.addView(selectFolderButton);
        
        // Spacer
        View spacer1 = new View(this);
        spacer1.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 24));
        layout.addView(spacer1);
        
        // isLooking checkbox
        isLookingCheckbox = new CheckBox(this);
        isLookingCheckbox.setText("isLooking? (for recording label)");
        isLookingCheckbox.setTextColor(0xFFFFFFFF);
        isLookingCheckbox.setChecked(true);
        layout.addView(isLookingCheckbox);
        
        // Start processing button
        startProcessingButton = new Button(this);
        startProcessingButton.setText("▶️ Start Processing");
        startProcessingButton.setTextSize(16);
        startProcessingButton.setTextColor(0xFFFFFFFF);
        startProcessingButton.setBackgroundColor(0xFF4CAF50);
        startProcessingButton.setEnabled(false);
        startProcessingButton.setOnClickListener(v -> startProcessing());
        LinearLayout.LayoutParams startParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        startParams.setMargins(0, 16, 0, 0);
        startProcessingButton.setLayoutParams(startParams);
        layout.addView(startProcessingButton);
        
        // Progress bar
        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgress(0);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        progressParams.setMargins(0, 24, 0, 0);
        progressBar.setLayoutParams(progressParams);
        layout.addView(progressBar);
        
        // Progress text
        progressText = new TextView(this);
        progressText.setText("");
        progressText.setTextSize(14);
        progressText.setTextColor(0xFFCCCCCC);
        progressText.setPadding(0, 8, 0, 16);
        layout.addView(progressText);
        
        // Preview image
        previewImage = new ImageView(this);
        previewImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        previewImage.setBackgroundColor(0xFF333333);
        LinearLayout.LayoutParams imageParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            400
        );
        imageParams.setMargins(0, 16, 0, 0);
        previewImage.setLayoutParams(imageParams);
        layout.addView(previewImage);
        
        // Back button
        Button backButton = new Button(this);
        backButton.setText("← Back to Menu");
        backButton.setTextSize(14);
        backButton.setTextColor(0xFFFFFFFF);
        backButton.setBackgroundColor(0xFF666666);
        LinearLayout.LayoutParams backParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        backParams.setMargins(0, 24, 0, 0);
        backButton.setLayoutParams(backParams);
        backButton.setOnClickListener(v -> {
            if (!isProcessing) {
                Intent intent = new Intent(this, LauncherActivity.class);
                startActivity(intent);
                finish();
            }
        });
        layout.addView(backButton);
        
        setContentView(layout);
    }
    
    private void initializeModels() {
        statusText.setText("Loading models...");
        
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
                
                mainHandler.post(() -> {
                    statusText.setText("Models loaded. Select a folder with images.");
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Failed to initialize models: " + e.getMessage(), e);
                mainHandler.post(() -> {
                    statusText.setText("Failed to load models: " + e.getMessage());
                    Toast.makeText(this, "Model loading failed", Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void selectFolder() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, REQUEST_FOLDER_PICKER);
    }
    
    private void scanForImages(Uri folderUri) {
        imageUris.clear();
        statusText.setText("Scanning for images...");
        
        executorService.execute(() -> {
            try {
                DocumentFile folder = DocumentFile.fromTreeUri(this, folderUri);
                if (folder != null) {
                    scanFolderRecursive(folder);
                }
                
                mainHandler.post(() -> {
                    totalCount = imageUris.size();
                    if (totalCount > 0) {
                        statusText.setText("Found " + totalCount + " images. Ready to process.");
                        startProcessingButton.setEnabled(true);
                    } else {
                        statusText.setText("No images found in selected folder.");
                        startProcessingButton.setEnabled(false);
                    }
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Error scanning folder: " + e.getMessage(), e);
                mainHandler.post(() -> {
                    statusText.setText("Error scanning folder: " + e.getMessage());
                });
            }
        });
    }
    
    private void scanFolderRecursive(DocumentFile folder) {
        if (folder == null || !folder.isDirectory()) return;
        
        DocumentFile[] files = folder.listFiles();
        if (files == null) return;
        
        for (DocumentFile file : files) {
            if (file.isDirectory()) {
                scanFolderRecursive(file);
            } else if (file.isFile()) {
                String name = file.getName();
                if (name != null) {
                    String lower = name.toLowerCase();
                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || 
                        lower.endsWith(".png") || lower.endsWith(".bmp")) {
                        imageUris.add(file.getUri());
                    }
                }
            }
        }
    }
    
    private void startProcessing() {
        if (isProcessing || imageUris.isEmpty()) return;
        
        // Check if OpenCV is ready
        if (!isOpenCVReady) {
            Toast.makeText(this, "Please wait - OpenCV is still loading...", Toast.LENGTH_SHORT).show();
            return;
        }
        
        isProcessing = true;
        processedCount = 0;
        selectFolderButton.setEnabled(false);
        startProcessingButton.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);
        
        // Start recording
        startRecording();
        
        executorService.execute(() -> {
            for (int i = 0; i < imageUris.size() && isProcessing; i++) {
                Uri imageUri = imageUris.get(i);
                processImage(imageUri, i);
                
                final int progress = (int) ((i + 1) * 100.0 / totalCount);
                final int current = i + 1;
                mainHandler.post(() -> {
                    progressBar.setProgress(progress);
                    progressText.setText("Processing: " + current + " / " + totalCount);
                });
            }
            
            // Stop recording
            stopRecording();
            
            mainHandler.post(() -> {
                isProcessing = false;
                selectFolderButton.setEnabled(true);
                startProcessingButton.setEnabled(true);
                statusText.setText("Processing complete! " + processedCount + " images processed.");
                progressText.setText("Results saved to: " + currentRecordingFile);
                Toast.makeText(this, "Processing complete!", Toast.LENGTH_SHORT).show();
            });
        });
    }
    
    private void processImage(Uri imageUri, int index) {
        try {
            // Load image
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            if (inputStream == null) {
                Log.e(TAG, "Failed to open image: " + imageUri);
                return;
            }
            
            Bitmap originalBitmap = BitmapFactory.decodeStream(inputStream);
            inputStream.close();
            
            if (originalBitmap == null) {
                Log.e(TAG, "Failed to decode image: " + imageUri);
                return;
            }
            
            // Convert to Mat (NO cropping - use full image as-is)
            Mat img = bitmap2mat(originalBitmap);
            int imgWidth = img.width();
            int imgHeight = img.height();
            
            // Update preview on main thread
            final Bitmap previewBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, false);
            mainHandler.post(() -> previewImage.setImageBitmap(previewBitmap));
            
            // === FACE DETECTION ===
            float[][] boxes = null;
            
            if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
                // Resize image for face detection
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 
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
                Log.d(TAG, "No face detected in: " + imageUri);
                // Record with null data
                recordFrame(imageUri.toString(), null, null, null);
                processedCount++;
                return;
            }
            
            // Process first detected face
            float[] box = boxes[0];
            
            // === LANDMARK DETECTION ===
            ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
            float[] landmark_detection_input = landmark_preprocess_result.input;
            
            int[] landmarkInputDims = {1, DemoConfig.landmark_detection_input_H, 
                DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
            float[] landmarkOutput = landmark_detection_qnn.runInference(
                "face_image", landmark_detection_input, landmarkInputDims);
            
            if (landmarkOutput == null) {
                Log.e(TAG, "Landmark detection failed for: " + imageUri);
                recordFrame(imageUri.toString(), null, null, null);
                processedCount++;
                return;
            }
            
            float[] landmark = landmark_postprocess(landmark_preprocess_result, landmarkOutput);
            
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
            
            float[] gaze_pitchyaw = null;
            if (gazeOutput != null) {
                gaze_pitchyaw = gaze_postprocess(gazeOutput, gaze_preprocess_result.R);
            }
            
            // Record results (now includes head pose from rvec)
            recordFrame(imageUri.toString(), landmark, gaze_pitchyaw, gaze_preprocess_result.rvec);
            processedCount++;
            
            Log.d(TAG, "Processed image " + (index + 1) + "/" + totalCount + ": " + imageUri);
            
        } catch (Exception e) {
            Log.e(TAG, "Error processing image: " + imageUri, e);
        }
    }
    
    private void startRecording() {
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
            String timestamp = sdf.format(new Date());
            boolean isLooking = isLookingCheckbox != null && isLookingCheckbox.isChecked();
            String suffix = isLooking ? "_looking" : "_notlooking";
            String filename = "batch_" + timestamp + suffix + ".csv";
            
            File dir = getExternalFilesDir(null);
            if (dir == null) {
                throw new IOException("External files directory not available");
            }
            File file = new File(dir, filename);
            currentRecordingFile = file.getAbsolutePath();
            
            recordingWriter = new BufferedWriter(new FileWriter(file));
            
            // Write CSV header
            StringBuilder header = new StringBuilder();
            header.append("filename,isLooking");
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
            
            Log.i(TAG, "Started recording to: " + currentRecordingFile);
            
        } catch (IOException e) {
            Log.e(TAG, "Failed to start recording: " + e.getMessage());
        }
    }
    
    private void stopRecording() {
        if (recordingWriter != null) {
            try {
                recordingWriter.close();
                recordingWriter = null;
                Log.i(TAG, "Stopped recording, saved to: " + currentRecordingFile);
            } catch (IOException e) {
                Log.e(TAG, "Error closing recording file: " + e.getMessage());
            }
        }
    }
    
    /**
     * Extract head pose Euler angles (pitch, yaw, roll) from rotation vector
     */
    private float[] extractHeadPose(Mat rvec) {
        if (rvec == null || rvec.empty()) {
            return new float[]{0, 0, 0};
        }
        
        try {
            Mat rotMat = new Mat();
            org.opencv.calib3d.Calib3d.Rodrigues(rvec, rotMat);
            
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
            Log.e(TAG, "Error extracting head pose: " + e.getMessage());
            return new float[]{0, 0, 0};
        }
    }
    
    private void recordFrame(String filename, float[] landmarks, float[] gaze, Mat rvec) {
        if (recordingWriter == null) return;
        
        try {
            StringBuilder line = new StringBuilder();
            
            // Filename
            line.append("\"").append(filename).append("\"");
            
            // isLooking
            boolean isLooking = isLookingCheckbox != null && isLookingCheckbox.isChecked();
            line.append(",").append(isLooking ? "1" : "0");
            
            // Landmarks
            if (landmarks != null && landmarks.length >= 196) {
                for (int i = 0; i < 196; i++) {
                    line.append(",").append(landmarks[i]);
                }
            } else {
                for (int i = 0; i < 196; i++) {
                    line.append(",");
                }
            }
            
            // Extract head pose from rvec
            float[] headPose = extractHeadPose(rvec);
            float head_pitch = headPose[0];
            float head_yaw = headPose[1];
            float head_roll = headPose[2];
            
            // Gaze
            float gaze_pitch = (gaze != null && gaze.length >= 2) ? gaze[0] : 0;
            float gaze_yaw = (gaze != null && gaze.length >= 2) ? gaze[1] : 0;
            
            // Relative gaze
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
            Log.e(TAG, "Error writing frame: " + e.getMessage());
        }
    }
    
    @Override
    protected void onDestroy() {
        isProcessing = false;
        
        if (recordingWriter != null) {
            stopRecording();
        }
        
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

