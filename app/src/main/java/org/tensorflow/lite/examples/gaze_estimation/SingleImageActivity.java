package org.tensorflow.lite.examples.gaze_estimation;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
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
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

// Same imports as ClassifierActivity
import static com.example.gazedemo1.ProcessFactory.bitmap2mat;
import static com.example.gazedemo1.ProcessFactory.gaze_postprocess;
import static com.example.gazedemo1.ProcessFactory.gaze_preprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_postprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_preprocess;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawbox;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawgaze;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawlandmark;

import com.example.gazedemo1.ProcessFactory;

/**
 * Single Image Activity - feeds a static image to the SAME processing pipeline as camera.
 * 
 * Camera does: camera frame -> resize to 480x480 -> face detection -> landmarks -> gaze
 * This does:   static image -> resize to 480x480 -> face detection -> landmarks -> gaze
 * 
 * SAME logic, just different input source.
 */
public class SingleImageActivity extends AppCompatActivity {
    private static final String TAG = "SingleImage";
    private static final String RECORD_FILE = "/data/local/tmp/record.csv";
    
    // UI
    private TextView statusText;
    private ImageView resultImageView;
    private Button selectImageButton;
    private ProgressBar progressBar;
    
    // Models - SAME as ClassifierActivity
    private TFLiteFaceDetector tfliteFaceDetector = null;
    private QNNModelRunner landmark_detection_qnn = null;
    private QNNModelRunner gaze_estimation_qnn = null;
    
    private ExecutorService executorService;
    private Handler mainHandler;
    
    private static final int REQUEST_IMAGE_PICKER = 2001;
    private volatile boolean modelsReady = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Init OpenCV - same as ClassifierActivity
        OpenCVLoader.initDebug();
        
        mainHandler = new Handler(Looper.getMainLooper());
        executorService = Executors.newSingleThreadExecutor();
        
        createUI();
        initModels();
    }
    
    private void createUI() {
        ScrollView scrollView = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 48, 32, 48);
        layout.setBackgroundColor(0xFF1A1A2E);
        
        TextView title = new TextView(this);
        title.setText("Single Image Mode");
        title.setTextSize(24);
        title.setTextColor(0xFFFFFFFF);
        title.setGravity(Gravity.CENTER);
        layout.addView(title);
        
        statusText = new TextView(this);
        statusText.setText("Loading models...");
        statusText.setTextSize(14);
        statusText.setTextColor(0xFFFFD700);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(0, 16, 0, 16);
        layout.addView(statusText);
        
        progressBar = new ProgressBar(this);
        LinearLayout.LayoutParams pParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        pParams.gravity = Gravity.CENTER;
        progressBar.setLayoutParams(pParams);
        layout.addView(progressBar);
        
        selectImageButton = new Button(this);
        selectImageButton.setText("Select Image");
        selectImageButton.setEnabled(false);
        selectImageButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("image/*");
            startActivityForResult(intent, REQUEST_IMAGE_PICKER);
        });
        LinearLayout.LayoutParams bParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        bParams.setMargins(0, 24, 0, 24);
        selectImageButton.setLayoutParams(bParams);
        layout.addView(selectImageButton);
        
        resultImageView = new ImageView(this);
        resultImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        resultImageView.setAdjustViewBounds(true);
        resultImageView.setBackgroundColor(0xFF333333);
        LinearLayout.LayoutParams iParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 500);
        resultImageView.setLayoutParams(iParams);
        layout.addView(resultImageView);
        
        Button backButton = new Button(this);
        backButton.setText("Back");
        backButton.setOnClickListener(v -> {
            startActivity(new Intent(this, LauncherActivity.class));
            finish();
        });
        LinearLayout.LayoutParams backParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        backParams.setMargins(0, 24, 0, 0);
        backButton.setLayoutParams(backParams);
        layout.addView(backButton);
        
        scrollView.addView(layout);
        setContentView(scrollView);
    }
    
    /**
     * Initialize models - SAME as ClassifierActivity.onPreviewSizeChosen()
     */
    private void initModels() {
        executorService.execute(() -> {
            try {
                // Model 1: Face Detection - same as ClassifierActivity
                if (DemoConfig.USE_TFLITE_FACE_DETECTION) {
                    tfliteFaceDetector = new TFLiteFaceDetector(this);
                    tfliteFaceDetector.initialize();
                    Log.i(TAG, "Face detector loaded");
                }
                
                // Model 2: Landmark Detection - same as ClassifierActivity  
                landmark_detection_qnn = new QNNModelRunner();
                landmark_detection_qnn.loadModel(
                    DemoConfig.landmark_detection_model_path,
                    DemoConfig.landmark_detection_model_out,
                    QNNModelRunner.Backend.GPU);
                Log.i(TAG, "Landmark model loaded");
                
                // Model 3: Gaze Estimation - same as ClassifierActivity
                gaze_estimation_qnn = new QNNModelRunner();
                gaze_estimation_qnn.loadModel(
                    DemoConfig.gaze_estimation_model_path,
                    DemoConfig.gaze_estimation_model_out,
                    QNNModelRunner.Backend.GPU);
                Log.i(TAG, "Gaze model loaded");
                
                modelsReady = true;
                mainHandler.post(() -> {
                    statusText.setText("Ready - Select an image");
                    statusText.setTextColor(0xFF00FF00);
                    progressBar.setVisibility(View.GONE);
                    selectImageButton.setEnabled(true);
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Model loading failed: " + e.getMessage());
                mainHandler.post(() -> {
                    statusText.setText("Failed: " + e.getMessage());
                    statusText.setTextColor(0xFFFF0000);
                    progressBar.setVisibility(View.GONE);
                });
            }
        });
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_PICKER && resultCode == Activity.RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                processImage(uri);
            }
        }
    }
    
    /**
     * Process static image using EXACT SAME logic as ClassifierActivity.processImage()
     * 
     * The camera code does:
     *   1. Get camera frame
     *   2. Resize/crop to 480x480 (DemoConfig.crop_W x crop_H)
     *   3. Face detection on the 480x480 bitmap
     *   4. Landmark detection
     *   5. Gaze estimation
     *   6. Draw results
     * 
     * We do the SAME thing, just with a static image instead of camera frame.
     */
    private void processImage(Uri imageUri) {
        statusText.setText("Processing...");
        statusText.setTextColor(0xFFFFD700);
        progressBar.setVisibility(View.VISIBLE);
        selectImageButton.setEnabled(false);
        
        executorService.execute(() -> {
            try {
                // === STEP 1: Load image ===
                InputStream is = getContentResolver().openInputStream(imageUri);
                Bitmap originalBitmap = BitmapFactory.decodeStream(is);
                is.close();
                
                if (originalBitmap == null) {
                    throw new Exception("Failed to load image");
                }
                
                // === STEP 2: Resize to 480x480 - SAME as camera croppedBitmap ===
                // This is what ClassifierActivity does: resize camera frame to DemoConfig.crop_W x crop_H
                Mat imgMat = bitmap2mat(originalBitmap);
                Imgproc.resize(imgMat, imgMat, new org.opencv.core.Size(DemoConfig.crop_W, DemoConfig.crop_H));
                
                Bitmap croppedBitmap = Bitmap.createBitmap(DemoConfig.crop_W, DemoConfig.crop_H, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(imgMat, croppedBitmap);
                
                Log.i(TAG, "Image resized to " + DemoConfig.crop_W + "x" + DemoConfig.crop_H);
                
                // === From here: EXACT SAME as ClassifierActivity.processImage() ===
                
                Bitmap processedBitmap = croppedBitmap;
                Mat img = bitmap2mat(processedBitmap);
                
                // === FACE DETECTION - same as ClassifierActivity ===
                float[][] boxes = null;
                if (DemoConfig.USE_TFLITE_FACE_DETECTION && tfliteFaceDetector != null) {
                    boxes = tfliteFaceDetector.detectFaces(processedBitmap);
                    Log.i(TAG, "Face detection: " + (boxes != null ? boxes.length : 0) + " faces");
                }
                
                if (boxes == null || boxes.length == 0) {
                    // No face - save empty record and show result
                    saveRecord(imageUri.getLastPathSegment(), null, null);
                    showResult(croppedBitmap, "No face detected");
                    return;
                }
                
                Vector<float[]> landmarks = new Vector<>();
                Vector<float[]> gazes = new Vector<>();
                
                // === LANDMARK + GAZE for each face - same as ClassifierActivity ===
                for (int b = 0; b < boxes.length; b++) {
                    float[] box = boxes[b];
                    
                    // Landmark detection - same as ClassifierActivity
                    ProcessFactory.LandmarkPreprocessResult lmResult = landmark_preprocess(img, box);
                    int[] lmDims = {1, DemoConfig.landmark_detection_input_H, 
                        DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C};
                    float[] lmOutput = landmark_detection_qnn.runInference("face_image", lmResult.input, lmDims);
                    
                    if (lmOutput == null) continue;
                    
                    float[] landmark = landmark_postprocess(lmResult, lmOutput);
                    landmarks.add(landmark);
                    
                    // Gaze estimation - same as ClassifierActivity
                    ProcessFactory.GazePreprocessResult gazeResult = gaze_preprocess(img, landmark);
                    String[] inputNames = {"left_eye", "right_eye", "face"};
                    float[][] inputArrays = {gazeResult.leye, gazeResult.reye, gazeResult.face};
                    int[][] inputDims = {
                        {1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C},
                        {1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C},
                        {1, DemoConfig.gaze_estimation_face_input_H, DemoConfig.gaze_estimation_face_input_W, DemoConfig.gaze_estimation_face_input_C}
                    };
                    float[] gazeOutput = gaze_estimation_qnn.runMultiInputInference(inputNames, inputArrays, inputDims);
                    
                    if (gazeOutput != null) {
                        float[] gaze = gaze_postprocess(gazeOutput, gazeResult.R);
                        gazes.add(gaze);
                        Log.i(TAG, "Gaze: pitch=" + Math.toDegrees(gaze[0]) + ", yaw=" + Math.toDegrees(gaze[1]));
                    }
                }
                
                // === DRAW RESULTS - same as ClassifierActivity ===
                for (int i = 0; i < gazes.size(); i++) {
                    drawgaze(img, gazes.get(i), landmarks.get(i));
                }
                
                Utils.matToBitmap(img, processedBitmap);
                int[] pixels = new int[DemoConfig.crop_W * DemoConfig.crop_H];
                processedBitmap.getPixels(pixels, 0, DemoConfig.crop_W, 0, 0, DemoConfig.crop_W, DemoConfig.crop_H);
                
                drawbox(pixels, boxes, DemoConfig.crop_H, DemoConfig.crop_W);
                for (float[] lm : landmarks) {
                    drawlandmark(pixels, lm, DemoConfig.crop_H, DemoConfig.crop_W);
                }
                
                processedBitmap.setPixels(pixels, 0, DemoConfig.crop_W, 0, 0, DemoConfig.crop_W, DemoConfig.crop_H);
                
                // === SAVE RECORD ===
                float[] firstLandmark = landmarks.size() > 0 ? landmarks.get(0) : null;
                float[] firstGaze = gazes.size() > 0 ? gazes.get(0) : null;
                saveRecord(imageUri.getLastPathSegment(), firstLandmark, firstGaze);
                
                String msg = "Faces: " + boxes.length;
                if (firstGaze != null) {
                    msg += String.format("\nPitch: %.1f°, Yaw: %.1f°", 
                        Math.toDegrees(firstGaze[0]), Math.toDegrees(firstGaze[1]));
                }
                showResult(processedBitmap, msg);
                
            } catch (Exception e) {
                Log.e(TAG, "Error: " + e.getMessage(), e);
                mainHandler.post(() -> {
                    statusText.setText("Error: " + e.getMessage());
                    statusText.setTextColor(0xFFFF0000);
                    progressBar.setVisibility(View.GONE);
                    selectImageButton.setEnabled(true);
                });
            }
        });
    }
    
    private void saveRecord(String filename, float[] landmarks, float[] gaze) {
        try {
            File file = new File(RECORD_FILE);
            boolean isNew = !file.exists();
            BufferedWriter w = new BufferedWriter(new FileWriter(file, true));
            
            if (isNew) {
                StringBuilder h = new StringBuilder("timestamp,filename");
                for (int i = 0; i < 98; i++) h.append(",lm").append(i).append("_x,lm").append(i).append("_y");
                h.append(",pitch,yaw\n");
                w.write(h.toString());
            }
            
            StringBuilder line = new StringBuilder();
            line.append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()));
            line.append(",").append(filename != null ? filename : "");
            
            if (landmarks != null && landmarks.length >= 196) {
                for (int i = 0; i < 196; i++) line.append(",").append(landmarks[i]);
            } else {
                for (int i = 0; i < 196; i++) line.append(",");
            }
            
            if (gaze != null && gaze.length >= 2) {
                line.append(",").append(gaze[0]).append(",").append(gaze[1]);
            } else {
                line.append(",,");
            }
            line.append("\n");
            
            w.write(line.toString());
            w.close();
            Log.i(TAG, "Saved to " + RECORD_FILE);
        } catch (IOException e) {
            Log.e(TAG, "Save failed: " + e.getMessage());
        }
    }
    
    private void showResult(Bitmap bitmap, String message) {
        mainHandler.post(() -> {
            resultImageView.setImageBitmap(bitmap);
            statusText.setText(message + "\nSaved to " + RECORD_FILE);
            statusText.setTextColor(0xFF00FF00);
            progressBar.setVisibility(View.GONE);
            selectImageButton.setEnabled(true);
        });
    }
    
    @Override
    protected void onDestroy() {
        if (tfliteFaceDetector != null) tfliteFaceDetector.close();
        if (landmark_detection_qnn != null) landmark_detection_qnn.close();
        if (gaze_estimation_qnn != null) gaze_estimation_qnn.close();
        if (executorService != null) executorService.shutdown();
        super.onDestroy();
    }
}
