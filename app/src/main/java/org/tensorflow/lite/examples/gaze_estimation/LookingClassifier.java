package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.calib3d.Calib3d;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Classifier to determine if user is looking at the camera.
 * 
 * Uses a TFLite model trained on gaze, head pose, and eye landmark features.
 * The model has built-in normalization, so raw features can be passed directly.
 * 
 * Input features (41 total):
 *   - 7 core features: gaze_pitch, gaze_yaw, head_pitch, head_yaw, head_roll, relative_pitch, relative_yaw
 *   - 34 landmark features: eye landmarks (lm60-67, lm68-75) and nose tip (lm54)
 * 
 * Output: probability (0-1) that user is looking at camera
 */
public class LookingClassifier {
    private static final String TAG = "LookingClassifier";
    private static final String MODEL_PATH = "looking_classifier.tflite";
    
    // Feature indices for landmarks (must match training order)
    // Order: lm64, lm65, lm66, lm67, lm68, lm69, lm70, lm71, lm72, lm73, lm74, lm75, lm54, lm60, lm61, lm62, lm63
    private static final int[] LANDMARK_INDICES = {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 54, 60, 61, 62, 63};
    
    private static final int NUM_FEATURES = 41;
    private static final float THRESHOLD = 0.5f;
    
    private Interpreter interpreter;
    private boolean isInitialized = false;
    
    // Reusable arrays for inference
    private float[][] inputArray = new float[1][NUM_FEATURES];
    private float[][] outputArray = new float[1][1];
    
    // Last prediction result
    private float lastProbability = 0f;
    private boolean lastIsLooking = false;
    
    public LookingClassifier() {}
    
    /**
     * Initialize the classifier by loading the TFLite model
     */
    public boolean initialize(Context context) {
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(2);
            interpreter = new Interpreter(modelBuffer, options);
            isInitialized = true;
            Log.i(TAG, "Looking classifier initialized successfully");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize looking classifier: " + e.getMessage());
            isInitialized = false;
            return false;
        }
    }
    
    /**
     * Load the TFLite model from assets
     */
    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    /**
     * Extract head pose Euler angles from rotation vector
     * @return float[3] = {pitch, yaw, roll} in radians
     */
    private float[] extractHeadPose(Mat rvec) {
        if (rvec == null || rvec.empty()) {
            return new float[]{0, 0, 0};
        }
        
        try {
            Mat rotMat = new Mat();
            Calib3d.Rodrigues(rvec, rotMat);
            
            double sy = Math.sqrt(rotMat.get(0, 0)[0] * rotMat.get(0, 0)[0] + 
                                  rotMat.get(1, 0)[0] * rotMat.get(1, 0)[0]);
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
    
    /**
     * Run inference to determine if user is looking at camera
     * 
     * @param gazePitchYaw Gaze pitch and yaw from gaze estimation (2 values)
     * @param rvec Rotation vector from head pose estimation
     * @param landmarks Facial landmarks array (196 values = 98 points * 2)
     * @return true if user is looking at camera
     */
    public boolean predict(float[] gazePitchYaw, Mat rvec, float[] landmarks) {
        if (!isInitialized || interpreter == null) {
            Log.w(TAG, "Classifier not initialized");
            return false;
        }
        
        if (gazePitchYaw == null || gazePitchYaw.length < 2) {
            return false;
        }
        
        try {
            // Extract gaze values
            float gaze_pitch = gazePitchYaw[0];
            float gaze_yaw = gazePitchYaw[1];
            
            // Extract head pose from rvec
            float[] headPose = extractHeadPose(rvec);
            float head_pitch = headPose[0];
            float head_yaw = headPose[1];
            float head_roll = headPose[2];
            
            // Calculate relative gaze
            float relative_pitch = gaze_pitch - head_pitch;
            float relative_yaw = gaze_yaw - head_yaw;
            
            // Build input array
            int idx = 0;
            
            // Core features (7)
            inputArray[0][idx++] = gaze_pitch;
            inputArray[0][idx++] = gaze_yaw;
            inputArray[0][idx++] = head_pitch;
            inputArray[0][idx++] = head_yaw;
            inputArray[0][idx++] = head_roll;
            inputArray[0][idx++] = relative_pitch;
            inputArray[0][idx++] = relative_yaw;
            
            // Landmark features (17 landmarks * 2 = 34)
            // Order must match training: lm64, lm65, lm66, lm67, lm68, lm69, lm70, lm71, lm72, lm73, lm74, lm75, lm54, lm60, lm61, lm62, lm63
            if (landmarks != null && landmarks.length >= 196) {
                for (int lmIdx : LANDMARK_INDICES) {
                    inputArray[0][idx++] = landmarks[lmIdx * 2];     // x
                    inputArray[0][idx++] = landmarks[lmIdx * 2 + 1]; // y
                }
            } else {
                // Fill with zeros if no landmarks
                for (int i = 0; i < 34; i++) {
                    inputArray[0][idx++] = 0;
                }
            }
            
            // Run inference
            interpreter.run(inputArray, outputArray);
            
            // Get result
            lastProbability = outputArray[0][0];
            lastIsLooking = lastProbability > THRESHOLD;
            
            return lastIsLooking;
            
        } catch (Exception e) {
            Log.e(TAG, "Inference error: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Get the probability from the last prediction
     */
    public float getLastProbability() {
        return lastProbability;
    }
    
    /**
     * Get whether user was looking in last prediction
     */
    public boolean isLooking() {
        return lastIsLooking;
    }
    
    /**
     * Check if classifier is ready
     */
    public boolean isInitialized() {
        return isInitialized;
    }
    
    /**
     * Release resources
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        isInitialized = false;
    }
}

