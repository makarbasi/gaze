package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.calib3d.Calib3d;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Classifier to determine if user is looking at the camera.
 * 
 * Uses a TFLite model trained on gaze, head pose, and eye landmark features.
 * IMPORTANT: The exported TFLite models in this repo already include normalization
 * inside the graph (see training scripts). This class therefore feeds RAW features
 * in the exact order specified by the metadata "features" list.
 * 
 * Output: probability (0-1) that user is looking at camera
 */
public class LookingClassifier {
    private static final String TAG = "LookingClassifier";
    // Prefer v2 names; keep fallbacks for older asset naming.
    private static final String MODEL_PATH_PRIMARY = "looking_classifier_v2.tflite";
    private static final String MODEL_PATH_FALLBACK = "looking_classifier.tflite";
    private static final String METADATA_PATH_PRIMARY = "model_metadata_v2.json";
    private static final String METADATA_PATH_FALLBACK = "model_metadata.json";

    private static final float THRESHOLD = 0.5f;
    
    private Interpreter interpreter;
    private boolean isInitialized = false;
    private String lastInitError = null;
    
    // Feature list/order from metadata (must match training & model graph).
    private String[] featureNames = null;
    private FeatureSpec[] featureSpecs = null;
    private int numFeatures = -1;

    // Reusable arrays for inference (allocated after metadata is loaded)
    private float[][] inputArray = null;
    private final float[][] outputArray = new float[1][1];
    
    // Last prediction result
    private float lastProbability = 0f;
    private boolean lastIsLooking = false;
    
    public LookingClassifier() {}
    
    /**
     * Initialize the classifier by loading the TFLite model and metadata
     */
    public boolean initialize(Context context) {
        try {
            lastInitError = null;
            // Load normalization metadata first
            if (!loadMetadata(context)) {
                Log.e(TAG, "Failed to load model metadata");
                if (lastInitError == null) {
                    lastInitError = "Failed to load metadata (unknown)";
                }
                return false;
            }
            
            // Load TFLite model
            MappedByteBuffer modelBuffer = loadModelFile(context);
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(2);
            interpreter = new Interpreter(modelBuffer, options);
            isInitialized = true;
            lastInitError = null;
            Log.i(TAG, "✓ Looking classifier initialized with " + numFeatures + " features");
            return true;
        } catch (Exception e) {
            lastInitError = e.getClass().getSimpleName() + ": " + e.getMessage();
            Log.e(TAG, "Failed to initialize looking classifier: " + lastInitError);
            e.printStackTrace();
            isInitialized = false;
            return false;
        }
    }
    
    /**
     * A small parsed representation of each feature to fill at runtime.
     */
    private static final class FeatureSpec {
        enum Kind {
            GAZE_PITCH, GAZE_YAW,
            HEAD_PITCH, HEAD_YAW, HEAD_ROLL,
            REL_PITCH, REL_YAW,
            LANDMARK_X, LANDMARK_Y,
            UNKNOWN
        }
        final Kind kind;
        final int landmarkIndex; // only for LANDMARK_X / LANDMARK_Y
        FeatureSpec(Kind kind, int landmarkIndex) {
            this.kind = kind;
            this.landmarkIndex = landmarkIndex;
        }
    }

    /**
     * Load metadata JSON from assets and precompute feature mapping.
     *
     * Expected keys:
     *  - mean: [num_features] (for reference; model already embeds normalization)
     *  - scale: [num_features] (for reference)
     *  - features: [num_features] list of feature names in training order
     */
    private boolean loadMetadata(Context context) {
        try {
            JSONObject json = readJsonAsset(context, METADATA_PATH_PRIMARY, METADATA_PATH_FALLBACK);
            if (json == null) {
                if (lastInitError == null) {
                    lastInitError = "Could not open metadata asset(s)";
                }
                return false;
            }

            JSONArray featuresArray = json.optJSONArray("features");
            if (featuresArray == null) {
                Log.e(TAG, "Metadata missing 'features' array");
                lastInitError = "Metadata missing 'features' array";
                return false;
            }

            featureNames = new String[featuresArray.length()];
            featureSpecs = new FeatureSpec[featuresArray.length()];
            for (int i = 0; i < featuresArray.length(); i++) {
                String name = featuresArray.getString(i);
                featureNames[i] = name;
                featureSpecs[i] = parseFeatureSpec(name);
            }
            numFeatures = featureNames.length;

            // Allocate input buffer now that we know the correct size
            inputArray = new float[1][numFeatures];

            // Optional sanity check: mean/scale lengths should match feature length (but not required)
            JSONArray meanArray = json.optJSONArray("mean");
            JSONArray scaleArray = json.optJSONArray("scale");
            if (meanArray != null && scaleArray != null &&
                    (meanArray.length() != numFeatures || scaleArray.length() != numFeatures)) {
                Log.w(TAG, "Metadata mean/scale length mismatch: mean=" +
                        meanArray.length() + ", scale=" + scaleArray.length() +
                        ", features=" + numFeatures);
            }

            Log.i(TAG, "Loaded metadata: features=" + numFeatures);
            return numFeatures > 0;
        } catch (Exception e) {
            Log.e(TAG, "Error loading metadata: " + e.getMessage());
            lastInitError = "Error loading metadata: " + e.getMessage();
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Load the TFLite model from assets
     */
    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor;
        try {
            fileDescriptor = context.getAssets().openFd(MODEL_PATH_PRIMARY);
            Log.i(TAG, "Using model asset: " + MODEL_PATH_PRIMARY);
        } catch (IOException primaryErr) {
            try {
                fileDescriptor = context.getAssets().openFd(MODEL_PATH_FALLBACK);
                Log.w(TAG, "Using fallback model asset: " + MODEL_PATH_FALLBACK);
            } catch (IOException fallbackErr) {
                lastInitError =
                        "openFd failed for model assets: " + MODEL_PATH_PRIMARY + ", " + MODEL_PATH_FALLBACK +
                        " (" + fallbackErr.getMessage() + ")";
                throw fallbackErr;
            }
        }

        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
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
        if (inputArray == null || featureSpecs == null || numFeatures <= 0) {
            Log.w(TAG, "Metadata not loaded");
            return false;
        }
        
        if (gazePitchYaw == null || gazePitchYaw.length < 2) {
            Log.w(TAG, "Invalid gaze data");
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
            
            // Build input array (RAW features in training order).
            // Do NOT apply StandardScaler normalization here — models are exported with normalization embedded.
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
                    case UNKNOWN:
                    default:
                        v = 0f;
                }
                inputArray[0][i] = v;
            }
            
            // Run inference
            interpreter.run(inputArray, outputArray);
            
            // Get result
            lastProbability = outputArray[0][0];
            lastIsLooking = lastProbability > THRESHOLD;
            
            // Per-frame logging is noisy; only log when explicitly enabled via display_config.json
            if (DisplayConfig.getInstance().debugLogs) {
                Log.d(TAG, String.format("ML: prob=%.2f%%, looking=%b (gaze: p=%.2f, y=%.2f)",
                    lastProbability * 100, lastIsLooking, gaze_pitch, gaze_yaw));
            }
            
            return lastIsLooking;
            
        } catch (Exception e) {
            Log.e(TAG, "Inference error: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    private static float getLandmarkCoord(float[] landmarks, int idx, boolean isX) {
        if (landmarks == null) {
            return 0f;
        }
        int base = idx * 2 + (isX ? 0 : 1);
        if (base < 0 || base >= landmarks.length) {
            return 0f;
        }
        return landmarks[base];
    }

    private static FeatureSpec parseFeatureSpec(String name) {
        // Core features
        switch (name) {
            case "gaze_pitch": return new FeatureSpec(FeatureSpec.Kind.GAZE_PITCH, -1);
            case "gaze_yaw": return new FeatureSpec(FeatureSpec.Kind.GAZE_YAW, -1);
            case "head_pitch": return new FeatureSpec(FeatureSpec.Kind.HEAD_PITCH, -1);
            case "head_yaw": return new FeatureSpec(FeatureSpec.Kind.HEAD_YAW, -1);
            case "head_roll": return new FeatureSpec(FeatureSpec.Kind.HEAD_ROLL, -1);
            case "relative_pitch": return new FeatureSpec(FeatureSpec.Kind.REL_PITCH, -1);
            case "relative_yaw": return new FeatureSpec(FeatureSpec.Kind.REL_YAW, -1);
        }

        // Landmark features: lm{idx}_x or lm{idx}_y
        // Example: lm64_x, lm60_y
        if (name != null && name.startsWith("lm") && name.length() >= 5) {
            try {
                int underscore = name.indexOf('_');
                // Allow 1-char axis ("x"/"y"): underscore must be before last char
                if (underscore > 2 && underscore < name.length() - 1) {
                    int lmIdx = Integer.parseInt(name.substring(2, underscore));
                    String axis = name.substring(underscore + 1).trim();
                    if ("x".equals(axis)) {
                        return new FeatureSpec(FeatureSpec.Kind.LANDMARK_X, lmIdx);
                    } else if ("y".equals(axis)) {
                        return new FeatureSpec(FeatureSpec.Kind.LANDMARK_Y, lmIdx);
                    }
                }
            } catch (Exception ignored) {
                // fall through
            }
        }

        Log.w(TAG, "Unknown feature name in metadata: '" + name + "' (using 0)");
        return new FeatureSpec(FeatureSpec.Kind.UNKNOWN, -1);
    }

    private static JSONObject readJsonAsset(Context context, String primary, String fallback) throws Exception {
        JSONObject json = null;
        Exception last = null;
        for (String path : new String[]{primary, fallback}) {
            if (path == null) continue;
            try (InputStream is = context.getAssets().open(path);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                }
                json = new JSONObject(sb.toString());
                Log.i(TAG, "Using metadata asset: " + path);
                return json;
            } catch (Exception e) {
                last = e;
            }
        }
        if (last != null) {
            Log.e(TAG, "Failed to load metadata from assets. Tried: " + primary + ", " + fallback + ". Error: " + last.getMessage());
        }
        return null;
    }

    /**
     * If initialization failed, this gives the last error seen (for UI/logcat debugging).
     */
    public String getLastInitError() {
        return lastInitError;
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
