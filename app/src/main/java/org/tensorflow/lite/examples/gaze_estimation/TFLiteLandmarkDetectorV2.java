package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;

/**
 * TensorFlow Lite based Landmark Detector V2 using FaceMap_3DMM model.
 * This is the more accurate model from HotGaze project.
 * 
 * Input:  face_image (128×128×3) - RGB UINT8
 * Model:  facial_landmark_detection.tflite (FaceMap_3DMM)
 * Output: 265 3DMM parameters which are projected to 98 landmarks (196 values)
 *
 * 3DMM Output Structure (265 values):
 *   [0-218]:   alpha_id (shape coefficients)
 *   [219-257]: alpha_exp (expression coefficients)
 *   [258]:     pitch angle (normalized to [-1, 1])
 *   [259]:     yaw angle (normalized to [-1, 1])
 *   [260]:     roll angle (normalized to [-1, 1])
 *   [261-264]: translation (tX, tY) and focal length
 */
public class TFLiteLandmarkDetectorV2 {
    private static final String TAG = "TFLiteLandmarkV2";
    
    private Interpreter interpreter;
    private NnApiDelegate nnApiDelegate;
    private String acceleratorType = "CPU";
    
    // Model parameters for FaceMap_3DMM
    public static final int INPUT_HEIGHT = 128;
    public static final int INPUT_WIDTH = 128;
    public static final int INPUT_CHANNELS = 3;
    public static final int TDDM_OUTPUT_SIZE = 265;  // 3DMM parameters
    public static final int LANDMARK_COUNT = 98;     // Number of landmarks to project
    public static final int OUTPUT_SIZE = 196;       // 98 landmarks × 2 (x, y) - compatible with existing pipeline
    
    private final Context context;
    private boolean isInitialized = false;
    
    // Reusable buffers
    private ByteBuffer inputBuffer;
    private float[] outputArray;
    private float[] projectedLandmarks;
    
    // 3D face landmark template (98 points for full face model)
    private float[][] landmarkTemplate;
    
    public TFLiteLandmarkDetectorV2(Context context) {
        this.context = context;
        initializeLandmarkTemplate();
    }
    
    /**
     * Initialize the TFLite landmark detector with NNAPI acceleration
     */
    public void initialize() throws IOException {
        Log.i(TAG, "========================================");
        Log.i(TAG, "Initializing TFLite Landmark Detection V2");
        Log.i(TAG, "Model: FaceMap_3DMM (facial_landmark_detection.tflite)");
        Log.i(TAG, "========================================");
        
        // Load facial landmark detection model
        Log.i(TAG, "Loading Facial Landmark Detection model...");
        MappedByteBuffer modelBuffer = AssetUtils.loadMappedFile(context, "facial_landmark_detection.tflite");
        
        Interpreter.Options options = new Interpreter.Options();
        
        // Try to use NNAPI delegate for GPU acceleration
        Log.i(TAG, "Device: " + Build.MANUFACTURER + " " + Build.MODEL);
        Log.i(TAG, "Configuring NNAPI for GPU acceleration...");
        
        try {
            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
            nnApiOptions.setAllowFp16(true);
            nnApiOptions.setUseNnapiCpu(false);  // NO CPU FALLBACK
            
            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            options.addDelegate(nnApiDelegate);
            
            // Identify device type
            if (Build.HARDWARE.toLowerCase().contains("qcom") || 
                Build.HARDWARE.toLowerCase().contains("qualcomm")) {
                acceleratorType = "Qualcomm Snapdragon (NNAPI → Adreno GPU + Hexagon DSP)";
            } else {
                acceleratorType = "NNAPI (Hardware Accelerated)";
            }
            
            Log.i(TAG, "✓ NNAPI delegate configured");
            Log.i(TAG, "Accelerator: " + acceleratorType);
        } catch (Exception e) {
            Log.w(TAG, "NNAPI initialization failed, using CPU: " + e.getMessage());
            acceleratorType = "CPU (NNAPI unavailable)";
        }
        
        options.setNumThreads(4);
        
        interpreter = new Interpreter(modelBuffer, options);
        
        // Log input/output tensor info
        int inputCount = interpreter.getInputTensorCount();
        int outputCount = interpreter.getOutputTensorCount();
        Log.i(TAG, "Model inputs: " + inputCount + ", outputs: " + outputCount);
        
        for (int i = 0; i < inputCount; i++) {
            int[] shape = interpreter.getInputTensor(i).shape();
            Log.i(TAG, "Input " + i + " shape: " + shapeToString(shape) + ", type: " + interpreter.getInputTensor(i).dataType());
        }
        for (int i = 0; i < outputCount; i++) {
            int[] shape = interpreter.getOutputTensor(i).shape();
            Log.i(TAG, "Output " + i + " shape: " + shapeToString(shape) + ", type: " + interpreter.getOutputTensor(i).dataType());
        }
        
        // Pre-allocate buffers
        // Input is UINT8 for quantized model
        int inputSize = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
        inputBuffer = ByteBuffer.allocateDirect(inputSize);
        inputBuffer.order(ByteOrder.nativeOrder());
        
        outputArray = new float[TDDM_OUTPUT_SIZE];
        projectedLandmarks = new float[OUTPUT_SIZE];
        
        isInitialized = true;
        
        Log.i(TAG, "════════════════════════════════════════");
        Log.i(TAG, "✓ Landmark Detection V2 Ready");
        Log.i(TAG, "  Input size: " + INPUT_WIDTH + "x" + INPUT_HEIGHT + "x" + INPUT_CHANNELS);
        Log.i(TAG, "  3DMM output: " + TDDM_OUTPUT_SIZE + " parameters");
        Log.i(TAG, "  Projected landmarks: " + OUTPUT_SIZE + " values (" + LANDMARK_COUNT + " points)");
        Log.i(TAG, "  Accelerator: " + acceleratorType);
        Log.i(TAG, "════════════════════════════════════════");
    }
    
    private String shapeToString(int[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * Run landmark detection on preprocessed input
     * @param input Float array from landmark_preprocess (normalized 0-1, size: 128*128*3)
     * @return Landmark output array (196 values = 98 landmarks × 2) in INPUT_SIZE coordinates
     */
    public float[] detect(float[] input) {
        if (!isInitialized || interpreter == null) {
            Log.e(TAG, "Landmark detector not initialized!");
            return null;
        }
        
        int expectedSize = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
        if (input == null || input.length != expectedSize) {
            Log.e(TAG, "Invalid input size: " + (input != null ? input.length : "null") + 
                       ", expected: " + expectedSize);
            return null;
        }
        
        try {
            // Convert float input (0-1) to UINT8 (0-255) for quantized model
            // This matches HotGaze's preprocessing: raw RGB bytes
            inputBuffer.rewind();
            for (float value : input) {
                int byteVal = (int) (value * 255.0f);
                byteVal = Math.max(0, Math.min(255, byteVal));
                inputBuffer.put((byte) byteVal);
            }
            inputBuffer.rewind();
            
            // Get output tensor info
            org.tensorflow.lite.Tensor outputTensor = interpreter.getOutputTensor(0);
            int[] outputShape = outputTensor.shape();
            
            // Create output buffer - model outputs UINT8 quantized (like HotGaze)
            byte[][] outputBuffer = new byte[outputShape[0]][outputShape[1]];
            
            // Run inference
            long startTime = System.currentTimeMillis();
            interpreter.run(inputBuffer, outputBuffer);
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Landmark detection inference: " + inferenceTime + "ms");
            
            // Dequantize output (following HotGaze exactly)
            float scale = outputTensor.quantizationParams().getScale();
            int zeroPoint = outputTensor.quantizationParams().getZeroPoint();
            
            Log.d(TAG, "Output quantization: scale=" + scale + ", zeroPoint=" + zeroPoint);
            
            float[] parameters = new float[outputBuffer[0].length];
            for (int i = 0; i < outputBuffer[0].length; i++) {
                int quantizedValue = outputBuffer[0][i] & 0xFF;
                parameters[i] = (quantizedValue - zeroPoint) * scale;
            }
            
            Log.d(TAG, "3DMM parameters size: " + parameters.length);
            
            // Project 98 landmarks from 3DMM parameters
            // Returns landmarks in INPUT_SIZE (128x128) coordinate space, normalized to [0,1]
            return projectLandmarks(parameters);
            
        } catch (Exception e) {
            Log.e(TAG, "Landmark detection error: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Run landmark detection on a Bitmap - EXACTLY like HotGaze
     * This bypasses float conversion and works directly with raw RGB bytes.
     * 
     * @param faceBitmap The 128x128 face crop bitmap (will be resized if not 128x128)
     * @return Landmark output array (196 values = 98 landmarks × 2) normalized to [0,1]
     */
    public float[] detectFromBitmap(Bitmap faceBitmap) {
        if (!isInitialized || interpreter == null) {
            Log.e(TAG, "Landmark detector not initialized!");
            return null;
        }
        
        if (faceBitmap == null) {
            Log.e(TAG, "Input bitmap is null!");
            return null;
        }
        
        try {
            // Resize to 128x128 if needed (like HotGaze: createScaledBitmap with bilinear filtering)
            Bitmap resizedFace = faceBitmap;
            if (faceBitmap.getWidth() != INPUT_WIDTH || faceBitmap.getHeight() != INPUT_HEIGHT) {
                resizedFace = Bitmap.createScaledBitmap(faceBitmap, INPUT_WIDTH, INPUT_HEIGHT, true);
            }
            
            // Create input buffer - EXACTLY like HotGaze
            inputBuffer.rewind();
            
            int[] intValues = new int[INPUT_WIDTH * INPUT_HEIGHT];
            resizedFace.getPixels(intValues, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
            
            // Convert to RGB bytes (UINT8 quantized input) - EXACTLY like HotGaze
            for (int pixelValue : intValues) {
                inputBuffer.put((byte) ((pixelValue >> 16) & 0xFF));  // R
                inputBuffer.put((byte) ((pixelValue >> 8) & 0xFF));   // G
                inputBuffer.put((byte) (pixelValue & 0xFF));          // B
            }
            inputBuffer.rewind();
            
            // Get output tensor info
            org.tensorflow.lite.Tensor outputTensor = interpreter.getOutputTensor(0);
            int[] outputShape = outputTensor.shape();
            
            // Create output buffer - EXACTLY like HotGaze: Array(outputShape[0]) { ByteArray(outputShape[1]) }
            byte[][] outputBuffer = new byte[outputShape[0]][outputShape[1]];
            
            // Run inference
            long startTime = System.currentTimeMillis();
            interpreter.run(inputBuffer, outputBuffer);
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Landmark detection (bitmap) inference: " + inferenceTime + "ms");
            
            // Get quantization parameters - EXACTLY like HotGaze
            float scale = outputTensor.quantizationParams().getScale();
            int zeroPoint = outputTensor.quantizationParams().getZeroPoint();
            
            Log.d(TAG, "Output quantization: scale=" + scale + ", zeroPoint=" + zeroPoint);
            
            // Dequantize all parameters (265 values) - EXACTLY like HotGaze
            float[] parameters = new float[outputBuffer[0].length];
            for (int i = 0; i < outputBuffer[0].length; i++) {
                int quantizedValue = outputBuffer[0][i] & 0xFF;
                parameters[i] = (quantizedValue - zeroPoint) * scale;
            }
            
            Log.d(TAG, "3DMM parameters size: " + parameters.length);
            
            // Log pose parameters for debugging
            if (parameters.length >= 264) {
                float pitch = parameters[258] * 90f;  // Normalized to degrees
                float yaw = parameters[259] * 90f;
                Log.d(TAG, "Head pose: pitch=" + pitch + "°, yaw=" + yaw + "°");
            }
            
            // Project landmarks
            return projectLandmarks(parameters);
            
        } catch (Exception e) {
            Log.e(TAG, "Landmark detection (bitmap) error: " + e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * Project 98 facial landmarks from 265 3DMM parameters.
     * Based on FaceMap_3DMM utils.py: project_landmark()
     * Following HotGaze's exact implementation.
     * 
     * @param parameters The 265 3DMM parameters from the model
     * @return Array of 196 values (98 landmarks × 2 coordinates) normalized to [0,1] in INPUT_SIZE space
     */
    private float[] projectLandmarks(float[] parameters) {
        if (parameters.length < 264) {
            Log.e(TAG, "Invalid 3DMM parameters size: " + parameters.length);
            return null;
        }
        
        // Parse 3DMM parameters (following HotGaze/utils.py)
        float pitch = parameters[258];  // Range: [-1, 1]
        float yaw = parameters[259];    // Range: [-1, 1]
        float roll = parameters[260];   // Range: [-1, 1]
        float tX = parameters[261];
        float tY = parameters[262];
        float f = parameters[263];
        
        // De-normalize to original range (following HotGaze exactly)
        float pitchRad = pitch * (float) Math.PI / 2f;
        float yawRad = yaw * (float) Math.PI / 2f;
        float rollRad = roll * (float) Math.PI / 2f;
        float denormTX = tX * 60f;
        float denormTY = tY * 60f;
        float denormTZ = 500f;
        float denormF = f * 150f + 450f;
        
        // Create rotation matrices (following HotGaze)
        float cosPitch = (float) Math.cos(pitchRad);
        float sinPitch = (float) Math.sin(pitchRad);
        float cosYaw = (float) Math.cos(yawRad);
        float sinYaw = (float) Math.sin(yawRad);
        
        // Apply rotation and projection to each landmark (following HotGaze exactly)
        for (int i = 0; i < LANDMARK_COUNT; i++) {
            float x = landmarkTemplate[i][0];
            float y = landmarkTemplate[i][1];
            float z = landmarkTemplate[i][2];
            
            // Apply rotation (yaw and pitch only, like HotGaze)
            float x1 = x * cosYaw - z * sinYaw;
            float z1 = x * sinYaw + z * cosYaw;
            float y1 = y * cosPitch - z1 * sinPitch;
            float z2 = y * sinPitch + z1 * cosPitch;
            
            // Apply translation
            float x2 = x1 + denormTX;
            float y2 = y1 + denormTY;
            float z3 = z2 + denormTZ;
            
            // Project to 2D using perspective projection (following HotGaze exactly)
            // Note: HotGaze does NOT add image center offset here
            float projX = x2 * denormF / z3;
            float projY = y2 * denormF / z3;
            
            // Center on the 128x128 image (landmarks need to be in image coordinates)
            // The model expects face centered in image, so add half the image size
            projX = projX + INPUT_WIDTH / 2f;
            projY = projY + INPUT_HEIGHT / 2f;
            
            // Normalize to [0, 1] range for postprocessing
            projectedLandmarks[i * 2] = projX / INPUT_WIDTH;
            projectedLandmarks[i * 2 + 1] = projY / INPUT_HEIGHT;
        }
        
        return projectedLandmarks.clone();
    }
    
    /**
     * Initialize the 98-point 3D facial landmark template.
     * WFLW 98-point format (used by PFLD and the DLC model this replaces):
     * - 0-32: Jaw/chin (33 points)
     * - 33-41: Left eyebrow (9 points)
     * - 42-50: Right eyebrow (9 points)
     * - 51-54: Nose bridge (4 points)
     * - 55-59: Nose bottom (5 points)
     * - 60-67: Left eye (8 points)
     * - 68-75: Right eye (8 points)
     * - 76-87: Outer lips (12 points)
     * - 88-95: Inner lips (8 points)
     * - 96-97: Eye pupils
     * 
     * TRACKED_POINTS = {33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16}
     * 
     * This template is scaled to match HotGaze's 68-point template which uses:
     * - Jaw width: ~100 units (-50 to +50)
     * - Eyes at y ≈ -10
     * - Mouth at y ≈ 25
     * - Chin at y ≈ 40
     */
    private void initializeLandmarkTemplate() {
        landmarkTemplate = new float[LANDMARK_COUNT][3];
        
        // ===== JAW/CHIN (0-32): 33 points from right to left =====
        // Using HotGaze style: x from -50 to 50, y curved from 40 down to chin
        for (int i = 0; i <= 32; i++) {
            float t = (i - 16) / 16f;  // -1 to 1
            landmarkTemplate[i] = new float[]{
                t * 50f,                              // X: -50 to 50
                40f - Math.abs(t) * 20f,              // Y: 40 at edges, 20 at center (jaw line)
                0f                                     // Z: flat
            };
        }
        // Point 16 is chin center - critical for TRACKED_POINTS
        landmarkTemplate[16] = new float[]{0f, 40f, 0f};  // Chin center
        
        // ===== LEFT EYEBROW (33-41): 9 points =====
        // Points 33 (outer) and 38 (inner) are in TRACKED_POINTS
        // HotGaze: right eyebrow at -25 + t*15 for x, y=-25
        for (int i = 33; i <= 41; i++) {
            float t = (i - 37) / 4f;  // -1 to 1, centered on 37
            landmarkTemplate[i] = new float[]{
                -25f + t * 15f,   // X: -40 to -10 (left side)
                -25f,             // Y: eyebrow level
                -5f               // Z: slightly forward
            };
        }
        
        // ===== RIGHT EYEBROW (42-50): 9 points =====
        // Points 46 (middle) and 50 (outer) are in TRACKED_POINTS
        // HotGaze: left eyebrow at 10 + t*15 for x, y=-25
        for (int i = 42; i <= 50; i++) {
            float t = (i - 46) / 4f;  // -1 to 1, centered on 46
            landmarkTemplate[i] = new float[]{
                25f + t * 15f,    // X: 10 to 40 (right side)
                -25f,             // Y: eyebrow level
                -5f               // Z: slightly forward
            };
        }
        
        // ===== NOSE BRIDGE (51-54): 4 points =====
        for (int i = 51; i <= 54; i++) {
            float t = (i - 51) / 3f;  // 0 to 1
            landmarkTemplate[i] = new float[]{
                0f,                    // X: center
                -15f + t * 25f,        // Y: -15 to 10
                -10f - t * 5f          // Z: nose protrudes forward
            };
        }
        
        // ===== NOSE BOTTOM (55-59): 5 points =====
        // Points 55 and 59 are in TRACKED_POINTS
        // HotGaze: nose base at y=15, x from -10 to 10
        landmarkTemplate[55] = new float[]{-10f, 15f, -10f};  // Left nostril outer (TRACKED)
        landmarkTemplate[56] = new float[]{-5f, 15f, -12f};   // Left nostril
        landmarkTemplate[57] = new float[]{0f, 15f, -15f};    // Nose tip
        landmarkTemplate[58] = new float[]{5f, 15f, -12f};    // Right nostril
        landmarkTemplate[59] = new float[]{10f, 15f, -10f};   // Right nostril outer (TRACKED)
        
        // ===== LEFT EYE (60-67): 8 points =====
        // Points 60 (outer corner) and 64 (inner corner) are in TRACKED_POINTS
        // HotGaze: right eye at x=-30 to -15, y=-10
        landmarkTemplate[60] = new float[]{-30f, -10f, -5f};  // Outer corner (TRACKED)
        landmarkTemplate[61] = new float[]{-25f, -13f, -5f};  // Upper outer
        landmarkTemplate[62] = new float[]{-20f, -13f, -5f};  // Upper inner
        landmarkTemplate[63] = new float[]{-15f, -10f, -5f};  // Inner corner
        landmarkTemplate[64] = new float[]{-15f, -10f, -5f};  // Inner corner (TRACKED) - same as 63
        landmarkTemplate[65] = new float[]{-20f, -7f, -5f};   // Lower inner
        landmarkTemplate[66] = new float[]{-25f, -7f, -5f};   // Lower outer
        landmarkTemplate[67] = new float[]{-30f, -7f, -5f};   // Lower outer edge
        
        // ===== RIGHT EYE (68-75): 8 points =====
        // Points 68 (inner corner) and 72 (outer corner) are in TRACKED_POINTS
        // HotGaze: left eye at x=15 to 30, y=-10
        landmarkTemplate[68] = new float[]{15f, -10f, -5f};   // Inner corner (TRACKED)
        landmarkTemplate[69] = new float[]{20f, -13f, -5f};   // Upper inner
        landmarkTemplate[70] = new float[]{25f, -13f, -5f};   // Upper outer
        landmarkTemplate[71] = new float[]{30f, -10f, -5f};   // Outer corner
        landmarkTemplate[72] = new float[]{30f, -10f, -5f};   // Outer corner (TRACKED) - same as 71
        landmarkTemplate[73] = new float[]{25f, -7f, -5f};    // Lower outer
        landmarkTemplate[74] = new float[]{20f, -7f, -5f};    // Lower inner
        landmarkTemplate[75] = new float[]{15f, -7f, -5f};    // Lower inner edge
        
        // ===== OUTER LIPS (76-87): 12 points =====
        // Points 76, 82, 85 are in TRACKED_POINTS
        // WFLW format: 76=left corner, 79=top center, 82=right corner, 85=bottom center
        // Lips centered at y=25, width ~40, height ~16
        landmarkTemplate[76] = new float[]{-20f, 25f, -5f};   // Left corner (TRACKED)
        landmarkTemplate[77] = new float[]{-15f, 21f, -6f};   // Upper left outer
        landmarkTemplate[78] = new float[]{-7f, 19f, -7f};    // Upper left
        landmarkTemplate[79] = new float[]{0f, 18f, -8f};     // Upper center (top)
        landmarkTemplate[80] = new float[]{7f, 19f, -7f};     // Upper right
        landmarkTemplate[81] = new float[]{15f, 21f, -6f};    // Upper right outer
        landmarkTemplate[82] = new float[]{20f, 25f, -5f};    // Right corner (TRACKED)
        landmarkTemplate[83] = new float[]{15f, 29f, -6f};    // Lower right outer
        landmarkTemplate[84] = new float[]{7f, 32f, -7f};     // Lower right
        landmarkTemplate[85] = new float[]{0f, 33f, -8f};     // Lower center (TRACKED)
        landmarkTemplate[86] = new float[]{-7f, 32f, -7f};    // Lower left
        landmarkTemplate[87] = new float[]{-15f, 29f, -6f};   // Lower left outer
        
        // ===== INNER LIPS (88-95): 8 points =====
        // Inner lips follow same pattern but smaller
        landmarkTemplate[88] = new float[]{-12f, 25f, -4f};   // Left inner corner
        landmarkTemplate[89] = new float[]{-6f, 22f, -5f};    // Upper left inner
        landmarkTemplate[90] = new float[]{0f, 21f, -6f};     // Upper center inner
        landmarkTemplate[91] = new float[]{6f, 22f, -5f};     // Upper right inner
        landmarkTemplate[92] = new float[]{12f, 25f, -4f};    // Right inner corner
        landmarkTemplate[93] = new float[]{6f, 28f, -5f};     // Lower right inner
        landmarkTemplate[94] = new float[]{0f, 29f, -6f};     // Lower center inner
        landmarkTemplate[95] = new float[]{-6f, 28f, -5f};    // Lower left inner
        
        // ===== EYE PUPILS (96-97) - Most important for gaze estimation =====
        // Center of each eye (between inner and outer corners)
        landmarkTemplate[96] = new float[]{-22.5f, -10f, -8f};  // Left eye pupil center
        landmarkTemplate[97] = new float[]{22.5f, -10f, -8f};   // Right eye pupil center
    }
    
    /**
     * Get the accelerator type being used
     */
    public String getAcceleratorType() {
        return acceleratorType;
    }
    
    /**
     * Check if initialized
     */
    public boolean isInitialized() {
        return isInitialized;
    }
    
    /**
     * Get model input dimensions
     */
    public int getInputWidth() {
        return INPUT_WIDTH;
    }
    
    public int getInputHeight() {
        return INPUT_HEIGHT;
    }
    
    /**
     * Close the detector and release resources
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        isInitialized = false;
        Log.i(TAG, "TFLite Landmark Detector V2 closed");
    }
}

