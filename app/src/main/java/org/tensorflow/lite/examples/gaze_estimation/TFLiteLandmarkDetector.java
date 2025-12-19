package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;

/**
 * TensorFlow Lite based Landmark Detector.
 * 
 * Input:  face_image (112×112×3) - RGB float normalized
 * Model:  landmark_detection.tflite
 * Output: facial_landmark (136 values = 68 landmarks × 2 coords)
 */
public class TFLiteLandmarkDetector {
    private static final String TAG = "TFLiteLandmark";
    
    private Interpreter interpreter;
    private NnApiDelegate nnApiDelegate;
    private String acceleratorType = "CPU";
    
    // Model parameters - matching SNPE config
    public static final int INPUT_HEIGHT = 112;
    public static final int INPUT_WIDTH = 112;
    public static final int INPUT_CHANNELS = 3;
    public static final int OUTPUT_SIZE = 136;  // 68 landmarks × 2 (x, y)
    
    private final Context context;
    private boolean isInitialized = false;
    
    // Reusable buffers
    private ByteBuffer inputBuffer;
    private float[][] outputArray;
    
    public TFLiteLandmarkDetector(Context context) {
        this.context = context;
    }
    
    /**
     * Initialize the TFLite landmark detector with NNAPI acceleration
     */
    public void initialize() throws IOException {
        Log.i(TAG, "========================================");
        Log.i(TAG, "Initializing TFLite Landmark Detection");
        Log.i(TAG, "========================================");
        
        // Load landmark detection model
        Log.i(TAG, "Loading Landmark Detection model...");
        MappedByteBuffer modelBuffer = AssetUtils.loadMappedFile(context, "landmark_detection.tflite");
        
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
        
        // Pre-allocate buffers
        int inputSize = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * 4;  // float32
        inputBuffer = ByteBuffer.allocateDirect(inputSize);
        inputBuffer.order(ByteOrder.nativeOrder());
        
        outputArray = new float[1][OUTPUT_SIZE];
        
        isInitialized = true;
        
        Log.i(TAG, "════════════════════════════════════════");
        Log.i(TAG, "✓ Landmark Detection Ready");
        Log.i(TAG, "  Input size: " + INPUT_WIDTH + "x" + INPUT_HEIGHT + "x" + INPUT_CHANNELS);
        Log.i(TAG, "  Output size: " + OUTPUT_SIZE);
        Log.i(TAG, "  Accelerator: " + acceleratorType);
        Log.i(TAG, "════════════════════════════════════════");
    }
    
    /**
     * Run landmark detection on preprocessed input
     * @param input Float array from landmark_preprocess (already normalized, size: 112*112*3)
     * @return Landmark output array (136 values)
     */
    public float[] detect(float[] input) {
        if (!isInitialized || interpreter == null) {
            Log.e(TAG, "Landmark detector not initialized!");
            return null;
        }
        
        if (input == null || input.length != INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS) {
            Log.e(TAG, "Invalid input size: " + (input != null ? input.length : "null") + 
                       ", expected: " + (INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS));
            return null;
        }
        
        try {
            // Fill input buffer
            inputBuffer.rewind();
            for (float value : input) {
                inputBuffer.putFloat(value);
            }
            inputBuffer.rewind();
            
            // Run inference
            long startTime = System.currentTimeMillis();
            interpreter.run(inputBuffer, outputArray);
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Landmark detection inference: " + inferenceTime + "ms");
            
            return outputArray[0];
            
        } catch (Exception e) {
            Log.e(TAG, "Landmark detection error: " + e.getMessage(), e);
            return null;
        }
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
        Log.i(TAG, "TFLite Landmark Detector closed");
    }
}
