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
import java.util.HashMap;
import java.util.Map;

/**
 * TensorFlow Lite based Gaze Estimator.
 * 
 * Inputs:
 *   - face: (120×120×3) - RGB float normalized
 *   - left_eye: (60×60×3) - RGB float normalized
 *   - right_eye: (60×60×3) - RGB float normalized
 * 
 * Model: gaze_estimation.tflite
 * Output: gaze_pitchyaw (2 values: pitch, yaw)
 */
public class TFLiteGazeEstimator {
    private static final String TAG = "TFLiteGaze";
    
    private Interpreter interpreter;
    private NnApiDelegate nnApiDelegate;
    private String acceleratorType = "CPU";
    
    // Model parameters - matching SNPE config
    public static final int FACE_HEIGHT = 120;
    public static final int FACE_WIDTH = 120;
    public static final int EYE_HEIGHT = 60;
    public static final int EYE_WIDTH = 60;
    public static final int CHANNELS = 3;
    public static final int OUTPUT_SIZE = 2;  // pitch, yaw
    
    private final Context context;
    private boolean isInitialized = false;
    
    // Reusable buffers
    private ByteBuffer faceBuffer;
    private ByteBuffer leftEyeBuffer;
    private ByteBuffer rightEyeBuffer;
    private float[][] outputArray;
    
    // Input tensor indices (will be determined at runtime)
    private int faceInputIndex = -1;
    private int leftEyeInputIndex = -1;
    private int rightEyeInputIndex = -1;
    
    public TFLiteGazeEstimator(Context context) {
        this.context = context;
    }
    
    /**
     * Initialize the TFLite gaze estimator with NNAPI acceleration
     */
    public void initialize() throws IOException {
        Log.i(TAG, "========================================");
        Log.i(TAG, "Initializing TFLite Gaze Estimation");
        Log.i(TAG, "========================================");
        
        // Load gaze estimation model
        Log.i(TAG, "Loading Gaze Estimation model...");
        MappedByteBuffer modelBuffer = AssetUtils.loadMappedFile(context, "gaze_estimation.tflite");
        
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
        
        // Detect input tensor indices by inspecting tensor names/shapes
        detectInputIndices();
        
        // Pre-allocate buffers
        int faceSize = FACE_HEIGHT * FACE_WIDTH * CHANNELS * 4;  // float32
        int eyeSize = EYE_HEIGHT * EYE_WIDTH * CHANNELS * 4;  // float32
        
        faceBuffer = ByteBuffer.allocateDirect(faceSize);
        faceBuffer.order(ByteOrder.nativeOrder());
        
        leftEyeBuffer = ByteBuffer.allocateDirect(eyeSize);
        leftEyeBuffer.order(ByteOrder.nativeOrder());
        
        rightEyeBuffer = ByteBuffer.allocateDirect(eyeSize);
        rightEyeBuffer.order(ByteOrder.nativeOrder());
        
        outputArray = new float[1][OUTPUT_SIZE];
        
        isInitialized = true;
        
        Log.i(TAG, "════════════════════════════════════════");
        Log.i(TAG, "✓ Gaze Estimation Ready");
        Log.i(TAG, "  Face input: " + FACE_WIDTH + "x" + FACE_HEIGHT + "x" + CHANNELS);
        Log.i(TAG, "  Eye input: " + EYE_WIDTH + "x" + EYE_HEIGHT + "x" + CHANNELS);
        Log.i(TAG, "  Output size: " + OUTPUT_SIZE);
        Log.i(TAG, "  Accelerator: " + acceleratorType);
        Log.i(TAG, "════════════════════════════════════════");
    }
    
    /**
     * Detect input tensor indices by inspecting model structure
     */
    private void detectInputIndices() {
        int inputCount = interpreter.getInputTensorCount();
        Log.i(TAG, "Model has " + inputCount + " inputs");
        
        for (int i = 0; i < inputCount; i++) {
            int[] shape = interpreter.getInputTensor(i).shape();
            String shapeStr = shapeToString(shape);
            Log.i(TAG, "  Input " + i + ": shape=" + shapeStr);
            
            // Identify by shape: face is 120x120, eyes are 60x60
            if (shape.length >= 3) {
                int h = shape.length == 4 ? shape[1] : shape[0];
                int w = shape.length == 4 ? shape[2] : shape[1];
                
                if (h == FACE_HEIGHT && w == FACE_WIDTH) {
                    faceInputIndex = i;
                    Log.i(TAG, "    → Identified as FACE input");
                } else if (h == EYE_HEIGHT && w == EYE_WIDTH) {
                    // Assign to left_eye first, then right_eye
                    if (leftEyeInputIndex == -1) {
                        leftEyeInputIndex = i;
                        Log.i(TAG, "    → Identified as LEFT_EYE input");
                    } else if (rightEyeInputIndex == -1) {
                        rightEyeInputIndex = i;
                        Log.i(TAG, "    → Identified as RIGHT_EYE input");
                    }
                }
            }
        }
        
        // Fallback: assume order [left_eye, right_eye, face] as in SNPE
        if (faceInputIndex == -1 || leftEyeInputIndex == -1 || rightEyeInputIndex == -1) {
            Log.w(TAG, "Could not auto-detect all input indices, using default order");
            leftEyeInputIndex = 0;
            rightEyeInputIndex = 1;
            faceInputIndex = 2;
        }
        
        Log.i(TAG, "Input mapping: face=" + faceInputIndex + 
                   ", left_eye=" + leftEyeInputIndex + 
                   ", right_eye=" + rightEyeInputIndex);
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
     * Run gaze estimation on preprocessed inputs
     * @param face Float array from gaze_preprocess.face (120*120*3)
     * @param leftEye Float array from gaze_preprocess.leye (60*60*3)
     * @param rightEye Float array from gaze_preprocess.reye (60*60*3)
     * @return Gaze output array (2 values: pitch, yaw) or null on error
     */
    public float[] estimate(float[] face, float[] leftEye, float[] rightEye) {
        if (!isInitialized || interpreter == null) {
            Log.e(TAG, "Gaze estimator not initialized!");
            return null;
        }
        
        int expectedFaceSize = FACE_HEIGHT * FACE_WIDTH * CHANNELS;
        int expectedEyeSize = EYE_HEIGHT * EYE_WIDTH * CHANNELS;
        
        if (face == null || face.length != expectedFaceSize) {
            Log.e(TAG, "Invalid face input size: " + (face != null ? face.length : "null") + 
                       ", expected: " + expectedFaceSize);
            return null;
        }
        
        if (leftEye == null || leftEye.length != expectedEyeSize) {
            Log.e(TAG, "Invalid leftEye input size: " + (leftEye != null ? leftEye.length : "null") + 
                       ", expected: " + expectedEyeSize);
            return null;
        }
        
        if (rightEye == null || rightEye.length != expectedEyeSize) {
            Log.e(TAG, "Invalid rightEye input size: " + (rightEye != null ? rightEye.length : "null") + 
                       ", expected: " + expectedEyeSize);
            return null;
        }
        
        try {
            // Fill input buffers
            faceBuffer.rewind();
            for (float value : face) {
                faceBuffer.putFloat(value);
            }
            faceBuffer.rewind();
            
            leftEyeBuffer.rewind();
            for (float value : leftEye) {
                leftEyeBuffer.putFloat(value);
            }
            leftEyeBuffer.rewind();
            
            rightEyeBuffer.rewind();
            for (float value : rightEye) {
                rightEyeBuffer.putFloat(value);
            }
            rightEyeBuffer.rewind();
            
            // Prepare inputs map
            Object[] inputs = new Object[3];
            inputs[leftEyeInputIndex] = leftEyeBuffer;
            inputs[rightEyeInputIndex] = rightEyeBuffer;
            inputs[faceInputIndex] = faceBuffer;
            
            // Prepare outputs map
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputArray);
            
            // Run inference
            long startTime = System.currentTimeMillis();
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Gaze estimation inference: " + inferenceTime + "ms");
            
            return outputArray[0];
            
        } catch (Exception e) {
            Log.e(TAG, "Gaze estimation error: " + e.getMessage(), e);
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
     * Close the estimator and release resources
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
        Log.i(TAG, "TFLite Gaze Estimator closed");
    }
}
