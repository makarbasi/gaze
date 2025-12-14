package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.util.Log;

import java.io.Closeable;
import java.util.HashMap;
import java.util.Map;

/**
 * QNN Model Runner - Java interface for running DLC models using Qualcomm Neural Network (QNN)
 * 
 * This class provides a high-level Java API for:
 * - Loading DLC (Deep Learning Container) files
 * - Running inference with single or multiple inputs
 * - Managing model lifecycle
 * 
 * Usage:
 *   QNNModelRunner runner = new QNNModelRunner();
 *   runner.loadModel("/path/to/model.dlc", "output_layer_name", QNNModelRunner.Backend.HTP);
 *   float[] output = runner.runInference("input", inputData, new int[]{1, 112, 112, 3});
 *   runner.close();
 * 
 * Required QNN SDK libraries (place in app/src/main/jniLibs/arm64-v8a/):
 * - libQnnHtp.so (or libQnnGpu.so, libQnnDsp.so)
 * - libQnnModelDlc.so
 * - libQnnSystem.so
 * - HTP skel files (libQnnHtpV*Skel.so, libQnnHtpV*Stub.so)
 */
public class QNNModelRunner implements Closeable {
    private static final String TAG = "QNNModelRunner";
    
    // Backend types
    public enum Backend {
        HTP(0),   // Hexagon Tensor Processor (recommended for Snapdragon)
        GPU(1),   // Adreno GPU
        DSP(2),   // Hexagon DSP (legacy)
        CPU(3);   // CPU fallback
        
        private final int value;
        Backend(int value) { this.value = value; }
        public int getValue() { return value; }
    }
    
    // Load native library
    static {
        try {
            System.loadLibrary("qnn_model_runner");
            Log.i(TAG, "✓ QNN Model Runner native library loaded");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load QNN Model Runner native library: " + e.getMessage());
        }
    }
    
    private long modelHandle = -1;
    private boolean isInitialized = false;
    private String modelPath;
    private Backend backend;
    
    /**
     * Check if QNN is available on this device
     */
    public static boolean isQnnAvailable() {
        try {
            return nativeIsQnnAvailable();
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "QNN availability check failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Load a DLC model file
     * 
     * @param modelPath Path to the DLC file (e.g., "/data/local/tmp/model.dlc")
     * @param outputLayerName Name of the output layer
     * @param backend Backend to use for inference
     * @return true if successful
     */
    public boolean loadModel(String modelPath, String outputLayerName, Backend backend) {
        if (isInitialized) {
            Log.w(TAG, "Model already loaded. Call close() first.");
            return false;
        }
        
        this.modelPath = modelPath;
        this.backend = backend;
        
        try {
            modelHandle = nativeLoadModel(modelPath, outputLayerName, backend.getValue());
            if (modelHandle >= 0) {
                isInitialized = true;
                Log.i(TAG, "✓ Model loaded: " + modelPath);
                Log.i(TAG, "  Backend: " + getBackendInfo());
                return true;
            } else {
                Log.e(TAG, "Failed to load model: " + modelPath);
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading model: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Run inference with a single input
     * 
     * @param inputName Name of the input tensor
     * @param inputData Float array of input data (NHWC format)
     * @param inputDims Dimensions of the input (e.g., {1, 112, 112, 3})
     * @return Float array of output data, or null on error
     */
    public float[] runInference(String inputName, float[] inputData, int[] inputDims) {
        if (!isInitialized) {
            Log.e(TAG, "Model not loaded. Call loadModel() first.");
            return null;
        }
        
        try {
            return nativeRunInference(modelHandle, inputName, inputData, inputDims);
        } catch (Exception e) {
            Log.e(TAG, "Inference error: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Run inference with multiple inputs (e.g., for gaze estimation with face + eyes)
     * 
     * @param inputNames Array of input tensor names
     * @param inputDataArrays Array of input data arrays
     * @param inputDimsArrays Array of dimension arrays
     * @return Float array of output data, or null on error
     */
    public float[] runMultiInputInference(String[] inputNames, float[][] inputDataArrays, int[][] inputDimsArrays) {
        if (!isInitialized) {
            Log.e(TAG, "Model not loaded. Call loadModel() first.");
            return null;
        }
        
        if (inputNames.length != inputDataArrays.length || inputNames.length != inputDimsArrays.length) {
            Log.e(TAG, "Input array lengths must match");
            return null;
        }
        
        try {
            return nativeRunMultiInputInference(modelHandle, inputNames, inputDataArrays, inputDimsArrays);
        } catch (Exception e) {
            Log.e(TAG, "Multi-input inference error: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Get information about the backend being used
     */
    public String getBackendInfo() {
        if (!isInitialized) {
            return "Not initialized";
        }
        try {
            return nativeGetBackendInfo(modelHandle);
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }
    
    /**
     * Check if the model is loaded and ready
     */
    public boolean isReady() {
        return isInitialized && modelHandle >= 0;
    }
    
    /**
     * Get the model path
     */
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * Get the backend type
     */
    public Backend getBackend() {
        return backend;
    }
    
    /**
     * Release resources
     */
    @Override
    public void close() {
        if (isInitialized && modelHandle >= 0) {
            try {
                nativeReleaseModel(modelHandle);
                Log.i(TAG, "✓ Model released: " + modelPath);
            } catch (Exception e) {
                Log.e(TAG, "Error releasing model: " + e.getMessage());
            }
            modelHandle = -1;
            isInitialized = false;
        }
    }
    
    // ========== Native Methods ==========
    
    private static native boolean nativeIsQnnAvailable();
    
    private native long nativeLoadModel(String modelPath, String outputLayerName, int backendType);
    
    private native float[] nativeRunInference(long handle, String inputName, float[] inputData, int[] inputDims);
    
    private native float[] nativeRunMultiInputInference(long handle, String[] inputNames, 
                                                         float[][] inputDataArrays, int[][] inputDimsArrays);
    
    private native void nativeReleaseModel(long handle);
    
    private native String nativeGetBackendInfo(long handle);
}

