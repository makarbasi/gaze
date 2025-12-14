package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * TensorFlow Lite based Face Detector using the face_detection.tflite model from HotGaze.
 * 
 * Input:  Bitmap (640×480) → Grayscale ByteBuffer
 * Model:  face_detection.tflite
 * Output: Heatmap (60×80), Bounding Boxes (60×80×4), Landmarks (60×80×10)
 *
 * Processing:
 *   1. Dequantize heatmap [0-255] → [0.0-1.0]
 *   2. Find local maxima (3×3 pooling) above threshold (0.5)
 *   3. Decode bounding boxes using offsets × stride(8)
 *   4. Apply NMS (IoU threshold 0.3)
 *   → Result: List<float[]> face bounding boxes [x1, y1, x2, y2, score]
 */
public class TFLiteFaceDetector {
    private static final String TAG = "TFLiteFaceDetector";
    
    private Interpreter interpreter;
    private NnApiDelegate nnApiDelegate;
    private String acceleratorType = "CPU";
    
    // Face detection model parameters
    private static final int FACE_DET_WIDTH = 640;
    private static final int FACE_DET_HEIGHT = 480;
    
    // Face detection thresholds
    private float faceDetectionThreshold = 0.5f;
    private int minFaceSize = 30;
    private float nmsIouThreshold = 0.3f;
    private boolean singleFaceMode = true;  // Only return the best face
    
    private final Context context;
    private boolean isInitialized = false;
    
    public TFLiteFaceDetector(Context context) {
        this.context = context;
    }
    
    /**
     * Initialize the TFLite face detector with GPU/NNAPI acceleration
     */
    public void initialize() throws IOException {
        Log.i(TAG, "========================================");
        Log.i(TAG, "Initializing TFLite Face Detection");
        Log.i(TAG, "========================================");
        
        // Load face detection model
        Log.i(TAG, "Loading Face Detection model...");
        MappedByteBuffer faceDetModelBuffer = FileUtil.loadMappedFile(context, "face_detection.tflite");
        
        Interpreter.Options options = new Interpreter.Options();
        
        // Try to use NNAPI delegate for GPU acceleration
        Log.i(TAG, "Device: " + Build.MANUFACTURER + " " + Build.MODEL);
        Log.i(TAG, "SOC: " + Build.HARDWARE);
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
        
        interpreter = new Interpreter(faceDetModelBuffer, options);
        isInitialized = true;
        
        Log.i(TAG, "════════════════════════════════════════");
        Log.i(TAG, "✓ Face Detection Ready");
        Log.i(TAG, "  Input size: " + FACE_DET_WIDTH + "x" + FACE_DET_HEIGHT);
        Log.i(TAG, "  Accelerator: " + acceleratorType);
        Log.i(TAG, "════════════════════════════════════════");
    }
    
    /**
     * Detect faces in the given bitmap
     * @param bitmap Input bitmap (will be scaled to 640x480 internally)
     * @return Array of detected face bounding boxes [x1, y1, x2, y2, score]
     */
    public float[][] detectFaces(Bitmap bitmap) {
        if (!isInitialized || interpreter == null) {
            Log.e(TAG, "Face detector not initialized!");
            return new float[0][];
        }
        
        try {
            // Scale bitmap to 640x480
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, FACE_DET_WIDTH, FACE_DET_HEIGHT, true);
            
            // Store scale factors to map back to original coordinates
            float scaleX = (float) bitmap.getWidth() / FACE_DET_WIDTH;
            float scaleY = (float) bitmap.getHeight() / FACE_DET_HEIGHT;
            
            // Create input buffer - GRAYSCALE (1 channel), UINT8 quantized
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(FACE_DET_WIDTH * FACE_DET_HEIGHT);
            inputBuffer.order(ByteOrder.nativeOrder());
            
            int[] intValues = new int[FACE_DET_WIDTH * FACE_DET_HEIGHT];
            resizedBitmap.getPixels(intValues, 0, FACE_DET_WIDTH, 0, 0, FACE_DET_WIDTH, FACE_DET_HEIGHT);
            
            // Convert RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
            for (int pixelValue : intValues) {
                int r = (pixelValue >> 16) & 0xFF;
                int g = (pixelValue >> 8) & 0xFF;
                int b = pixelValue & 0xFF;
                byte gray = (byte) (0.299f * r + 0.587f * g + 0.114f * b);
                inputBuffer.put(gray);
            }
            inputBuffer.rewind();
            
            // Query output tensor shapes
            int outputCount = interpreter.getOutputTensorCount();
            Log.d(TAG, "Model has " + outputCount + " outputs");
            
            // Get output tensor shapes
            int[] shape0 = interpreter.getOutputTensor(0).shape();
            int[] shape1 = interpreter.getOutputTensor(1).shape();
            int[] shape2 = interpreter.getOutputTensor(2).shape();
            
            // Allocate output arrays
            byte[][][][] outputArray0 = new byte[shape0[0]][shape0[1]][shape0[2]][shape0[3]];
            byte[][][][] outputArray1 = new byte[shape1[0]][shape1[1]][shape1[2]][shape1[3]];
            byte[][][][] outputArray2 = new byte[shape2[0]][shape2[1]][shape2[2]][shape2[3]];
            
            Object[] outputs = new Object[3];
            outputs[0] = outputArray0;
            outputs[1] = outputArray1;
            outputs[2] = outputArray2;
            
            // Run inference
            long startTime = System.currentTimeMillis();
            interpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer}, 
                new java.util.HashMap<Integer, Object>() {{
                    put(0, outputArray0);
                    put(1, outputArray1);
                    put(2, outputArray2);
                }});
            long inferenceTime = System.currentTimeMillis() - startTime;
            Log.d(TAG, "Face detection inference: " + inferenceTime + "ms");
            
            // Identify which output is which based on channel count
            byte[][][][] heatmapOutput;
            byte[][][][] bboxOutput;
            
            if (shape0[3] == 1) {
                heatmapOutput = outputArray0;
                bboxOutput = (shape1[3] == 4) ? outputArray1 : outputArray2;
            } else if (shape1[3] == 1) {
                heatmapOutput = outputArray1;
                bboxOutput = (shape0[3] == 4) ? outputArray0 : outputArray2;
            } else {
                heatmapOutput = outputArray2;
                bboxOutput = (shape0[3] == 4) ? outputArray0 : outputArray1;
            }
            
            // Parse face detection results
            List<float[]> faceBoxes = parseFaceDetectionOutputs(heatmapOutput, bboxOutput, 
                FACE_DET_WIDTH, FACE_DET_HEIGHT, scaleX, scaleY);
            
            // Convert to array
            return faceBoxes.toArray(new float[0][]);
            
        } catch (Exception e) {
            Log.e(TAG, "Face detection error: " + e.getMessage(), e);
            return new float[0][];
        }
    }
    
    private List<float[]> parseFaceDetectionOutputs(
            byte[][][][] heatmapOutput,
            byte[][][][] bboxOutput,
            int imageWidth,
            int imageHeight,
            float scaleX,
            float scaleY) {
        
        List<float[]> faces = new ArrayList<>();
        
        try {
            int heatmapHeight = 60;
            int heatmapWidth = 80;
            int stride = 8;
            
            // Step 1: Dequantize heatmap
            float[][] heatmapFloat = new float[heatmapHeight][heatmapWidth];
            
            for (int y = 0; y < heatmapHeight; y++) {
                for (int x = 0; x < heatmapWidth; x++) {
                    int raw = heatmapOutput[0][y][x][0] & 0xFF;
                    heatmapFloat[y][x] = raw / 255.0f;
                }
            }
            
            // Step 2: Find local maxima using 3x3 max pooling
            List<float[]> candidates = new ArrayList<>();  // [x, y, score]
            
            for (int cy = 1; cy < heatmapHeight - 1; cy++) {
                for (int cx = 1; cx < heatmapWidth - 1; cx++) {
                    float centerValue = heatmapFloat[cy][cx];
                    
                    if (centerValue < faceDetectionThreshold) continue;
                    
                    // Check if local maximum
                    boolean isLocalMax = true;
                    outerLoop:
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dy == 0 && dx == 0) continue;
                            if (heatmapFloat[cy + dy][cx + dx] > centerValue) {
                                isLocalMax = false;
                                break outerLoop;
                            }
                        }
                    }
                    
                    if (isLocalMax) {
                        candidates.add(new float[]{cx, cy, centerValue});
                    }
                }
            }
            
            // Sort by score descending
            Collections.sort(candidates, (a, b) -> Float.compare(b[2], a[2]));
            
            // Take top 10 candidates
            List<float[]> topCandidates = candidates.subList(0, Math.min(10, candidates.size()));
            
            Log.i(TAG, "Found " + topCandidates.size() + " face candidates (threshold=" + faceDetectionThreshold + ")");
            
            // Step 3: Decode bounding boxes
            List<float[]> rawBoxes = new ArrayList<>();  // [x1, y1, x2, y2, score]
            
            // Get quantization parameters
            float bboxScale = interpreter.getOutputTensor(1).quantizationParams().getScale();
            int bboxZeroPoint = interpreter.getOutputTensor(1).quantizationParams().getZeroPoint();
            
            for (float[] candidate : topCandidates) {
                int cx = (int) candidate[0];
                int cy = (int) candidate[1];
                float score = candidate[2];
                
                // Get bbox offsets
                byte[] bboxData = bboxOutput[0][cy][cx];
                
                // Dequantize
                float xOff = ((bboxData[0] & 0xFF) - bboxZeroPoint) * bboxScale;
                float yOff = ((bboxData[1] & 0xFF) - bboxZeroPoint) * bboxScale;
                float rOff = ((bboxData[2] & 0xFF) - bboxZeroPoint) * bboxScale;
                float bOff = ((bboxData[3] & 0xFF) - bboxZeroPoint) * bboxScale;
                
                // Convert to absolute coordinates
                int left = Math.max(0, Math.min(imageWidth, (int) ((cx - xOff) * stride)));
                int top = Math.max(0, Math.min(imageHeight, (int) ((cy - yOff) * stride)));
                int right = Math.max(0, Math.min(imageWidth, (int) ((cx + rOff) * stride)));
                int bottom = Math.max(0, Math.min(imageHeight, (int) ((cy + bOff) * stride)));
                
                int width = right - left;
                int height = bottom - top;
                
                // Validate bounding box
                if (right > left && bottom > top &&
                    left > 5 && top > 5 &&
                    width > minFaceSize && height > minFaceSize &&
                    width < imageWidth * 0.9f && height < imageHeight * 0.9f) {
                    
                    // Scale back to original image coordinates
                    float[] box = new float[]{
                        left * scaleX,
                        top * scaleY,
                        right * scaleX,
                        bottom * scaleY,
                        score
                    };
                    rawBoxes.add(box);
                }
            }
            
            // Step 4: Apply NMS
            List<float[]> nmsResults = applyNMS(rawBoxes, nmsIouThreshold);
            
            // In single face mode, only return the best face (highest confidence)
            if (singleFaceMode && nmsResults.size() > 1) {
                faces.add(nmsResults.get(0));  // Already sorted by score descending
                Log.i(TAG, "Single face mode: returning best face (score=" + nmsResults.get(0)[4] + ") out of " + nmsResults.size());
            } else {
                faces.addAll(nmsResults);
            }
            
            Log.i(TAG, "Detected " + faces.size() + " face(s) after NMS");
            
        } catch (Exception e) {
            Log.e(TAG, "Error parsing face detection outputs: " + e.getMessage(), e);
        }
        
        return faces;
    }
    
    private List<float[]> applyNMS(List<float[]> boxes, float iouThreshold) {
        if (boxes.isEmpty()) return new ArrayList<>();
        
        // Sort by score descending
        Collections.sort(boxes, (a, b) -> Float.compare(b[4], a[4]));
        
        List<float[]> selectedBoxes = new ArrayList<>();
        
        for (float[] box : boxes) {
            boolean shouldSelect = true;
            
            for (float[] selectedBox : selectedBoxes) {
                float iou = calculateIoU(box, selectedBox);
                if (iou > iouThreshold) {
                    shouldSelect = false;
                    break;
                }
            }
            
            if (shouldSelect) {
                selectedBoxes.add(box);
            }
        }
        
        return selectedBoxes;
    }
    
    private float calculateIoU(float[] box1, float[] box2) {
        float x1 = Math.max(box1[0], box2[0]);
        float y1 = Math.max(box1[1], box2[1]);
        float x2 = Math.min(box1[2], box2[2]);
        float y2 = Math.min(box1[3], box2[3]);
        
        if (x2 < x1 || y2 < y1) return 0f;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        float union = area1 + area2 - intersection;
        
        return union > 0 ? intersection / union : 0f;
    }
    
    /**
     * Get the accelerator type being used
     */
    public String getAcceleratorType() {
        return acceleratorType;
    }
    
    /**
     * Set face detection threshold (0.0 to 1.0)
     */
    public void setFaceDetectionThreshold(float threshold) {
        this.faceDetectionThreshold = Math.max(0f, Math.min(1f, threshold));
    }
    
    /**
     * Set minimum face size in pixels
     */
    public void setMinFaceSize(int size) {
        this.minFaceSize = Math.max(10, Math.min(200, size));
    }
    
    /**
     * Set NMS IoU threshold
     */
    public void setNmsIouThreshold(float threshold) {
        this.nmsIouThreshold = Math.max(0f, Math.min(1f, threshold));
    }
    
    /**
     * Set single face mode - only return the best face when enabled
     */
    public void setSingleFaceMode(boolean enabled) {
        this.singleFaceMode = enabled;
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
        Log.i(TAG, "TFLite Face Detector closed");
    }
}

