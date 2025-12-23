package org.tensorflow.lite.examples.gaze_estimation;

import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

/**
 * Logger for landmark model input and output
 * Saves:
 * - Input images (112x112 RGB) as PNG files
 * - Output landmarks as CSV file
 */
public class LandmarkLogger {
    private static final String TAG = "LandmarkLogger";
    
    private File outputDir;
    private String timestamp;
    private int frameCount = 0;
    
    private BufferedWriter landmarkOutputWriter = null;
    private File landmarkOutputFile = null;
    private File inputImagesDir = null;
    
    private DecimalFormat df;
    
    public LandmarkLogger(File outputDir, String timestamp) throws IOException {
        this.outputDir = outputDir;
        this.timestamp = timestamp;
        
        // Create subdirectory for this logging session
        File sessionDir = new File(outputDir, "session_" + timestamp);
        if (!sessionDir.exists()) {
            sessionDir.mkdirs();
        }
        
        // Create directory for input images
        inputImagesDir = new File(sessionDir, "input_images");
        if (!inputImagesDir.exists()) {
            inputImagesDir.mkdirs();
        }
        
        // Create CSV file for landmark outputs
        landmarkOutputFile = new File(sessionDir, "landmark_outputs.csv");
        landmarkOutputWriter = new BufferedWriter(new FileWriter(landmarkOutputFile));
        
        // Write CSV header
        landmarkOutputWriter.write("frame,timestamp_ms");
        for (int i = 0; i < 136; i++) { // Assuming 136 landmark values (68 landmarks * 2)
            landmarkOutputWriter.write(",landmark_" + i);
        }
        landmarkOutputWriter.write("\n");
        landmarkOutputWriter.flush();
        
        // Decimal formatter for consistent float output
        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
        df = new DecimalFormat("#.######", symbols);
        
        Log.i(TAG, "LandmarkLogger initialized. Output directory: " + sessionDir.getAbsolutePath());
    }
    
    /**
     * Log a frame's input image and output landmarks
     * @param inputImage Float array of 112x112x3 RGB values (normalized 0-1)
     * @param outputLandmarks Float array of landmark output values
     * @param timestampMs Time since logging started in milliseconds
     */
    public void logFrame(float[] inputImage, float[] outputLandmarks, long timestampMs) {
        try {
            frameCount++;
            
            // Save input image as PNG
            saveInputImage(inputImage, frameCount);
            
            // Save landmark output to CSV
            saveLandmarkOutput(outputLandmarks, frameCount, timestampMs);
            
        } catch (Exception e) {
            Log.e(TAG, "Error logging frame " + frameCount + ": " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Convert float array (112x112x3, normalized 0-1) to Bitmap and save as PNG
     */
    private void saveInputImage(float[] inputImage, int frameNumber) throws IOException {
        // Create bitmap from float array
        int width = 112;
        int height = 112;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        
        int[] pixels = new int[width * height];
        for (int i = 0; i < pixels.length; i++) {
            float r = inputImage[i * 3] * 255.0f;
            float g = inputImage[i * 3 + 1] * 255.0f;
            float b = inputImage[i * 3 + 2] * 255.0f;
            
            int rInt = Math.max(0, Math.min(255, (int)r));
            int gInt = Math.max(0, Math.min(255, (int)g));
            int bInt = Math.max(0, Math.min(255, (int)b));
            
            pixels[i] = 0xFF000000 | (rInt << 16) | (gInt << 8) | bInt;
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        
        // Save as PNG
        String filename = String.format(Locale.US, "input_frame_%04d.png", frameNumber);
        File imageFile = new File(inputImagesDir, filename);
        
        java.io.FileOutputStream out = new java.io.FileOutputStream(imageFile);
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        out.close();
    }
    
    /**
     * Save landmark output to CSV
     */
    private void saveLandmarkOutput(float[] landmarks, int frameNumber, long timestampMs) throws IOException {
        landmarkOutputWriter.write(String.valueOf(frameNumber));
        landmarkOutputWriter.write(",");
        landmarkOutputWriter.write(String.valueOf(timestampMs));
        
        for (float landmark : landmarks) {
            landmarkOutputWriter.write(",");
            landmarkOutputWriter.write(df.format(landmark));
        }
        
        landmarkOutputWriter.write("\n");
        landmarkOutputWriter.flush();
    }
    
    /**
     * Close all file writers
     */
    public void close() {
        try {
            if (landmarkOutputWriter != null) {
                landmarkOutputWriter.close();
                landmarkOutputWriter = null;
            }
            Log.i(TAG, "Logged " + frameCount + " frames. Files saved to: " + outputDir.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "Error closing logger: " + e.getMessage());
        }
    }
    
    /**
     * Get the path where logs are saved
     */
    public String getLogPath() {
        if (outputDir != null) {
            return outputDir.getAbsolutePath() + "/session_" + timestamp;
        }
        return "Unknown";
    }
}

