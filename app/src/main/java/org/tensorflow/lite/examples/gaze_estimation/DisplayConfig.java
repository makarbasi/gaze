package org.tensorflow.lite.examples.gaze_estimation;

import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Display Configuration for camera preview and detection image rotation
 * 
 * Reads settings from:
 *   /data/local/tmp/display_config.json
 * 
 * JSON format:
 * {
 *   "preview_rotation": -90,       // TextureView rotation (main camera preview)
 *   "detection_rotation": 90,      // ImageView rotation (cropped image with landmarks)
 *   "preview_scale_x": 1.0,        // TextureView X scale (use -1 to flip horizontally)
 *   "preview_scale_y": 1.0,        // TextureView Y scale (use -1 to flip vertically)
 *   "detection_scale_x": 1.0,      // ImageView X scale
 *   "detection_scale_y": 1.0,      // ImageView Y scale
 *   "img_orientation": 90,         // Transformation matrix rotation for cropping
 *   
 *   // Fisheye camera settings (for OX05B1S 200° lens)
 *   "fisheye_enabled": true,       // Enable fisheye undistortion
 *   "fisheye_strength": 0.5,       // Undistortion strength (0.0-1.0, higher = more correction)
 *   "fisheye_zoom": 1.5,           // Zoom factor after undistortion (1.0 = no zoom)
 *   
 *   // Face detection tuning (for fisheye/wide-angle)
 *   "face_detection_threshold": 0.3,  // Lower = more sensitive (0.1-0.9)
 *   "min_face_size": 15,              // Minimum face size in pixels (smaller for fisheye)
 *   "face_scale_factor": 1.5          // Scale up detected face box for better landmark detection
 * }
 * 
 * Common rotation values:
 *   0   = no rotation
 *   90  = 90° clockwise
 *   -90 = 90° counter-clockwise (same as 270)
 *   180 = 180° rotation
 */
public class DisplayConfig {
    private static final String TAG = "DisplayConfig";
    public static final String CONFIG_PATH = "/data/local/tmp/display_config.json";
    
    // Default values - Display
    public float previewRotation = -90f;      // Main preview rotation
    public float detectionRotation = 90f;     // Detection/landmark display rotation
    public float previewScaleX = 1.0f;        // Preview horizontal scale (-1 to flip)
    public float previewScaleY = 1.0f;        // Preview vertical scale (-1 to flip)
    public float detectionScaleX = 1.0f;      // Detection horizontal scale
    public float detectionScaleY = 1.0f;      // Detection vertical scale
    public int imgOrientation = 90;           // Crop transformation rotation
    
    // Fisheye camera settings
    public boolean fisheyeEnabled = false;    // Enable fisheye undistortion
    public float fisheyeStrength = 0.5f;      // Undistortion strength (0.0-1.0)
    public float fisheyeZoom = 1.5f;          // Zoom factor after undistortion
    
    // Face detection tuning
    public float faceDetectionThreshold = 0.3f;  // Lower for fisheye (default 0.5)
    public int minFaceSize = 15;                 // Smaller for fisheye (default 30)
    public float faceScaleFactor = 1.5f;         // Scale up face box for landmarks
    
    // Crop region settings (for wide FOV cameras)
    // crop_offset_x: 0.0 = left edge, 0.25 = left half center, 0.5 = center, 0.75 = right half center, 1.0 = right edge
    // crop_offset_y: 0.0 = top, 0.5 = center, 1.0 = bottom
    // crop_scale: 1.0 = use full width, 0.5 = use half width (zoom in 2x)
    public float cropOffsetX = 0.0f;             // Left edge (for driver side)
    public float cropOffsetY = 0.5f;             // Vertical center
    public float cropScale = 0.5f;               // Use half the width (left half only)
    
    // Face crop settings
    public float faceCropPadding = 2.0f;         // Padding around face box (2.0 = 2x the box size)
    public int faceCropDisplaySize = 300;        // Display size for face crop (pixels)
    
    // Gaze detection thresholds (in radians, ~0.17 rad = 10°)
    public float gazePitchThreshold = 0.25f;     // Pitch threshold (~14°) - looking up/down
    public float gazeYawThreshold = 0.35f;       // Yaw threshold (~20°) - looking left/right
    public boolean gazePitchOnly = false;        // If true, only use pitch for detection (ignore yaw)
    public int gazeConsecutiveFrames = 3;        // Consecutive frames needed to change state
    
    // Smoothing parameters for 1-Euro filter (lower values = more smoothing, less reactive)
    // min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag
    // beta: Speed coefficient. Higher = faster response but more jitter
    public double landmarkMinCutoff = 0.01;      // Landmark smoothing cutoff (default 0.01)
    public double landmarkBeta = 0.05;           // Landmark speed coef (LOWERED from 0.1 - less jitter)
    public double gazeMinCutoff = 0.01;          // Gaze smoothing cutoff (default 0.01)
    public double gazeBeta = 0.1;                // Gaze speed coef (LOWERED from 0.8 - much less jitter!)
    public double faceMinCutoff = 0.01;          // Face box smoothing cutoff
    public double faceBeta = 0.05;               // Face box speed coef (LOWERED from 0.1)
    
    // Additional display smoothing (simple exponential moving average)
    // Applied AFTER 1-Euro filter for ultra-stable display values
    public float displaySmoothingAlpha = 0.3f;   // 0.0 = no change, 1.0 = no smoothing (0.3 = heavy smoothing)
    
    // Singleton instance
    private static DisplayConfig instance = null;
    private long lastModified = 0;
    
    public static DisplayConfig getInstance() {
        if (instance == null) {
            instance = new DisplayConfig();
            instance.load();
        }
        return instance;
    }
    
    /**
     * Check if config file has been modified and reload if needed
     */
    public void checkAndReload() {
        File configFile = new File(CONFIG_PATH);
        if (configFile.exists()) {
            long currentModified = configFile.lastModified();
            if (currentModified > lastModified) {
                Log.i(TAG, "Config file modified (was: " + lastModified + ", now: " + currentModified + "), reloading...");
                load();
            }
        } else {
            Log.w(TAG, "Config file does not exist at: " + CONFIG_PATH);
        }
    }
    
    /**
     * Load configuration from JSON file
     * @return true if file was loaded successfully
     */
    public boolean load() {
        File configFile = new File(CONFIG_PATH);
        if (!configFile.exists()) {
            Log.i(TAG, "Config file not found at " + CONFIG_PATH + ", creating with defaults");
            save();
            return false;
        }
        
        lastModified = configFile.lastModified();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            StringBuilder jsonBuilder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonBuilder.append(line);
            }
            
            String jsonString = jsonBuilder.toString();
            Log.i(TAG, "Raw JSON content: " + jsonString);
            
            JSONObject json = new JSONObject(jsonString);
            
            previewRotation = (float) json.optDouble("preview_rotation", previewRotation);
            detectionRotation = (float) json.optDouble("detection_rotation", detectionRotation);
            previewScaleX = (float) json.optDouble("preview_scale_x", previewScaleX);
            previewScaleY = (float) json.optDouble("preview_scale_y", previewScaleY);
            detectionScaleX = (float) json.optDouble("detection_scale_x", detectionScaleX);
            detectionScaleY = (float) json.optDouble("detection_scale_y", detectionScaleY);
            imgOrientation = json.optInt("img_orientation", imgOrientation);
            
            // Fisheye settings
            fisheyeEnabled = json.optBoolean("fisheye_enabled", fisheyeEnabled);
            fisheyeStrength = (float) json.optDouble("fisheye_strength", fisheyeStrength);
            fisheyeZoom = (float) json.optDouble("fisheye_zoom", fisheyeZoom);
            
            // Face detection tuning
            faceDetectionThreshold = (float) json.optDouble("face_detection_threshold", faceDetectionThreshold);
            minFaceSize = json.optInt("min_face_size", minFaceSize);
            faceScaleFactor = (float) json.optDouble("face_scale_factor", faceScaleFactor);
            
            // Crop region settings
            cropOffsetX = (float) json.optDouble("crop_offset_x", cropOffsetX);
            cropOffsetY = (float) json.optDouble("crop_offset_y", cropOffsetY);
            cropScale = (float) json.optDouble("crop_scale", cropScale);
            
            // Face crop settings
            faceCropPadding = (float) json.optDouble("face_crop_padding", faceCropPadding);
            faceCropDisplaySize = json.optInt("face_crop_display_size", faceCropDisplaySize);
            
            // Gaze detection thresholds
            gazePitchThreshold = (float) json.optDouble("gaze_pitch_threshold", gazePitchThreshold);
            gazeYawThreshold = (float) json.optDouble("gaze_yaw_threshold", gazeYawThreshold);
            gazePitchOnly = json.optBoolean("gaze_pitch_only", gazePitchOnly);
            gazeConsecutiveFrames = json.optInt("gaze_consecutive_frames", gazeConsecutiveFrames);
            
            // Smoothing parameters
            landmarkMinCutoff = json.optDouble("landmark_min_cutoff", landmarkMinCutoff);
            landmarkBeta = json.optDouble("landmark_beta", landmarkBeta);
            gazeMinCutoff = json.optDouble("gaze_min_cutoff", gazeMinCutoff);
            gazeBeta = json.optDouble("gaze_beta", gazeBeta);
            faceMinCutoff = json.optDouble("face_min_cutoff", faceMinCutoff);
            faceBeta = json.optDouble("face_beta", faceBeta);
            displaySmoothingAlpha = (float) json.optDouble("display_smoothing_alpha", displaySmoothingAlpha);
            
            Log.i(TAG, "════════════════════════════════════════");
            Log.i(TAG, "✓ Display config loaded from " + CONFIG_PATH);
            Log.i(TAG, "  preview_rotation: " + previewRotation);
            Log.i(TAG, "  detection_rotation: " + detectionRotation);
            Log.i(TAG, "  preview_scale_x: " + previewScaleX);
            Log.i(TAG, "  preview_scale_y: " + previewScaleY);
            Log.i(TAG, "  detection_scale_x: " + detectionScaleX);
            Log.i(TAG, "  detection_scale_y: " + detectionScaleY);
            Log.i(TAG, "  img_orientation: " + imgOrientation);
            Log.i(TAG, "  --- Fisheye Settings ---");
            Log.i(TAG, "  fisheye_enabled: " + fisheyeEnabled);
            Log.i(TAG, "  fisheye_strength: " + fisheyeStrength);
            Log.i(TAG, "  fisheye_zoom: " + fisheyeZoom);
            Log.i(TAG, "  --- Face Detection ---");
            Log.i(TAG, "  face_detection_threshold: " + faceDetectionThreshold);
            Log.i(TAG, "  min_face_size: " + minFaceSize);
            Log.i(TAG, "  face_scale_factor: " + faceScaleFactor);
            Log.i(TAG, "  --- Crop Region ---");
            Log.i(TAG, "  crop_offset_x: " + cropOffsetX + " (0=left, 0.5=center, 1=right)");
            Log.i(TAG, "  crop_offset_y: " + cropOffsetY + " (0=top, 0.5=center, 1=bottom)");
            Log.i(TAG, "  crop_scale: " + cropScale + " (0.5=half/2x zoom, 1.0=full)");
            Log.i(TAG, "  --- Face Crop ---");
            Log.i(TAG, "  face_crop_padding: " + faceCropPadding + "x");
            Log.i(TAG, "  face_crop_display_size: " + faceCropDisplaySize + "px");
            Log.i(TAG, "  --- Gaze Detection ---");
            Log.i(TAG, "  gaze_pitch_threshold: " + gazePitchThreshold + " rad (" + Math.toDegrees(gazePitchThreshold) + "°)");
            Log.i(TAG, "  gaze_yaw_threshold: " + gazeYawThreshold + " rad (" + Math.toDegrees(gazeYawThreshold) + "°)");
            Log.i(TAG, "  gaze_pitch_only: " + gazePitchOnly);
            Log.i(TAG, "  gaze_consecutive_frames: " + gazeConsecutiveFrames);
            Log.i(TAG, "  --- Smoothing (1-Euro Filter) ---");
            Log.i(TAG, "  landmark_min_cutoff: " + landmarkMinCutoff);
            Log.i(TAG, "  landmark_beta: " + landmarkBeta + " (lower=smoother)");
            Log.i(TAG, "  gaze_min_cutoff: " + gazeMinCutoff);
            Log.i(TAG, "  gaze_beta: " + gazeBeta + " (lower=smoother)");
            Log.i(TAG, "  face_min_cutoff: " + faceMinCutoff);
            Log.i(TAG, "  face_beta: " + faceBeta + " (lower=smoother)");
            Log.i(TAG, "  display_smoothing_alpha: " + displaySmoothingAlpha + " (lower=smoother)");
            Log.i(TAG, "════════════════════════════════════════");
            
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error reading config file: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Save configuration to JSON file
     * @return true if file was saved successfully
     */
    public boolean save() {
        File configFile = new File(CONFIG_PATH);
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(configFile))) {
            // Write with nice formatting
            writer.println("{");
            writer.println("  \"preview_rotation\": " + previewRotation + ",");
            writer.println("  \"detection_rotation\": " + detectionRotation + ",");
            writer.println("  \"preview_scale_x\": " + previewScaleX + ",");
            writer.println("  \"preview_scale_y\": " + previewScaleY + ",");
            writer.println("  \"detection_scale_x\": " + detectionScaleX + ",");
            writer.println("  \"detection_scale_y\": " + detectionScaleY + ",");
            writer.println("  \"img_orientation\": " + imgOrientation + ",");
            writer.println("");
            writer.println("  \"fisheye_enabled\": " + fisheyeEnabled + ",");
            writer.println("  \"fisheye_strength\": " + fisheyeStrength + ",");
            writer.println("  \"fisheye_zoom\": " + fisheyeZoom + ",");
            writer.println("");
            writer.println("  \"face_detection_threshold\": " + faceDetectionThreshold + ",");
            writer.println("  \"min_face_size\": " + minFaceSize + ",");
            writer.println("  \"face_scale_factor\": " + faceScaleFactor + ",");
            writer.println("");
            writer.println("  \"crop_offset_x\": " + cropOffsetX + ",");
            writer.println("  \"crop_offset_y\": " + cropOffsetY + ",");
            writer.println("  \"crop_scale\": " + cropScale + ",");
            writer.println("");
            writer.println("  \"face_crop_padding\": " + faceCropPadding + ",");
            writer.println("  \"face_crop_display_size\": " + faceCropDisplaySize + ",");
            writer.println("");
            writer.println("  \"gaze_pitch_threshold\": " + gazePitchThreshold + ",");
            writer.println("  \"gaze_yaw_threshold\": " + gazeYawThreshold + ",");
            writer.println("  \"gaze_pitch_only\": " + gazePitchOnly + ",");
            writer.println("  \"gaze_consecutive_frames\": " + gazeConsecutiveFrames + ",");
            writer.println("");
            writer.println("  \"landmark_min_cutoff\": " + landmarkMinCutoff + ",");
            writer.println("  \"landmark_beta\": " + landmarkBeta + ",");
            writer.println("  \"gaze_min_cutoff\": " + gazeMinCutoff + ",");
            writer.println("  \"gaze_beta\": " + gazeBeta + ",");
            writer.println("  \"face_min_cutoff\": " + faceMinCutoff + ",");
            writer.println("  \"face_beta\": " + faceBeta + ",");
            writer.println("  \"display_smoothing_alpha\": " + displaySmoothingAlpha);
            writer.println("}");
            
            lastModified = configFile.lastModified();
            Log.i(TAG, "✓ Config saved to " + CONFIG_PATH);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error writing config file: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Force reload configuration from file
     */
    public void reload() {
        lastModified = 0;
        load();
    }
}

