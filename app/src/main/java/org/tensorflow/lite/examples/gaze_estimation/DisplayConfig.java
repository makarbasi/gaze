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
    public float previewRotation = 0f;        // Main preview rotation
    public float detectionRotation = 0f;      // Detection/landmark display rotation
    public float previewScaleX = 1.0f;        // Preview horizontal scale (-1 to flip)
    public float previewScaleY = 1.0f;        // Preview vertical scale (-1 to flip)
    public float detectionScaleX = 1.0f;      // Detection horizontal scale
    public float detectionScaleY = 1.0f;      // Detection vertical scale
    public int imgOrientation = 0;            // Crop transformation rotation
    
    // Fisheye camera settings
    public boolean fisheyeEnabled = false;    // Enable fisheye undistortion
    public float fisheyeStrength = 0.5f;      // Undistortion strength (0.0-1.0)
    public float fisheyeZoom = 1.5f;          // Zoom factor after undistortion
    
    // Face detection tuning
    public float faceDetectionThreshold = 0.3f;  // Lower for fisheye (default 0.5)
    public int minFaceSize = 15;                 // Smaller for fisheye (default 30)
    public float faceScaleFactor = 1.5f;         // Scale up face box for landmarks
    
    // Crop region settings (for wide FOV cameras)
    // crop_side: "left", "right", "center", or "full" - easy preset for which half to use
    // crop_offset_x/y and crop_scale are computed from crop_side, or can be set manually
    public String cropSide = "left";             // Which side to use: "left", "right", "center", "full"
    public float cropOffsetX = 0.0f;             // Computed from cropSide (0=left, 0.5=center, 1=right)
    public float cropOffsetY = 0.5f;             // Vertical center
    public float cropScale = 0.5f;               // Computed from cropSide (0.5=half, 1.0=full)
    
    // Face crop settings
    public float faceCropPadding = 2.0f;         // Padding around face box (2.0 = 2x the box size)
    public int faceCropDisplaySize = 300;        // Display size for face crop (pixels)
    
    // Gaze detection thresholds (in DEGREES - matches display values!)
    public float gazePitchThresholdDegrees = 15f;   // Pitch threshold in degrees - looking up/down
    public float gazeYawThresholdDegrees = 20f;     // Yaw threshold in degrees - looking left/right
    public boolean gazePitchOnly = false;           // If true, only use pitch for detection (ignore yaw)
    public int gazeConsecutiveFrames = 3;           // Consecutive frames needed to change state
    
    // Computed thresholds in radians (for internal use)
    public float gazePitchThreshold = (float) Math.toRadians(15);
    public float gazeYawThreshold = (float) Math.toRadians(20);
    
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
    
    // Frame processing settings (for slower devices)
    // Frames are automatically skipped when AI is still processing (no forced skipping)
    // Use min_frame_interval_ms to add extra throttling if needed
    public int minFrameIntervalMs = 0;           // Minimum milliseconds between processed frames (0=no limit, 100=max 10fps, 200=max 5fps)
    
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
            // If crop_side is specified, it takes priority over manual offset/scale values
            if (json.has("crop_side")) {
                cropSide = json.optString("crop_side", cropSide);
                applyCropSidePreset(cropSide);
                // Ignore crop_offset_x and crop_scale when crop_side is used
                Log.i(TAG, "Using crop_side='" + cropSide + "', ignoring any crop_offset_x/crop_scale values");
            } else {
                // No crop_side specified, use manual values
                cropOffsetX = (float) json.optDouble("crop_offset_x", cropOffsetX);
                cropScale = (float) json.optDouble("crop_scale", cropScale);
            }
            // crop_offset_y is always read (vertical position)
            cropOffsetY = (float) json.optDouble("crop_offset_y", cropOffsetY);
            
            // Face crop settings
            faceCropPadding = (float) json.optDouble("face_crop_padding", faceCropPadding);
            faceCropDisplaySize = json.optInt("face_crop_display_size", faceCropDisplaySize);
            
            // Gaze detection thresholds (read in DEGREES, convert to radians internally)
            gazePitchThresholdDegrees = (float) json.optDouble("gaze_pitch_threshold", gazePitchThresholdDegrees);
            gazeYawThresholdDegrees = (float) json.optDouble("gaze_yaw_threshold", gazeYawThresholdDegrees);
            // Convert to radians for internal use
            gazePitchThreshold = (float) Math.toRadians(gazePitchThresholdDegrees);
            gazeYawThreshold = (float) Math.toRadians(gazeYawThresholdDegrees);
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
            
            // Frame processing settings
            minFrameIntervalMs = json.optInt("min_frame_interval_ms", minFrameIntervalMs);
            
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
            Log.i(TAG, "  crop_side: '" + cropSide + "' (left/right/center/full)");
            Log.i(TAG, "  crop_offset_x: " + cropOffsetX + " (computed from crop_side)");
            Log.i(TAG, "  crop_offset_y: " + cropOffsetY);
            Log.i(TAG, "  crop_scale: " + cropScale);
            Log.i(TAG, "  --- Face Crop ---");
            Log.i(TAG, "  face_crop_padding: " + faceCropPadding + "x");
            Log.i(TAG, "  face_crop_display_size: " + faceCropDisplaySize + "px");
            Log.i(TAG, "  --- Gaze Detection ---");
            Log.i(TAG, "  gaze_pitch_threshold: " + gazePitchThresholdDegrees + "° (config uses DEGREES now!)");
            Log.i(TAG, "  gaze_yaw_threshold: " + gazeYawThresholdDegrees + "°");
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
            Log.i(TAG, "  --- Frame Processing ---");
            Log.i(TAG, "  min_frame_interval_ms: " + minFrameIntervalMs + "ms (0=no limit, frames auto-skipped during processing)");
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
            writer.println("  \"crop_side\": \"" + cropSide + "\",");
            writer.println("  \"crop_offset_y\": " + cropOffsetY + ",");
            writer.println("");
            writer.println("  \"face_crop_padding\": " + faceCropPadding + ",");
            writer.println("  \"face_crop_display_size\": " + faceCropDisplaySize + ",");
            writer.println("");
            writer.println("  \"gaze_pitch_threshold\": " + gazePitchThresholdDegrees + ",");
            writer.println("  \"gaze_yaw_threshold\": " + gazeYawThresholdDegrees + ",");
            writer.println("  \"gaze_pitch_only\": " + gazePitchOnly + ",");
            writer.println("  \"gaze_consecutive_frames\": " + gazeConsecutiveFrames + ",");
            writer.println("");
            writer.println("  \"landmark_min_cutoff\": " + landmarkMinCutoff + ",");
            writer.println("  \"landmark_beta\": " + landmarkBeta + ",");
            writer.println("  \"gaze_min_cutoff\": " + gazeMinCutoff + ",");
            writer.println("  \"gaze_beta\": " + gazeBeta + ",");
            writer.println("  \"face_min_cutoff\": " + faceMinCutoff + ",");
            writer.println("  \"face_beta\": " + faceBeta + ",");
            writer.println("  \"display_smoothing_alpha\": " + displaySmoothingAlpha + ",");
            writer.println("");
            writer.println("  \"min_frame_interval_ms\": " + minFrameIntervalMs);
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
     * Apply crop side preset - sets cropOffsetX and cropScale based on side name
     * @param side "left", "right", "center", or "full"
     */
    private void applyCropSidePreset(String side) {
        switch (side.toLowerCase()) {
            case "left":
                cropOffsetX = 0.0f;      // Start from left edge
                cropScale = 0.5f;        // Use half width
                break;
            case "right":
                cropOffsetX = 1.0f;      // Start from right edge
                cropScale = 0.5f;        // Use half width
                break;
            case "center":
                cropOffsetX = 0.5f;      // Center
                cropScale = 0.5f;        // Use half width (center half)
                break;
            case "full":
                cropOffsetX = 0.5f;      // Center (doesn't matter for full)
                cropScale = 1.0f;        // Use full width
                break;
            default:
                Log.w(TAG, "Unknown crop_side: " + side + ", using 'left'");
                cropOffsetX = 0.0f;
                cropScale = 0.5f;
        }
        Log.d(TAG, "Applied crop_side='" + side + "' -> offsetX=" + cropOffsetX + ", scale=" + cropScale);
    }
    
    /**
     * Force reload configuration from file
     */
    public void reload() {
        lastModified = 0;
        load();
    }
}

