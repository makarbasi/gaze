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
 *   "img_orientation": 90          // Transformation matrix rotation for cropping
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
    
    // Default values
    public float previewRotation = -90f;      // Main preview rotation
    public float detectionRotation = 90f;     // Detection/landmark display rotation
    public float previewScaleX = 1.0f;        // Preview horizontal scale (-1 to flip)
    public float previewScaleY = 1.0f;        // Preview vertical scale (-1 to flip)
    public float detectionScaleX = 1.0f;      // Detection horizontal scale
    public float detectionScaleY = 1.0f;      // Detection vertical scale
    public int imgOrientation = 90;           // Crop transformation rotation
    
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
        if (configFile.exists() && configFile.lastModified() > lastModified) {
            Log.i(TAG, "Config file modified, reloading...");
            load();
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
            
            JSONObject json = new JSONObject(jsonBuilder.toString());
            
            previewRotation = (float) json.optDouble("preview_rotation", previewRotation);
            detectionRotation = (float) json.optDouble("detection_rotation", detectionRotation);
            previewScaleX = (float) json.optDouble("preview_scale_x", previewScaleX);
            previewScaleY = (float) json.optDouble("preview_scale_y", previewScaleY);
            detectionScaleX = (float) json.optDouble("detection_scale_x", detectionScaleX);
            detectionScaleY = (float) json.optDouble("detection_scale_y", detectionScaleY);
            imgOrientation = json.optInt("img_orientation", imgOrientation);
            
            Log.i(TAG, "════════════════════════════════════════");
            Log.i(TAG, "✓ Display config loaded from " + CONFIG_PATH);
            Log.i(TAG, "  preview_rotation: " + previewRotation);
            Log.i(TAG, "  detection_rotation: " + detectionRotation);
            Log.i(TAG, "  preview_scale_x: " + previewScaleX);
            Log.i(TAG, "  preview_scale_y: " + previewScaleY);
            Log.i(TAG, "  detection_scale_x: " + detectionScaleX);
            Log.i(TAG, "  detection_scale_y: " + detectionScaleY);
            Log.i(TAG, "  img_orientation: " + imgOrientation);
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
            JSONObject json = new JSONObject();
            json.put("preview_rotation", previewRotation);
            json.put("detection_rotation", detectionRotation);
            json.put("preview_scale_x", previewScaleX);
            json.put("preview_scale_y", previewScaleY);
            json.put("detection_scale_x", detectionScaleX);
            json.put("detection_scale_y", detectionScaleY);
            json.put("img_orientation", imgOrientation);
            
            // Write with nice formatting
            writer.println("{");
            writer.println("  \"preview_rotation\": " + previewRotation + ",");
            writer.println("  \"detection_rotation\": " + detectionRotation + ",");
            writer.println("  \"preview_scale_x\": " + previewScaleX + ",");
            writer.println("  \"preview_scale_y\": " + previewScaleY + ",");
            writer.println("  \"detection_scale_x\": " + detectionScaleX + ",");
            writer.println("  \"detection_scale_y\": " + detectionScaleY + ",");
            writer.println("  \"img_orientation\": " + imgOrientation);
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

