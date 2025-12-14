package org.tensorflow.lite.examples.gaze_estimation;

import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Gaze Calibration Configuration
 * 
 * Reads and writes gaze calibration settings from/to:
 *   /data/local/tmp/gaze_config.txt
 * 
 * Config file format (key=value):
 *   reference_pitch=0.0       # Reference pitch in radians
 *   reference_yaw=0.0         # Reference yaw in radians
 *   threshold_base=0.20       # Base threshold in radians (~11.5 degrees)
 *   threshold_max=0.35        # Max threshold in radians (~20 degrees)
 *   smoothing_frames=3        # Consecutive frames needed to change state
 *   smoothing_alpha=0.4       # Exponential smoothing factor (0-1)
 *   auto_calibrate=false      # If true, use saved reference on startup
 */
public class GazeConfig {
    private static final String TAG = "GazeConfig";
    public static final String CONFIG_PATH = "/data/local/tmp/gaze_config.txt";
    
    // Default values
    public float referencePitch = 0.0f;
    public float referenceYaw = 0.0f;
    public float thresholdBase = 0.20f;      // ~11.5 degrees
    public float thresholdMax = 0.35f;       // ~20 degrees
    public int smoothingFrames = 3;
    public float smoothingAlpha = 0.4f;
    public boolean autoCalibrate = false;    // Use saved reference on startup
    
    // Singleton instance
    private static GazeConfig instance = null;
    
    public static GazeConfig getInstance() {
        if (instance == null) {
            instance = new GazeConfig();
            instance.load();
        }
        return instance;
    }
    
    /**
     * Load configuration from file
     * @return true if file was loaded successfully
     */
    public boolean load() {
        File configFile = new File(CONFIG_PATH);
        if (!configFile.exists()) {
            Log.i(TAG, "Config file not found at " + CONFIG_PATH + ", using defaults");
            // Create default config file
            save();
            return false;
        }
        
        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                
                // Skip empty lines and comments
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                
                // Parse key=value
                int equalsIndex = line.indexOf('=');
                if (equalsIndex > 0) {
                    String key = line.substring(0, equalsIndex).trim().toLowerCase();
                    String value = line.substring(equalsIndex + 1).trim();
                    
                    // Remove inline comments
                    int commentIndex = value.indexOf('#');
                    if (commentIndex >= 0) {
                        value = value.substring(0, commentIndex).trim();
                    }
                    
                    parseKeyValue(key, value);
                }
            }
            
            Log.i(TAG, "✓ Config loaded from " + CONFIG_PATH);
            Log.i(TAG, "  reference_pitch=" + Math.toDegrees(referencePitch) + "°");
            Log.i(TAG, "  reference_yaw=" + Math.toDegrees(referenceYaw) + "°");
            Log.i(TAG, "  threshold_base=" + Math.toDegrees(thresholdBase) + "°");
            Log.i(TAG, "  threshold_max=" + Math.toDegrees(thresholdMax) + "°");
            Log.i(TAG, "  smoothing_frames=" + smoothingFrames);
            Log.i(TAG, "  smoothing_alpha=" + smoothingAlpha);
            Log.i(TAG, "  auto_calibrate=" + autoCalibrate);
            
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error reading config file: " + e.getMessage());
            return false;
        }
    }
    
    private void parseKeyValue(String key, String value) {
        try {
            switch (key) {
                case "reference_pitch":
                    referencePitch = Float.parseFloat(value);
                    break;
                case "reference_yaw":
                    referenceYaw = Float.parseFloat(value);
                    break;
                case "threshold_base":
                    thresholdBase = Float.parseFloat(value);
                    break;
                case "threshold_max":
                    thresholdMax = Float.parseFloat(value);
                    break;
                case "smoothing_frames":
                    smoothingFrames = Integer.parseInt(value);
                    break;
                case "smoothing_alpha":
                    smoothingAlpha = Float.parseFloat(value);
                    break;
                case "auto_calibrate":
                    autoCalibrate = Boolean.parseBoolean(value);
                    break;
                default:
                    Log.w(TAG, "Unknown config key: " + key);
            }
        } catch (NumberFormatException e) {
            Log.e(TAG, "Error parsing value for " + key + ": " + value);
        }
    }
    
    /**
     * Save configuration to file
     * @return true if file was saved successfully
     */
    public boolean save() {
        File configFile = new File(CONFIG_PATH);
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(configFile))) {
            writer.println("# Gaze Calibration Configuration");
            writer.println("# Values in radians unless noted otherwise");
            writer.println();
            writer.println("# Reference gaze position (set by calibration button)");
            writer.println("reference_pitch=" + referencePitch);
            writer.println("reference_yaw=" + referenceYaw);
            writer.println();
            writer.println("# Threshold settings");
            writer.println("# Base threshold: ~" + String.format("%.1f", Math.toDegrees(thresholdBase)) + " degrees");
            writer.println("threshold_base=" + thresholdBase);
            writer.println("# Max threshold (with head movement compensation): ~" + String.format("%.1f", Math.toDegrees(thresholdMax)) + " degrees");
            writer.println("threshold_max=" + thresholdMax);
            writer.println();
            writer.println("# Smoothing settings");
            writer.println("# Number of consecutive frames to change looking/not-looking state");
            writer.println("smoothing_frames=" + smoothingFrames);
            writer.println("# Exponential smoothing alpha (0-1, higher = more responsive)");
            writer.println("smoothing_alpha=" + smoothingAlpha);
            writer.println();
            writer.println("# Auto-calibrate: if true, use saved reference on startup");
            writer.println("auto_calibrate=" + autoCalibrate);
            
            Log.i(TAG, "✓ Config saved to " + CONFIG_PATH);
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error writing config file: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Update reference values and save to file
     */
    public void updateReference(float pitch, float yaw) {
        this.referencePitch = pitch;
        this.referenceYaw = yaw;
        this.autoCalibrate = true;  // Enable auto-calibrate after manual calibration
        save();
    }
    
    /**
     * Reload configuration from file
     */
    public void reload() {
        load();
    }
}
