package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Intent;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

/**
 * Launcher activity that lets user choose between Camera mode, Single Image mode, and Batch mode.
 */
public class LauncherActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Create a scroll view for smaller screens
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        
        // Create a simple layout programmatically
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(android.view.Gravity.CENTER);
        layout.setPadding(48, 64, 48, 64);
        
        // Create gradient background
        GradientDrawable gradient = new GradientDrawable(
            GradientDrawable.Orientation.TOP_BOTTOM,
            new int[]{0xFF0D1B2A, 0xFF1B263B, 0xFF415A77}
        );
        layout.setBackground(gradient);
        
        // Title
        TextView title = new TextView(this);
        title.setText("👁 Gaze Estimation");
        title.setTextSize(32);
        title.setTextColor(0xFFE0E1DD);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setGravity(android.view.Gravity.CENTER);
        title.setPadding(0, 0, 0, 8);
        layout.addView(title);
        
        // Subtitle
        TextView subtitle = new TextView(this);
        subtitle.setText("Select input mode");
        subtitle.setTextSize(16);
        subtitle.setTextColor(0xFF778DA9);
        subtitle.setGravity(android.view.Gravity.CENTER);
        subtitle.setPadding(0, 0, 0, 48);
        layout.addView(subtitle);
        
        // === Camera Button ===
        Button cameraButton = createStyledButton(
            "📷  Live Camera",
            "Real-time gaze tracking from camera feed",
            0xFF00B4D8,
            0xFF0077B6
        );
        LinearLayout.LayoutParams cameraParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        cameraParams.setMargins(0, 0, 0, 20);
        cameraButton.setLayoutParams(cameraParams);
        cameraButton.setOnClickListener(v -> startCameraMode());
        layout.addView(cameraButton);
        
        // === Select Camera Button (new) ===
        Button selectCameraButton = createStyledButton(
            getString(R.string.select_camera_mode_title),
            getString(R.string.select_camera_mode_desc),
            0xFF2A9D8F,
            0xFF1F7A6E
        );
        LinearLayout.LayoutParams selectCamParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        selectCamParams.setMargins(0, 0, 0, 20);
        selectCameraButton.setLayoutParams(selectCamParams);
        selectCameraButton.setOnClickListener(v -> startCameraPickerMode());
        layout.addView(selectCameraButton);

        // === Single Image Button ===
        Button singleImageButton = createStyledButton(
            "🖼  Single Image",
            "Select an image and analyze gaze direction",
            0xFF7B2CBF,
            0xFF5A189A
        );
        LinearLayout.LayoutParams singleParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        singleParams.setMargins(0, 0, 0, 20);
        singleImageButton.setLayoutParams(singleParams);
        singleImageButton.setOnClickListener(v -> startSingleImageMode());
        layout.addView(singleImageButton);
        
        // === Batch Processing Button ===
        Button batchButton = createStyledButton(
            "📁  Batch Processing",
            "Process multiple images from a folder",
            0xFFFF6B35,
            0xFFE85D04
        );
        LinearLayout.LayoutParams batchParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
        batchParams.setMargins(0, 0, 0, 0);
        batchButton.setLayoutParams(batchParams);
        batchButton.setOnClickListener(v -> startFilesMode());
        layout.addView(batchButton);
        
        // Info section
        TextView info = new TextView(this);
        info.setText("\n─────────────────\n\n" +
            "• Live Camera: Use device camera for real-time tracking\n\n" +
            "• Select Camera: Preview all cameras and choose one\n\n" +
            "• Single Image: Pick one photo to analyze\n\n" +
            "• Batch: Process many images at once");
        info.setTextSize(13);
        info.setTextColor(0xFF778DA9);
        info.setGravity(android.view.Gravity.CENTER);
        info.setPadding(0, 32, 0, 0);
        layout.addView(info);
        
        scrollView.addView(layout);
        setContentView(scrollView);
    }
    
    /**
     * Create a styled button with gradient background
     */
    private Button createStyledButton(String text, String description, int colorStart, int colorEnd) {
        Button button = new Button(this);
        button.setText(text + "\n" + description);
        button.setTextSize(16);
        button.setTextColor(0xFFFFFFFF);
        button.setPadding(48, 40, 48, 40);
        button.setAllCaps(false);
        
        // Create gradient drawable for button
        GradientDrawable buttonGradient = new GradientDrawable(
            GradientDrawable.Orientation.LEFT_RIGHT,
            new int[]{colorStart, colorEnd}
        );
        buttonGradient.setCornerRadius(24);
        button.setBackground(buttonGradient);
        
        return button;
    }
    
    private void startCameraMode() {
        Intent intent = new Intent(this, ClassifierActivity.class);
        startActivity(intent);
        finish();
    }

    private void startCameraPickerMode() {
        Intent intent = new Intent(this, CameraPickerActivity.class);
        startActivity(intent);
        // Do NOT finish here so the user can press Back to return to this launcher screen.
    }
    
    private void startSingleImageMode() {
        Intent intent = new Intent(this, SingleImageActivity.class);
        startActivity(intent);
        finish();
    }
    
    private void startFilesMode() {
        Intent intent = new Intent(this, ImageProcessingActivity.class);
        startActivity(intent);
        finish();
    }
}

