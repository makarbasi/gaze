package org.tensorflow.lite.examples.gaze_estimation;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import android.view.View;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.RadioButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Lets the user choose a camera WITH a live preview before entering the main live-camera pipeline.
 *
 * Flow:
 *  - Enumerate all cameras (Camera2)
 *  - Show a live preview for each camera (when possible)
 *  - User selects camera via radio button
 *  - User confirms -> start {@link ClassifierActivity} with selected cameraId extra
 */
public class CameraPickerActivity extends AppCompatActivity {
  private static final int PERMISSIONS_REQUEST = 99;
  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

  public static final String EXTRA_CAMERA_ID = "extra_camera_id";

  private static final String PREFS_NAME = "gaze_prefs";
  private static final String PREF_SELECTED_CAMERA_ID = "selected_camera_id";

  private Button useButton;
  private TextView statusText;
  private LinearLayout cameraListContainer;

  private HandlerThread backgroundThread;
  private Handler backgroundHandler;

  private String selectedCameraId;
  private List<CameraOption> cameraOptions = new ArrayList<>();
  private final List<CameraPreviewController> previewControllers = new ArrayList<>();

  private static final class CameraOption {
    final String cameraId;
    final String label;

    CameraOption(String cameraId, String label) {
      this.cameraId = cameraId;
      this.label = label;
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera_picker);

    useButton = findViewById(R.id.camera_picker_use_button);
    statusText = findViewById(R.id.camera_picker_status);
    cameraListContainer = findViewById(R.id.camera_picker_list);

    useButton.setEnabled(false);

    useButton.setOnClickListener(
        v -> {
          if (selectedCameraId == null) return;
          persistSelectedCameraId(selectedCameraId);
          Intent intent = new Intent(this, ClassifierActivity.class);
          intent.putExtra(EXTRA_CAMERA_ID, selectedCameraId);
          startActivity(intent);
          finish();
        });

    if (hasPermission()) {
      loadCameraListAndBindUi();
    } else {
      requestPermission();
    }
  }

  @Override
  protected void onResume() {
    super.onResume();
    startBackgroundThread();
    if (hasPermission()) {
      loadCameraListAndBindUi();
      startAllPreviewsIfReady();
    }
  }

  @Override
  protected void onPause() {
    stopAllPreviews();
    stopBackgroundThread();
    super.onPause();
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        loadCameraListAndBindUi();
        startAllPreviewsIfReady();
      } else {
        finish();
      }
    }
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
    return true;
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  private void startBackgroundThread() {
    if (backgroundThread != null) return;
    backgroundThread = new HandlerThread("CameraPickerBackground");
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    if (backgroundThread == null) return;
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
    } catch (InterruptedException ignored) {
    } finally {
      backgroundThread = null;
      backgroundHandler = null;
    }
  }

  private String loadPersistedCameraId() {
    SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
    return prefs.getString(PREF_SELECTED_CAMERA_ID, null);
  }

  private void persistSelectedCameraId(String cameraId) {
    SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
    prefs.edit().putString(PREF_SELECTED_CAMERA_ID, cameraId).apply();
  }

  private void loadCameraListAndBindUi() {
    cameraOptions = getAvailableCameraOptions();
    if (cameraOptions.isEmpty()) {
      statusText.setText(getString(R.string.no_cameras_found));
      useButton.setEnabled(false);
      return;
    }

    String persisted = loadPersistedCameraId();
    int initialIndex = 0;
    if (persisted != null) {
      for (int i = 0; i < cameraOptions.size(); i++) {
        if (persisted.equals(cameraOptions.get(i).cameraId)) {
          initialIndex = i;
          break;
        }
      }
    }

    selectedCameraId = cameraOptions.get(initialIndex).cameraId;
    useButton.setEnabled(true);
    statusText.setText(
        getString(R.string.select_camera_title) + ": " + cameraOptions.get(initialIndex).label);

    // Rebuild list UI + previews
    buildCameraListUi(initialIndex);
  }

  private void buildCameraListUi(int initiallySelectedIndex) {
    stopAllPreviews();
    previewControllers.clear();
    if (cameraListContainer == null) return;
    cameraListContainer.removeAllViews();

    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

    for (int i = 0; i < cameraOptions.size(); i++) {
      final int index = i;
      final CameraOption opt = cameraOptions.get(i);

      View row = getLayoutInflater().inflate(R.layout.item_camera_option, cameraListContainer, false);
      TextView label = row.findViewById(R.id.camera_option_label);
      RadioButton radio = row.findViewById(R.id.camera_option_radio);
      TextureView texture = row.findViewById(R.id.camera_option_texture);
      TextView overlay = row.findViewById(R.id.camera_option_overlay);

      label.setText(opt.label);
      radio.setChecked(index == initiallySelectedIndex);
      overlay.setText(getString(R.string.camera_preview_label));

      View.OnClickListener selectListener =
          v -> {
            selectedCameraId = opt.cameraId;
            useButton.setEnabled(true);
            statusText.setText(getString(R.string.select_camera_title) + ": " + opt.label);
            // Exclusive selection
            for (int j = 0; j < cameraListContainer.getChildCount(); j++) {
              View child = cameraListContainer.getChildAt(j);
              RadioButton rb = child.findViewById(R.id.camera_option_radio);
              if (rb != null) rb.setChecked(j == index);
            }
          };

      radio.setOnClickListener(selectListener);
      row.setOnClickListener(selectListener);

      CameraPreviewController controller =
          new CameraPreviewController(opt.cameraId, texture, overlay);
      previewControllers.add(controller);

      texture.setSurfaceTextureListener(
          new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
              if (!hasPermission() || backgroundHandler == null) return;
              controller.start(manager, backgroundHandler);
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
              controller.stop();
              return true;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {}
          });

      cameraListContainer.addView(row);
    }

    startAllPreviewsIfReady();
  }

  private void startAllPreviewsIfReady() {
    if (!hasPermission() || backgroundHandler == null) return;
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    for (CameraPreviewController c : previewControllers) {
      c.start(manager, backgroundHandler);
    }
  }

  private void stopAllPreviews() {
    for (CameraPreviewController c : previewControllers) {
      c.stop();
    }
  }

  private List<CameraOption> getAvailableCameraOptions() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    final List<CameraOption> options = new ArrayList<>();
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);

        final String facingLabel;
        if (facing == null) {
          facingLabel = "Unknown";
        } else if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
          facingLabel = "Front";
        } else if (facing == CameraCharacteristics.LENS_FACING_BACK) {
          facingLabel = "Back";
        } else if (facing == CameraCharacteristics.LENS_FACING_EXTERNAL) {
          facingLabel = "External";
        } else {
          facingLabel = "Other";
        }

        // Ensure it can stream to a SurfaceTexture.
        final StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) continue;

        options.add(new CameraOption(cameraId, "Camera " + cameraId + " (" + facingLabel + ")"));
      }
    } catch (CameraAccessException ignored) {
    }

    Collections.sort(
        options,
        new Comparator<CameraOption>() {
          @Override
          public int compare(CameraOption a, CameraOption b) {
            try {
              return Integer.compare(Integer.parseInt(a.cameraId), Integer.parseInt(b.cameraId));
            } catch (Exception ignored) {
              return a.cameraId.compareTo(b.cameraId);
            }
          }
        });

    return options;
  }

  /**
   * Per-row Camera2 preview controller (opens one camera and renders into one TextureView).
   * Best-effort: if the device cannot open multiple cameras at once, some rows may fail to start.
   */
  private static final class CameraPreviewController {
    private final String cameraId;
    private final TextureView textureView;
    private final TextView overlay;

    private final Semaphore cameraOpenCloseLock = new Semaphore(1);
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewRequestBuilder;

    CameraPreviewController(String cameraId, TextureView textureView, TextView overlay) {
      this.cameraId = cameraId;
      this.textureView = textureView;
      this.overlay = overlay;
    }

    void start(CameraManager manager, Handler handler) {
      if (cameraDevice != null) return;
      if (textureView == null || !textureView.isAvailable()) return;
      if (manager == null || handler == null) return;

      try {
        if (!cameraOpenCloseLock.tryAcquire(250, TimeUnit.MILLISECONDS)) return;

        try {
          manager.openCamera(
              cameraId,
              new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                  cameraOpenCloseLock.release();
                  cameraDevice = camera;
                  if (overlay != null) overlay.setText("Camera " + cameraId);
                  createPreviewSession(manager, handler);
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                  cameraOpenCloseLock.release();
                  camera.close();
                  cameraDevice = null;
                  if (overlay != null) overlay.setText("Disconnected: " + cameraId);
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                  cameraOpenCloseLock.release();
                  camera.close();
                  cameraDevice = null;
                  if (overlay != null) overlay.setText("Unavailable: " + cameraId);
                }
              },
              handler);
        } catch (CameraAccessException e) {
          cameraOpenCloseLock.release();
          if (overlay != null) overlay.setText("Access error: " + cameraId);
        }
      } catch (InterruptedException ignored) {
        cameraOpenCloseLock.release();
      }
    }

    private void createPreviewSession(CameraManager manager, Handler handler) {
      try {
        if (cameraDevice == null || !textureView.isAvailable()) return;
        SurfaceTexture texture = textureView.getSurfaceTexture();
        if (texture == null) return;

        Size previewSize = choosePreviewSize(manager, cameraId);
        if (previewSize != null) {
          texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
        }

        Surface surface = new Surface(texture);
        previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        previewRequestBuilder.addTarget(surface);
        previewRequestBuilder.set(
            CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

        cameraDevice.createCaptureSession(
            Arrays.asList(surface),
            new CameraCaptureSession.StateCallback() {
              @Override
              public void onConfigured(@NonNull CameraCaptureSession session) {
                if (cameraDevice == null) return;
                captureSession = session;
                try {
                  session.setRepeatingRequest(previewRequestBuilder.build(), null, handler);
                } catch (CameraAccessException ignored) {
                }
              }

              @Override
              public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                if (overlay != null) overlay.setText("Preview failed: " + cameraId);
              }
            },
            handler);
      } catch (Exception e) {
        if (overlay != null) overlay.setText("Preview error: " + cameraId);
      }
    }

    private static Size choosePreviewSize(CameraManager manager, String cameraId) {
      try {
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
        StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) return null;
        Size[] sizes = map.getOutputSizes(SurfaceTexture.class);
        if (sizes == null || sizes.length == 0) return null;

        // Prefer a smaller preview to make multi-camera feasible.
        Size best = sizes[0];
        for (Size s : sizes) {
          if (s.getWidth() == 640 && s.getHeight() == 480) return s;
          if (s.getWidth() == 320 && s.getHeight() == 240) return s;
        }
        // Otherwise pick the smallest area size.
        for (Size s : sizes) {
          if ((long) s.getWidth() * (long) s.getHeight()
              < (long) best.getWidth() * (long) best.getHeight()) {
            best = s;
          }
        }
        return best;
      } catch (CameraAccessException e) {
        return null;
      }
    }

    void stop() {
      try {
        cameraOpenCloseLock.acquire();
        if (captureSession != null) {
          captureSession.close();
          captureSession = null;
        }
        if (cameraDevice != null) {
          cameraDevice.close();
          cameraDevice = null;
        }
      } catch (InterruptedException ignored) {
      } finally {
        cameraOpenCloseLock.release();
      }
    }
  }
}


