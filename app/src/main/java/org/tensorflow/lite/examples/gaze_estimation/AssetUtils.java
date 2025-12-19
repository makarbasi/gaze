package org.tensorflow.lite.examples.gaze_estimation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Minimal asset helpers to avoid depending on tensorflow-lite-support's FileUtil.
 */
public final class AssetUtils {
  private AssetUtils() {}

  public static MappedByteBuffer loadMappedFile(Context context, String assetPath) throws IOException {
    AssetFileDescriptor fileDescriptor = context.getAssets().openFd(assetPath);
    try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }
}


