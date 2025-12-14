QNN SDK Libraries Installation
==============================

To use QNN for running DLC files, you need to copy the QNN SDK libraries to this directory.

1. Download QNN SDK from Qualcomm:
   https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Engine_Direct
   
   Or from Qualcomm AI Hub:
   https://aihub.qualcomm.com/

2. After downloading and extracting, copy the following libraries to this directory:

   Required Core Libraries:
   - libQnnHtp.so          (HTP backend - recommended for Snapdragon)
   - libQnnModelDlc.so     (DLC model loader - REQUIRED for loading .dlc files)
   - libQnnSystem.so       (System utilities)
   
   Optional Backends (for fallback):
   - libQnnGpu.so          (GPU backend - Adreno)
   - libQnnDsp.so          (DSP backend - legacy Hexagon)
   - libQnnCpu.so          (CPU backend - reference implementation)
   
   HTP Skel/Stub Libraries (copy the ones matching your device's Hexagon version):
   - libQnnHtpPrepare.so
   - libQnnHtpV68Skel.so, libQnnHtpV68Stub.so  (Snapdragon 855, 865)
   - libQnnHtpV69Skel.so, libQnnHtpV69Stub.so  (Snapdragon 888)
   - libQnnHtpV73Skel.so, libQnnHtpV73Stub.so  (Snapdragon 8 Gen 1)
   - libQnnHtpV75Skel.so, libQnnHtpV75Stub.so  (Snapdragon 8 Gen 2)
   - libQnnHtpV79Skel.so, libQnnHtpV79Stub.so  (Snapdragon 8 Gen 3)
   - libQnnHtpV81Skel.so, libQnnHtpV81Stub.so  (Snapdragon 8 Gen 3 Elite)
   - libQnnHtpV85Skel.so, libQnnHtpV85Stub.so  (Snapdragon 8 Gen 4)

3. The QNN SDK is usually found at:
   <QNN_SDK>/lib/aarch64-android/

4. Also copy the QNN SDK headers to:
   app/src/main/cpp/include/
   
   Required headers:
   - QNN/*.h (all QNN headers)
   - DlSystem/*.hpp
   - DlContainer/*.hpp

5. Build and run the app. Check logcat for "QNN" tag to see initialization status.

Troubleshooting:
- If you see "Failed to load libQnnHtp.so", make sure the library is in this directory
- If you see "Failed to load libQnnModelDlc.so", the DLC loader is missing
- Make sure to also push the DLC files to /data/local/tmp/ on the device

Hexagon Version by Snapdragon SoC:
- V68: Snapdragon 855/855+/865/865+/870
- V69: Snapdragon 888/888+
- V73: Snapdragon 8 Gen 1/8+ Gen 1
- V75: Snapdragon 8 Gen 2
- V79: Snapdragon 8 Gen 3/8s Gen 3
- V81: Snapdragon 8 Elite
- V85: Future devices

