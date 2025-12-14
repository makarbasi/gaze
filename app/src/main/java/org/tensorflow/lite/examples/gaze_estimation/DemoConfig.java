package org.tensorflow.lite.examples.gaze_estimation;

public class DemoConfig {
    /* application config */
    final public static float NMS_THR = 0.3f;  // Updated for TFLite model
    final public static float CONF_THR = 0.5f;

    public final static double REID_DIFF_THR = 50.0f;
    public final static int REID_MAX_TICK = 20;

    final public static Boolean USE_VERTICAL = true;
    final public static Boolean USE_FRONT_CAM = true;

    final public static double face_min_cutoff = 0.01;
    final public static double face_beta = 0.1;
    final public static double landmark_min_cutoff = 0.01;
    final public static double landmark_beta = 0.1;
    final public static double gaze_min_cutoff = 0.01;
    final public static double gaze_beta = 0.8;

    // ==== TFLite Face Detection (from HotGaze - more accurate model) ====
    final public static boolean USE_TFLITE_FACE_DETECTION = true;  // Toggle to switch between TFLite and QNN (DLC)
    final public static String tflite_face_detection_model = "face_detection.tflite";
    final public static int tflite_face_detection_input_W = 640;
    final public static int tflite_face_detection_input_H = 480;
    final public static float tflite_face_detection_threshold = 0.5f;
    final public static int tflite_min_face_size = 30;
    final public static float tflite_nms_iou_threshold = 0.3f;

    // ==== TFLite Landmark Detection ====
    // TFLite + NNAPI uses QNN on Qualcomm v81+ devices
    final public static boolean USE_TFLITE_LANDMARK_DETECTION = false;  // Set to false to use QNN with DLC model
    final public static String tflite_landmark_detection_model = "landmark_detection.tflite";

    // ==== TFLite Gaze Estimation ====
    // TFLite + NNAPI uses QNN on Qualcomm v81+ devices
    final public static boolean USE_TFLITE_GAZE_ESTIMATION = false;  // Set to false to use QNN with DLC model
    final public static String tflite_gaze_estimation_model = "gaze_estimation.tflite";

    // ==== QNN DLC Models (replaces SNPE) ====
    // QNN is used to run DLC files for landmark detection and gaze estimation
    // Set USE_TFLITE_* = false to use these DLC models with QNN
    final public static String face_detection_model_path = "/data/local/tmp/face_detection.dlc";
    final public static int face_detection_input_H = 160;
    final public static int face_detection_input_W = 128;
    final public static int face_detection_input_C = 3;
    final public static String face_detection_model_op_out = "Transpose_419";
    final public static String face_detection_model_out = "bbox_det";

    final public static String landmark_detection_model_path = "/data/local/tmp/landmark_detection.dlc";
    final public static int landmark_detection_input_H = 112;
    final public static int landmark_detection_input_W = 112;
    final public static int landmark_detection_input_C = 3;
    final public static String landmark_detection_model_op_out = "Gemm_137";
    final public static String landmark_detection_model_out = "facial_landmark";

    final public static String gaze_estimation_model_path = "/data/local/tmp/gaze_estimation.dlc";
    final public static int gaze_estimation_face_input_H = 120;
    final public static int gaze_estimation_face_input_W = 120;
    final public static int gaze_estimation_face_input_C = 3;
    final public static int gaze_estimation_eye_input_H = 60;
    final public static int gaze_estimation_eye_input_W = 60;
    final public static int gaze_estimation_eye_input_C = 3;
    final public static String gaze_estimation_model_op_out = "Gemm_602";
    final public static String gaze_estimation_model_out = "gaze_pitchyaw";


    final public static int Preview_H = 480;
    final public static int Preview_W = 480;
    final public static int crop_W = 480;
    final public static int crop_H = 480;
    final public static int img_orientation = 90;


    public static int getFaceDetectionInputsize(){
        return face_detection_input_H * face_detection_input_W * face_detection_input_C;
    }

    public static int getLandmarkDetectionInputsize(){
        return landmark_detection_input_H * landmark_detection_input_W * landmark_detection_input_C;
    }
    public static int getGazeEstimationFaceInputsize(){
        return gaze_estimation_face_input_H * gaze_estimation_face_input_W * gaze_estimation_face_input_C;
    }
    public static int getGazeEstimationEyeInputsize(){
        return gaze_estimation_eye_input_H * gaze_estimation_eye_input_W * gaze_estimation_eye_input_C;
    }
}