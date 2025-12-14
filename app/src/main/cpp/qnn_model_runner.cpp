/*
 * QNN Model Runner - JNI wrapper for Qualcomm Neural Network (QNN) SDK
 * 
 * This file provides a Java-accessible interface for loading and running DLC files
 * using the QNN framework instead of SNPE.
 * 
 * Required QNN SDK libraries (place in app/src/main/jniLibs/arm64-v8a/):
 * - libQnnHtp.so (or libQnnDsp.so, libQnnGpu.so depending on backend)
 * - libQnnSystem.so
 * - HTP skel files (libQnnHtpV*Skel.so, libQnnHtpV*Stub.so)
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <dlfcn.h>
#include <cstring>
#include <fstream>

// QNN headers - from the QNN SDK (in include/QNN/)
#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnProperty.h"
#include "QNN/QnnMem.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemDlc.h"

#define LOG_TAG "QNNModelRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// QNN Backend types
enum class QnnBackendType {
    HTP = 0,  // Hexagon Tensor Processor (recommended for Snapdragon)
    GPU = 1,  // Adreno GPU
    DSP = 2,  // Hexagon DSP (legacy)
    CPU = 3   // CPU fallback
};

// Function pointer types for QNN interface
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

// Structure to hold a loaded QNN model
struct QnnModelContext {
    // Library handles
    void* backendLibHandle = nullptr;
    void* systemLibHandle = nullptr;
    
    // QNN handles
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_ContextHandle_t contextHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    QnnSystemDlc_Handle_t dlcHandle = nullptr;
    
    // Interface pointers
    const QnnInterface_t* qnnInterface = nullptr;
    const QnnSystemInterface_t* systemInterface = nullptr;
    
    // Graph info
    QnnSystemContext_GraphInfo_t* graphsInfo = nullptr;
    uint32_t numGraphs = 0;
    Qnn_GraphHandle_t graphHandle = nullptr;
    
    // Tensor info
    std::vector<Qnn_Tensor_t> inputTensors;
    std::vector<Qnn_Tensor_t> outputTensors;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;
    
    // Model info
    std::string modelPath;
    std::string outputLayerName;
    QnnBackendType backendType;
    
    bool isInitialized = false;
};

// Global map to store model contexts (handle -> context)
static std::map<jlong, std::unique_ptr<QnnModelContext>> g_modelContexts;
static jlong g_nextHandle = 1;

// Get backend library name
static const char* getBackendLibName(QnnBackendType type) {
    switch (type) {
        case QnnBackendType::HTP: return "libQnnHtp.so";
        case QnnBackendType::GPU: return "libQnnGpu.so";
        case QnnBackendType::DSP: return "libQnnDsp.so";
        case QnnBackendType::CPU: return "libQnnCpu.so";
        default: return nullptr;
    }
}

// Load a shared library
static void* loadLibrary(const char* libName, std::string& errorMsg) {
    void* handle = dlopen(libName, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        errorMsg = std::string("Failed to load ") + libName + ": " + dlerror();
        LOGE("%s", errorMsg.c_str());
    } else {
        LOGI("✓ Loaded library: %s", libName);
    }
    return handle;
}

// Get QNN interface from backend library
static const QnnInterface_t* getQnnInterface(void* libHandle, std::string& errorMsg) {
    auto getProviders = (QnnInterfaceGetProvidersFn_t)dlsym(libHandle, "QnnInterface_getProviders");
    if (!getProviders) {
        errorMsg = "QnnInterface_getProviders not found: " + std::string(dlerror());
        return nullptr;
    }
    
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    
    if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
        errorMsg = "Failed to get QNN providers";
        return nullptr;
    }
    
    LOGI("Got QNN interface, version: %d.%d.%d", 
         providers[0]->apiVersion.coreApiVersion.major,
         providers[0]->apiVersion.coreApiVersion.minor,
         providers[0]->apiVersion.coreApiVersion.patch);
    
    return providers[0];
}

// Get QNN System interface
static const QnnSystemInterface_t* getQnnSystemInterface(void* libHandle, std::string& errorMsg) {
    auto getProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(libHandle, "QnnSystemInterface_getProviders");
    if (!getProviders) {
        errorMsg = "QnnSystemInterface_getProviders not found: " + std::string(dlerror());
        return nullptr;
    }
    
    const QnnSystemInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    
    if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
        errorMsg = "Failed to get QNN System providers";
        return nullptr;
    }
    
    LOGI("Got QNN System interface");
    return providers[0];
}

// Cleanup function
static void cleanupContext(QnnModelContext* ctx) {
    if (!ctx) return;
    
    // Free graph info
    if (ctx->graphsInfo) {
        free(ctx->graphsInfo);
        ctx->graphsInfo = nullptr;
    }
    
    // Free DLC handle
    if (ctx->dlcHandle && ctx->systemInterface) {
        ctx->systemInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcFree(ctx->dlcHandle);
        ctx->dlcHandle = nullptr;
    }
    
    // Free context
    if (ctx->contextHandle && ctx->qnnInterface) {
        ctx->qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(ctx->contextHandle, nullptr);
        ctx->contextHandle = nullptr;
    }
    
    // Free device
    if (ctx->deviceHandle && ctx->qnnInterface) {
        ctx->qnnInterface->QNN_INTERFACE_VER_NAME.deviceFree(ctx->deviceHandle);
        ctx->deviceHandle = nullptr;
    }
    
    // Free backend
    if (ctx->backendHandle && ctx->qnnInterface) {
        ctx->qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(ctx->backendHandle);
        ctx->backendHandle = nullptr;
    }
    
    // Close libraries
    if (ctx->systemLibHandle) {
        dlclose(ctx->systemLibHandle);
        ctx->systemLibHandle = nullptr;
    }
    if (ctx->backendLibHandle) {
        dlclose(ctx->backendLibHandle);
        ctx->backendLibHandle = nullptr;
    }
    
    ctx->isInitialized = false;
}

extern "C" {

/*
 * Check if QNN is available on this device
 */
JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeIsQnnAvailable(
        JNIEnv* env,
        jclass clazz) {
    
    // Try to load HTP backend (most common on Snapdragon)
    void* handle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        dlclose(handle);
        LOGI("✓ QNN HTP backend is available");
        return JNI_TRUE;
    }
    
    // Try GPU backend
    handle = dlopen("libQnnGpu.so", RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        dlclose(handle);
        LOGI("✓ QNN GPU backend is available");
        return JNI_TRUE;
    }
    
    // Try System library
    handle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        dlclose(handle);
        LOGI("✓ QNN System library is available");
        return JNI_TRUE;
    }
    
    LOGW("No QNN backend available");
    return JNI_FALSE;
}

/*
 * Initialize QNN and load a DLC model
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeLoadModel(
        JNIEnv* env,
        jobject thiz,
        jstring modelPath,
        jstring outputLayerName,
        jint backendType) {
    
    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    const char* outputLayerStr = env->GetStringUTFChars(outputLayerName, nullptr);
    
    LOGI("════════════════════════════════════════");
    LOGI("Loading QNN model: %s", modelPathStr);
    LOGI("Output layer: %s", outputLayerStr);
    LOGI("Requested backend: %d", backendType);
    LOGI("════════════════════════════════════════");
    
    auto ctx = std::make_unique<QnnModelContext>();
    ctx->modelPath = modelPathStr;
    ctx->outputLayerName = outputLayerStr;
    ctx->backendType = static_cast<QnnBackendType>(backendType);
    
    env->ReleaseStringUTFChars(modelPath, modelPathStr);
    env->ReleaseStringUTFChars(outputLayerName, outputLayerStr);
    
    std::string errorMsg;
    Qnn_ErrorHandle_t error;
    
    // Step 1: Load QNN System library first (needed for all backends)
    ctx->systemLibHandle = loadLibrary("libQnnSystem.so", errorMsg);
    if (!ctx->systemLibHandle) {
        LOGE("Failed to load libQnnSystem.so: %s", errorMsg.c_str());
        cleanupContext(ctx.get());
        return -1;
    }
    
    ctx->systemInterface = getQnnSystemInterface(ctx->systemLibHandle, errorMsg);
    if (!ctx->systemInterface) {
        LOGE("Failed to get QNN System interface: %s", errorMsg.c_str());
        cleanupContext(ctx.get());
        return -1;
    }
    
    // Step 2: Try each backend until one works
    // IMPORTANT: HTP/DSP backends can cause SIGSEGV crashes on some devices during initialization
    // due to missing system libraries or incompatible DSP firmware. These crashes cannot be caught.
    // Therefore, we ONLY use GPU and CPU backends which are safe and stable.
    QnnBackendType backendsToTry[] = {
        QnnBackendType::GPU,  // GPU - best performance, safe on Qualcomm devices
        QnnBackendType::CPU   // CPU - always works, guaranteed fallback
    };
    
    // Log that we're skipping HTP/DSP for safety
    if (ctx->backendType == QnnBackendType::HTP || ctx->backendType == QnnBackendType::DSP) {
        LOGW("Requested HTP/DSP backend is skipped - these can crash on some devices");
        LOGW("Using GPU/CPU backends instead for stability");
    }
    
    bool backendInitialized = false;
    
    for (auto type : backendsToTry) {
        const char* libName = getBackendLibName(type);
        if (!libName) continue;
        
        LOGI("Trying backend: %s", libName);
        
        // Clean up previous failed attempt
        if (ctx->contextHandle) {
            ctx->qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(ctx->contextHandle, nullptr);
            ctx->contextHandle = nullptr;
        }
        if (ctx->backendHandle) {
            ctx->qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(ctx->backendHandle);
            ctx->backendHandle = nullptr;
        }
        if (ctx->backendLibHandle) {
            dlclose(ctx->backendLibHandle);
            ctx->backendLibHandle = nullptr;
        }
        ctx->qnnInterface = nullptr;
        
        // Try to load this backend
        ctx->backendLibHandle = loadLibrary(libName, errorMsg);
        if (!ctx->backendLibHandle) {
            LOGW("  Backend library not available: %s", libName);
            continue;
        }
        
        ctx->qnnInterface = getQnnInterface(ctx->backendLibHandle, errorMsg);
        if (!ctx->qnnInterface) {
            LOGW("  Failed to get interface from: %s", libName);
            continue;
        }
        
        // Try to create backend
        error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.backendCreate(
            nullptr,  // logger
            nullptr,  // config
            &ctx->backendHandle
        );
        if (error != QNN_SUCCESS) {
            LOGW("  Backend creation failed for %s: %lu", libName, (unsigned long)error);
            continue;
        }
        
        // Try to create context
        error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.contextCreate(
            ctx->backendHandle,
            ctx->deviceHandle,
            nullptr,  // config
            &ctx->contextHandle
        );
        if (error != QNN_SUCCESS) {
            LOGW("  Context creation failed for %s: %lu", libName, (unsigned long)error);
            // Clean up backend before trying next
            ctx->qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(ctx->backendHandle);
            ctx->backendHandle = nullptr;
            continue;
        }
        
        // Success!
        ctx->backendType = type;
        backendInitialized = true;
        LOGI("✓ Backend initialized: %s", libName);
        break;
    }
    
    if (!backendInitialized) {
        LOGE("Failed to initialize any QNN backend");
        cleanupContext(ctx.get());
        return -1;
    }
    
    LOGI("✓ Context created");
    
    // Step 6: Load DLC file
    error = ctx->systemInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcCreateFromFile(
        nullptr,  // logger
        ctx->modelPath.c_str(),
        &ctx->dlcHandle
    );
    if (error != QNN_SUCCESS) {
        LOGE("Failed to load DLC file: %lu", (unsigned long)error);
        cleanupContext(ctx.get());
        return -1;
    }
    LOGI("✓ DLC loaded: %s", ctx->modelPath.c_str());
    
    // Step 7: Compose graphs from DLC
    error = ctx->systemInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcComposeGraphs(
        ctx->dlcHandle,
        nullptr,  // graphConfigs
        0,        // numGraphConfigs
        ctx->backendHandle,
        ctx->contextHandle,
        *ctx->qnnInterface,
        QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1,
        &ctx->graphsInfo,
        &ctx->numGraphs
    );
    if (error != QNN_SUCCESS) {
        LOGE("Failed to compose graphs from DLC: %lu", (unsigned long)error);
        cleanupContext(ctx.get());
        return -1;
    }
    LOGI("✓ Composed %u graphs from DLC", ctx->numGraphs);
    
    if (ctx->numGraphs == 0) {
        LOGE("No graphs found in DLC");
        cleanupContext(ctx.get());
        return -1;
    }
    
    // Get graph info based on version
    const char* graphName = nullptr;
    if (ctx->graphsInfo[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        auto& v1 = ctx->graphsInfo[0].graphInfoV1;
        graphName = v1.graphName;
        ctx->numInputs = v1.numGraphInputs;
        ctx->numOutputs = v1.numGraphOutputs;
        
        LOGI("Graph: %s (V1), inputs: %u, outputs: %u", graphName, ctx->numInputs, ctx->numOutputs);
    } else if (ctx->graphsInfo[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        auto& v2 = ctx->graphsInfo[0].graphInfoV2;
        graphName = v2.graphName;
        ctx->numInputs = v2.numGraphInputs;
        ctx->numOutputs = v2.numGraphOutputs;
        
        LOGI("Graph: %s (V2), inputs: %u, outputs: %u", graphName, ctx->numInputs, ctx->numOutputs);
    } else {
        LOGE("Unknown graph info version: %d", ctx->graphsInfo[0].version);
        cleanupContext(ctx.get());
        return -1;
    }
    
    // Step 8: Retrieve graph handle by name
    error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve(
        ctx->contextHandle,
        graphName,
        &ctx->graphHandle
    );
    if (error != QNN_SUCCESS) {
        LOGE("Failed to retrieve graph handle: %lu", (unsigned long)error);
        cleanupContext(ctx.get());
        return -1;
    }
    LOGI("✓ Graph handle retrieved");
    
    // Step 9: Finalize the graph
    error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize(
        ctx->graphHandle,
        nullptr,  // profile
        nullptr   // signal
    );
    if (error != QNN_SUCCESS) {
        LOGE("Failed to finalize graph: %lu", (unsigned long)error);
        cleanupContext(ctx.get());
        return -1;
    }
    LOGI("✓ Graph finalized");
    
    // Mark as initialized
    ctx->isInitialized = true;
    
    jlong handle = g_nextHandle++;
    g_modelContexts[handle] = std::move(ctx);
    
    LOGI("════════════════════════════════════════");
    LOGI("✓ QNN model loaded successfully, handle: %ld", (long)handle);
    LOGI("════════════════════════════════════════");
    
    return handle;
}

/*
 * Run inference with a single input
 */
JNIEXPORT jfloatArray JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeRunInference(
        JNIEnv* env,
        jobject thiz,
        jlong handle,
        jstring inputName,
        jfloatArray inputData,
        jintArray inputDims) {
    
    auto it = g_modelContexts.find(handle);
    if (it == g_modelContexts.end() || !it->second->isInitialized) {
        LOGE("Invalid model handle: %ld", (long)handle);
        return nullptr;
    }
    
    QnnModelContext* ctx = it->second.get();
    
    const char* inputNameStr = env->GetStringUTFChars(inputName, nullptr);
    jfloat* inputPtr = env->GetFloatArrayElements(inputData, nullptr);
    jint* dimsPtr = env->GetIntArrayElements(inputDims, nullptr);
    jsize inputLen = env->GetArrayLength(inputData);
    (void)env->GetArrayLength(inputDims);  // Unused, dimensions come from model
    
    LOGD("Running inference: input=%s, size=%d", inputNameStr, inputLen);
    
    // Get graph input/output info
    auto& v1 = ctx->graphsInfo[0].graphInfoV1;
    uint32_t numInputs = v1.numGraphInputs;
    uint32_t numOutputs = v1.numGraphOutputs;
    
    LOGD("Graph has %u inputs, %u outputs", numInputs, numOutputs);
    
    // Create input tensors (for now just one, matching the first input)
    std::vector<Qnn_Tensor_t> inputTensors(numInputs);
    for (uint32_t i = 0; i < numInputs; i++) {
        memset(&inputTensors[i], 0, sizeof(Qnn_Tensor_t));
        inputTensors[i].version = QNN_TENSOR_VERSION_1;
        inputTensors[i].v1.id = v1.graphInputs[i].v1.id;
        inputTensors[i].v1.name = v1.graphInputs[i].v1.name;
        inputTensors[i].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        inputTensors[i].v1.dataFormat = v1.graphInputs[i].v1.dataFormat;
        inputTensors[i].v1.dataType = v1.graphInputs[i].v1.dataType;
        inputTensors[i].v1.quantizeParams = v1.graphInputs[i].v1.quantizeParams;
        inputTensors[i].v1.rank = v1.graphInputs[i].v1.rank;
        inputTensors[i].v1.dimensions = v1.graphInputs[i].v1.dimensions;
        inputTensors[i].v1.memType = QNN_TENSORMEMTYPE_RAW;
        
        if (i == 0) {
            // First input uses the provided data
            inputTensors[i].v1.clientBuf.data = inputPtr;
            inputTensors[i].v1.clientBuf.dataSize = inputLen * sizeof(float);
        }
        
        LOGD("Input[%u]: name=%s, rank=%u, dataType=%d", 
             i, inputTensors[i].v1.name, inputTensors[i].v1.rank, inputTensors[i].v1.dataType);
    }
    
    // Create output tensors for ALL outputs
    std::vector<Qnn_Tensor_t> outputTensors(numOutputs);
    std::vector<std::vector<float>> outputBuffers(numOutputs);
    uint32_t primaryOutputSize = 0;
    int primaryOutputIdx = 0;
    
    for (uint32_t i = 0; i < numOutputs; i++) {
        memset(&outputTensors[i], 0, sizeof(Qnn_Tensor_t));
        outputTensors[i].version = QNN_TENSOR_VERSION_1;
        outputTensors[i].v1.id = v1.graphOutputs[i].v1.id;
        outputTensors[i].v1.name = v1.graphOutputs[i].v1.name;
        outputTensors[i].v1.type = QNN_TENSOR_TYPE_APP_READ;
        outputTensors[i].v1.dataFormat = v1.graphOutputs[i].v1.dataFormat;
        outputTensors[i].v1.dataType = v1.graphOutputs[i].v1.dataType;
        outputTensors[i].v1.quantizeParams = v1.graphOutputs[i].v1.quantizeParams;
        outputTensors[i].v1.rank = v1.graphOutputs[i].v1.rank;
        outputTensors[i].v1.dimensions = v1.graphOutputs[i].v1.dimensions;
        outputTensors[i].v1.memType = QNN_TENSORMEMTYPE_RAW;
        
        // Calculate size for this output
        uint32_t outputSize = 1;
        for (uint32_t j = 0; j < outputTensors[i].v1.rank; j++) {
            outputSize *= outputTensors[i].v1.dimensions[j];
        }
        
        outputBuffers[i].resize(outputSize);
        outputTensors[i].v1.clientBuf.data = outputBuffers[i].data();
        outputTensors[i].v1.clientBuf.dataSize = outputSize * sizeof(float);
        
        LOGD("Output[%u]: name=%s, rank=%u, size=%u", 
             i, outputTensors[i].v1.name, outputTensors[i].v1.rank, outputSize);
        
        // Track the output that matches our requested output layer
        if (strcmp(outputTensors[i].v1.name, ctx->outputLayerName.c_str()) == 0) {
            primaryOutputIdx = i;
            primaryOutputSize = outputSize;
        } else if (i == 0) {
            // Default to first output
            primaryOutputIdx = 0;
            primaryOutputSize = outputSize;
        }
    }
    
    // Execute graph with ALL inputs and outputs
    Qnn_ErrorHandle_t error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute(
        ctx->graphHandle,
        inputTensors.data(), numInputs,
        outputTensors.data(), numOutputs,
        nullptr,  // profile
        nullptr   // signal
    );
    
    env->ReleaseStringUTFChars(inputName, inputNameStr);
    env->ReleaseFloatArrayElements(inputData, inputPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(inputDims, dimsPtr, JNI_ABORT);
    
    if (error != QNN_SUCCESS) {
        LOGE("Graph execution failed: %lu", (unsigned long)error);
        return nullptr;
    }
    
    LOGD("Inference successful, returning output[%d] with size %u", primaryOutputIdx, primaryOutputSize);
    
    // Create output array from the primary (requested) output
    jfloatArray result = env->NewFloatArray(primaryOutputSize);
    env->SetFloatArrayRegion(result, 0, primaryOutputSize, outputBuffers[primaryOutputIdx].data());
    
    return result;
}

/*
 * Run inference with multiple inputs
 */
JNIEXPORT jfloatArray JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeRunMultiInputInference(
        JNIEnv* env,
        jobject thiz,
        jlong handle,
        jobjectArray inputNames,
        jobjectArray inputDataArrays,
        jobjectArray inputDimsArrays) {
    
    auto it = g_modelContexts.find(handle);
    if (it == g_modelContexts.end() || !it->second->isInitialized) {
        LOGE("Invalid model handle: %ld", (long)handle);
        return nullptr;
    }
    
    QnnModelContext* ctx = it->second.get();
    int numInputs = env->GetArrayLength(inputNames);
    
    LOGD("Running multi-input inference with %d inputs", numInputs);
    
    auto& v1 = ctx->graphsInfo[0].graphInfoV1;
    
    // Prepare input tensors
    std::vector<Qnn_Tensor_t> inputs(numInputs);
    std::vector<jfloat*> inputPtrs(numInputs);
    std::vector<jfloatArray> inputArrays(numInputs);
    
    for (int i = 0; i < numInputs; i++) {
        inputArrays[i] = (jfloatArray)env->GetObjectArrayElement(inputDataArrays, i);
        inputPtrs[i] = env->GetFloatArrayElements(inputArrays[i], nullptr);
        jsize inputLen = env->GetArrayLength(inputArrays[i]);
        
        inputs[i] = v1.graphInputs[i];  // Copy tensor template
        inputs[i].v1.clientBuf.data = inputPtrs[i];
        inputs[i].v1.clientBuf.dataSize = inputLen * sizeof(float);
    }
    
    // Prepare output tensor
    Qnn_Tensor_t outputTensor = v1.graphOutputs[0];
    uint32_t outputSize = 1;
    for (uint32_t i = 0; i < outputTensor.v1.rank; i++) {
        outputSize *= outputTensor.v1.dimensions[i];
    }
    
    std::vector<float> outputBuffer(outputSize);
    outputTensor.v1.clientBuf.data = outputBuffer.data();
    outputTensor.v1.clientBuf.dataSize = outputSize * sizeof(float);
    
    // Execute
    Qnn_ErrorHandle_t error = ctx->qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute(
        ctx->graphHandle,
        inputs.data(), numInputs,
        &outputTensor, 1,
        nullptr,
        nullptr
    );
    
    // Cleanup
    for (int i = 0; i < numInputs; i++) {
        env->ReleaseFloatArrayElements(inputArrays[i], inputPtrs[i], JNI_ABORT);
    }
    
    if (error != QNN_SUCCESS) {
        LOGE("Multi-input graph execution failed: %lu", (unsigned long)error);
        return nullptr;
    }
    
    // Return result
    jfloatArray result = env->NewFloatArray(outputSize);
    env->SetFloatArrayRegion(result, 0, outputSize, outputBuffer.data());
    return result;
}

/*
 * Release a loaded model
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeReleaseModel(
        JNIEnv* env,
        jobject thiz,
        jlong handle) {
    
    auto it = g_modelContexts.find(handle);
    if (it == g_modelContexts.end()) {
        LOGW("Trying to release invalid handle: %ld", (long)handle);
        return;
    }
    
    cleanupContext(it->second.get());
    g_modelContexts.erase(it);
    LOGI("✓ Released QNN model handle: %ld", (long)handle);
}

/*
 * Get backend info string
 */
JNIEXPORT jstring JNICALL
Java_org_tensorflow_lite_examples_gaze_1estimation_QNNModelRunner_nativeGetBackendInfo(
        JNIEnv* env,
        jobject thiz,
        jlong handle) {
    
    auto it = g_modelContexts.find(handle);
    if (it == g_modelContexts.end()) {
        return env->NewStringUTF("Invalid handle");
    }
    
    QnnModelContext* ctx = it->second.get();
    
    const char* backendName = "Unknown";
    switch (ctx->backendType) {
        case QnnBackendType::HTP: backendName = "QNN HTP (Hexagon Tensor Processor)"; break;
        case QnnBackendType::GPU: backendName = "QNN GPU (Adreno)"; break;
        case QnnBackendType::DSP: backendName = "QNN DSP (Hexagon)"; break;
        case QnnBackendType::CPU: backendName = "QNN CPU"; break;
    }
    
    return env->NewStringUTF(backendName);
}

} // extern "C"
