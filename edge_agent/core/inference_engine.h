/*
 * Inference Engine for Edge Devices
 * 
 * Lightweight C++ inference engine using TensorFlow Lite.
 * Optimized for low-power, real-time deepfake detection.
 */

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <memory>
#include <string>
#include <vector>

namespace deepfake {

enum class ModelType {
    AUDIO,
    VIDEO,
    FUSION
};

struct InferenceResult {
    float authenticity_score;
    bool is_authentic;
    bool is_deepfake;
    float confidence;
    std::vector<float> attention_weights;
    int64_t inference_time_ms;
};

class InferenceEngine {
public:
    /**
     * Constructor
     * @param model_path Path to TFLite model file
     * @param model_type Type of model (audio/video/fusion)
     * @param num_threads Number of threads for inference
     */
    InferenceEngine(const std::string& model_path, 
                   ModelType model_type,
                   int num_threads = 4);
    
    ~InferenceEngine();
    
    /**
     * Initialize the inference engine
     * @return true if successful
     */
    bool Initialize();
    
    /**
     * Run inference on audio data
     * @param audio_data Audio waveform data
     * @param sample_rate Sample rate (must be 16kHz)
     * @param threshold Classification threshold
     * @return Inference result
     */
    InferenceResult InferAudio(const std::vector<float>& audio_data,
                              int sample_rate = 16000,
                              float threshold = 0.5f);
    
    /**
     * Run inference on video frames
     * @param frames Video frames (flattened array)
     * @param num_frames Number of frames
     * @param height Frame height
     * @param width Frame width
     * @param channels Number of channels (3 for RGB)
     * @param threshold Classification threshold
     * @return Inference result
     */
    InferenceResult InferVideo(const std::vector<float>& frames,
                              int num_frames,
                              int height,
                              int width,
                              int channels = 3,
                              float threshold = 0.5f);
    
    /**
     * Get model information
     * @return Model metadata
     */
    std::string GetModelInfo() const;
    
    /**
     * Check if model is loaded
     * @return true if model is ready
     */
    bool IsReady() const { return is_initialized_; }

private:
    std::string model_path_;
    ModelType model_type_;
    int num_threads_;
    bool is_initialized_;
    
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    tflite::ops::builtin::BuiltinOpResolver resolver_;
    
    /**
     * Preprocess audio data
     */
    std::vector<float> PreprocessAudio(const std::vector<float>& audio_data);
    
    /**
     * Preprocess video frames
     */
    std::vector<float> PreprocessVideo(const std::vector<float>& frames,
                                      int num_frames,
                                      int height,
                                      int width,
                                      int channels);
    
    /**
     * Normalize audio
     */
    void NormalizeAudio(std::vector<float>& audio);
    
    /**
     * Normalize video frames (ImageNet stats)
     */
    void NormalizeVideo(std::vector<float>& frames);
};

} // namespace deepfake

#endif // INFERENCE_ENGINE_H
