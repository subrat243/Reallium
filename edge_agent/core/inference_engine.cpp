/*
 * Inference Engine Implementation
 */

#include "inference_engine.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace deepfake {

InferenceEngine::InferenceEngine(const std::string& model_path,
                                ModelType model_type,
                                int num_threads)
    : model_path_(model_path),
      model_type_(model_type),
      num_threads_(num_threads),
      is_initialized_(false) {}

InferenceEngine::~InferenceEngine() {
    // Cleanup handled by unique_ptr
}

bool InferenceEngine::Initialize() {
    // Load model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        std::cerr << "Failed to load model from: " << model_path_ << std::endl;
        return false;
    }
    
    // Build interpreter
    tflite::InterpreterBuilder builder(*model_, resolver_);
    builder(&interpreter_);
    
    if (!interpreter_) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return false;
    }
    
    // Set number of threads
    interpreter_->SetNumThreads(num_threads_);
    
    // Allocate tensors
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return false;
    }
    
    is_initialized_ = true;
    std::cout << "Inference engine initialized successfully" << std::endl;
    std::cout << "Model: " << model_path_ << std::endl;
    std::cout << "Threads: " << num_threads_ << std::endl;
    
    return true;
}

InferenceResult InferenceEngine::InferAudio(const std::vector<float>& audio_data,
                                           int sample_rate,
                                           float threshold) {
    if (!is_initialized_) {
        throw std::runtime_error("Inference engine not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocess audio
    auto preprocessed = PreprocessAudio(audio_data);
    
    // Get input tensor
    int input_idx = interpreter_->inputs()[0];
    TfLiteTensor* input_tensor = interpreter_->tensor(input_idx);
    
    // Copy data to input tensor
    float* input_data = interpreter_->typed_input_tensor<float>(0);
    std::copy(preprocessed.begin(), preprocessed.end(), input_data);
    
    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Inference failed");
    }
    
    // Get output
    int output_idx = interpreter_->outputs()[0];
    TfLiteTensor* output_tensor = interpreter_->tensor(output_idx);
    float* output_data = interpreter_->typed_output_tensor<float>(0);
    
    float score = output_data[0];
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    InferenceResult result;
    result.authenticity_score = score;
    result.is_authentic = score >= threshold;
    result.is_deepfake = score < threshold;
    result.confidence = std::abs(score - 0.5f) * 2.0f;
    result.inference_time_ms = duration.count();
    
    return result;
}

InferenceResult InferenceEngine::InferVideo(const std::vector<float>& frames,
                                           int num_frames,
                                           int height,
                                           int width,
                                           int channels,
                                           float threshold) {
    if (!is_initialized_) {
        throw std::runtime_error("Inference engine not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocess video
    auto preprocessed = PreprocessVideo(frames, num_frames, height, width, channels);
    
    // Get input tensor
    float* input_data = interpreter_->typed_input_tensor<float>(0);
    std::copy(preprocessed.begin(), preprocessed.end(), input_data);
    
    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Inference failed");
    }
    
    // Get output
    float* output_data = interpreter_->typed_output_tensor<float>(0);
    float score = output_data[0];
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    InferenceResult result;
    result.authenticity_score = score;
    result.is_authentic = score >= threshold;
    result.is_deepfake = score < threshold;
    result.confidence = std::abs(score - 0.5f) * 2.0f;
    result.inference_time_ms = duration.count();
    
    return result;
}

std::vector<float> InferenceEngine::PreprocessAudio(const std::vector<float>& audio_data) {
    std::vector<float> processed = audio_data;
    
    // Normalize audio
    NormalizeAudio(processed);
    
    return processed;
}

std::vector<float> InferenceEngine::PreprocessVideo(const std::vector<float>& frames,
                                                   int num_frames,
                                                   int height,
                                                   int width,
                                                   int channels) {
    std::vector<float> processed = frames;
    
    // Normalize to [0, 1]
    for (auto& val : processed) {
        val /= 255.0f;
    }
    
    // Normalize with ImageNet stats
    NormalizeVideo(processed);
    
    return processed;
}

void InferenceEngine::NormalizeAudio(std::vector<float>& audio) {
    // Compute mean and std
    float mean = 0.0f;
    for (float val : audio) {
        mean += val;
    }
    mean /= audio.size();
    
    float std_dev = 0.0f;
    for (float val : audio) {
        std_dev += (val - mean) * (val - mean);
    }
    std_dev = std::sqrt(std_dev / audio.size());
    
    // Normalize
    if (std_dev > 1e-6f) {
        for (auto& val : audio) {
            val = (val - mean) / std_dev;
        }
    }
}

void InferenceEngine::NormalizeVideo(std::vector<float>& frames) {
    // ImageNet normalization
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    
    int pixels = frames.size() / 3;
    
    for (int i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            frames[idx] = (frames[idx] - mean[c]) / std[c];
        }
    }
}

std::string InferenceEngine::GetModelInfo() const {
    if (!is_initialized_) {
        return "Model not initialized";
    }
    
    std::string info = "Model Information:\n";
    info += "Path: " + model_path_ + "\n";
    info += "Inputs: " + std::to_string(interpreter_->inputs().size()) + "\n";
    info += "Outputs: " + std::to_string(interpreter_->outputs().size()) + "\n";
    
    return info;
}

} // namespace deepfake
