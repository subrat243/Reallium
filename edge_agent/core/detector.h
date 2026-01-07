/*
 * Detector - Main detection orchestrator
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include "inference_engine.h"
#include <string>
#include <vector>
#include <memory>

namespace deepfake {

struct DetectionConfig {
    float threshold = 0.5f;
    bool enable_caching = true;
    bool enable_explainability = false;
    int max_cache_size = 100;
};

class Detector {
public:
    /**
     * Constructor
     * @param audio_model_path Path to audio model
     * @param video_model_path Path to video model
     * @param config Detection configuration
     */
    Detector(const std::string& audio_model_path,
            const std::string& video_model_path,
            const DetectionConfig& config = DetectionConfig());
    
    ~Detector();
    
    /**
     * Initialize detector
     * @return true if successful
     */
    bool Initialize();
    
    /**
     * Detect deepfake in audio file
     * @param audio_path Path to audio file
     * @return Inference result
     */
    InferenceResult DetectAudio(const std::string& audio_path);
    
    /**
     * Detect deepfake in video file
     * @param video_path Path to video file
     * @return Inference result
     */
    InferenceResult DetectVideo(const std::string& video_path);
    
    /**
     * Detect deepfake in multimodal content
     * @param audio_path Path to audio
     * @param video_path Path to video
     * @return Combined inference result
     */
    InferenceResult DetectMultimodal(const std::string& audio_path,
                                     const std::string& video_path);
    
    /**
     * Clear result cache
     */
    void ClearCache();

private:
    std::unique_ptr<InferenceEngine> audio_engine_;
    std::unique_ptr<InferenceEngine> video_engine_;
    DetectionConfig config_;
    
    /**
     * Load audio file
     */
    std::vector<float> LoadAudio(const std::string& path);
    
    /**
     * Load video file
     */
    std::vector<float> LoadVideo(const std::string& path, int& num_frames);
    
    /**
     * Compute file hash for caching
     */
    std::string ComputeHash(const std::string& path);
};

} // namespace deepfake

#endif // DETECTOR_H
