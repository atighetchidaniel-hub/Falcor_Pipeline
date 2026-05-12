#pragma once
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/RasterPass.h"
#include "Core/Pass/ComputePass.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace Falcor;

class NeuralPVSExporter : public SampleApp
{
public:
    NeuralPVSExporter(const SampleAppConfig& config);
    ~NeuralPVSExporter();

    void onLoad(RenderContext* pRenderContext) override;
    void onShutdown() override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onHotReload(HotReloadFlags reloaded) override;

private:
    struct ExportSample
    {
        float3 center = float3(0.f);
        float3 forward = float3(0.f, 0.f, -1.f);
        float fovYDegrees = 60.f;
        bool hasCamera = false;
    };

    void loadScene(const std::filesystem::path& path);
    void createResources();
    void createPreviewPass();
    void createGVPass();
    void createPVVPass();
    void renderPreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void loadPreviewPath();
    void applyPreviewSample();
    void exportSceneVolumes(RenderContext* pRenderContext);
    void writeVolumePair(
        const std::vector<uint8_t>& gvBytes,
        const std::vector<uint8_t>& pvvBytes,
        const std::string& datasetName,
        uint32_t index
    );
    void writeVolumeFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes);
    void writeDebugProjections(
        const std::filesystem::path& datasetRoot,
        const std::vector<uint8_t>& bytes,
        const std::string& label,
        uint32_t index
    );
    std::vector<ExportSample> loadPathSamples(const std::filesystem::path& path) const;

    uint32_t mVolumeSize = 256;
    uint32_t mVolumeDepth = 256;
    uint32_t mRasterWidth = 2048;
    uint32_t mRasterHeight = 2048;
    uint32_t mExportIndex = 0;

    std::string mScenePathText = "media/Robolab/Robolab.fbx";
    std::string mOutputRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets";
    std::string mDatasetName = "falcor_robolab";

    std::filesystem::path mScenePath = mScenePathText;
    std::filesystem::path mOutputRoot = mOutputRootText;

    uint32_t mSamplesPerAxis = 5;
    float mVolumeExtentScale = 0.65f;
    float mSampleStepScale = 0.05f;
    uint32_t mExportMode = 2; // 0 = GV only, 1 = PVV only, 2 = GV + PVV, 3 = metadata only
    bool mWriteDebugProjections = false;

    uint32_t mSamplingMode = 0; // 0 = grid, 1 = path CSV
    std::string mPathCsvText = "C:/dev/Falcor/neuralpvs_paths/robolab_path.csv";
    uint32_t mVisibilityMode = 0; // 0 = view cell, 1 = camera frustum
    float mCameraAspectRatio = 1.777778f;

    bool mRenderScenePreview = true;
    bool mPreviewPlayback = false;
    float mPreviewFps = 12.f;
    uint32_t mPreviewSampleIndex = 0;
    double mPreviewAccumulator = 0.0;
    std::vector<ExportSample> mPreviewSamples;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    ref<RasterPass> mpPreviewPass;
    ref<RasterPass> mpGVPass;
    ref<ComputePass> mpPVVPass;
    ref<Texture> mpGVVolume;
    ref<Texture> mpPVVVolume;
    ref<Fbo> mpGVFbo;

    std::string mLastExportStatus = "Not exported yet.";
};
