#pragma once
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/FullScreenPass.h"
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

    struct VolumeProjectionParams
    {
        float3 viewCellPosition = float3(0.f);
        float3 forward = float3(0.f, 0.f, -1.f);
        float3 right = float3(1.f, 0.f, 0.f);
        float3 up = float3(0.f, 1.f, 0.f);
        float nearPlane = 0.3f;
        float farPlane = 30.f;
        float tanHalfFovX = 1.f;
        float tanHalfFovY = 1.f;
    };

    void loadScene(const std::filesystem::path& path);
    void createResources();
    void createPreviewPass();
    void createGVPass();
    void createPVVPass();
    void renderPreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void createPVVRenderPass();
    void renderPVVOverlay(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void loadPreviewPath();
    void applyPreviewSample();
    void loadRenderMetadata();
    void loadRenderVolume();
    void ensureRenderVolumeLoaded(bool forceReload);
    void startRenderPVVMode();
    std::filesystem::path resolveRenderVolumePath() const;
    std::filesystem::path getRenderFrameOutputPath() const;
    std::filesystem::path getRenderFrameStagingPath() const;
    std::filesystem::path getRenderVideoOutputPath() const;
    void captureRenderFrame(const ref<Fbo>& pTargetFbo);
    void finishRenderPVVMode();
    void encodeRenderVideo();
    void startProgressiveExport();
    void processProgressiveExportSample(RenderContext* pRenderContext);
    void finishProgressiveExport();
    void exportSceneVolumes(RenderContext* pRenderContext);
    std::vector<ExportSample> buildExportSamples(const float3& sceneCenter, const float3& sceneExtent) const;
    void validateExportSamples(const std::vector<ExportSample>& samples, bool useCameraFrustum) const;
    VolumeProjectionParams makeVolumeProjection(const ExportSample& sample, float viewCellRadius, float nearPlane, float farPlane) const;
    void exportOneSample(
        RenderContext* pRenderContext,
        const ExportSample& sample,
        uint32_t index,
        const float3& volumeExtent,
        float viewCellRadius,
        uint64_t& gvBitCount,
        uint64_t& pvvBitCount
    );
    void writeExportMetadata(
        const std::vector<ExportSample>& samples,
        const std::vector<uint64_t>& gvBitCounts,
        const std::vector<uint64_t>& pvvBitCounts,
        const AABB& sceneBounds,
        const float3& sceneCenter,
        const float3& sceneExtent,
        const float3& volumeExtent,
        const float3& sampleStep,
        bool useCameraFrustum
    );
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
    std::vector<uint8_t> readGzipStoredFile(const std::filesystem::path& path) const;

    uint32_t mVolumeSize = 256;
    uint32_t mVolumeDepth = 256;
    uint32_t mRasterWidth = 2048;
    uint32_t mRasterHeight = 2048;
    uint32_t mExportIndex = 0;

    std::string mScenePathText = "media/RobolabUSD/Robolab.usda";
    std::string mOutputRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets";
    std::string mDatasetName = "falcor_robolab_usd_frustum_128";

    std::filesystem::path mScenePath = mScenePathText;
    std::filesystem::path mOutputRoot = mOutputRootText;

    uint32_t mSamplesPerAxis = 5;
    float mVolumeExtentScale = 0.65f;
    float mSampleStepScale = 0.05f;
    uint32_t mExportMode = 2; // 0 = Generate GV, 1 = Generate PVV, 2 = Generate GV + PVV, 3 = Render PVV
    bool mWriteDebugProjections = false;

    uint32_t mSamplingMode = 1; // 0 = grid, 1 = path CSV
    std::string mPathCsvText = "C:/dev/Falcor/neuralpvs_paths/robolab_animated_camera_path_usd_zflip.csv";
    uint32_t mVisibilityMode = 1; // 0 = view cell, 1 = camera frustum
    float mCameraAspectRatio = 1.777778f;
    uint32_t mVolumeMappingMode = 1; // 0 = world AABB, 1 = Unity view-cell projection
    float mViewCellRadius = 0.3f;
    float mViewCellNearPlane = 0.3f;
    float mViewCellFarPlane = 30.0f;

    bool mRenderScenePreview = true;
    bool mPreviewPlayback = false;
    float mPreviewFps = 12.f;
    uint32_t mPreviewSampleIndex = 0;
    double mPreviewAccumulator = 0.0;
    std::vector<ExportSample> mPreviewSamples;

    std::string mRenderDatasetRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets/falcor_robolab_usd_frustum_128";
    std::string mPredictedPVVRootText = "C:/dev/Falcor/neuralpvs_predictions/falcor_robolab_usd_frustum_128";
    uint32_t mRenderVolumeSource = 0; // 0 = predicted PVV, 1 = dataset PVV, 2 = dataset GV
    uint32_t mRenderVolumeKind = 1; // 0 = GV, 1 = PVV, used by debug overlay coloring
    uint32_t mRenderPVVFilter = 3; // 1 = exact, 2 = box, 3 = conservative trilinear
    uint32_t mRenderSampleIndex = 0;
    uint32_t mRenderVolumeSize = 256;
    uint32_t mRenderVolumeDepth = 256;
    uint32_t mRenderVolumeMappingMode = 0; // Read from metadata. Missing metadata means legacy world AABB.
    float mRenderViewCellRadius = 0.3f;
    float mRenderViewCellNearPlane = 0.3f;
    float mRenderViewCellFarPlane = 30.0f;
    bool mRenderExportFrames = false;
    uint32_t mRenderFrameExportMode = 1; // 0 = PNG image sequence, 1 = lossless MKV video.
    bool mRenderPVVCullScene = false;
    bool mRenderKeepOutsidePVVInView = true;
    bool mRenderPVVOverlay = false;
    bool mRenderUseSampleCamera = true;
    float mRenderOpacity = 0.55f;
    float mRenderStepScale = 0.75f;
    float3 mRenderVolumeExtent = float3(1.f);
    std::filesystem::path mRenderDatasetRoot;
    std::vector<ExportSample> mRenderSamples;
    std::string mRenderStatus = "No render volume loaded.";
    uint32_t mRenderLoadedSampleIndex = 0xffffffffu;
    uint32_t mRenderLoadedVolumeSource = 0xffffffffu;
    uint32_t mRenderLastCapturedSampleIndex = 0xffffffffu;
    uint32_t mRenderCapturedFrameCount = 0;
    bool mRenderPVVFinished = false;
    std::filesystem::path mRenderLoadedVolumePath;

    bool mProgressiveExportActive = false;
    uint32_t mProgressiveExportIndex = 0;
    bool mProgressiveUseCameraFrustum = false;
    AABB mProgressiveSceneBounds;
    float3 mProgressiveSceneCenter = float3(0.f);
    float3 mProgressiveSceneExtent = float3(0.f);
    float3 mProgressiveVolumeExtent = float3(0.f);
    float3 mProgressiveSampleStep = float3(0.f);
    float mProgressiveViewCellRadius = 0.f;
    std::vector<ExportSample> mProgressiveExportSamples;
    std::vector<uint64_t> mProgressiveGVBitCounts;
    std::vector<uint64_t> mProgressivePVVBitCounts;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    ref<RasterPass> mpPreviewPass;
    ref<RasterPass> mpGVPass;
    ref<ComputePass> mpPVVPass;
    ref<FullScreenPass> mpPVVRenderPass;
    ref<Texture> mpGVVolume;
    ref<Texture> mpPVVVolume;
    ref<Texture> mpRenderVolume;
    ref<Fbo> mpGVFbo;

    std::string mLastExportStatus = "Not exported yet.";
};
