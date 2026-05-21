#pragma once
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/FullScreenPass.h"
#include "Core/Pass/RasterPass.h"
#include "Core/Pass/ComputePass.h"

#include <cstdint>
#include <chrono>
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
        float3 right = float3(1.f, 0.f, 0.f);
        float3 up = float3(0.f, 1.f, 0.f);
        float fovYDegrees = 60.f;
        bool hasCamera = false;
        bool hasBasis = false;
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
        float fovYRadians = 1.04719755f;
    };

    void loadScene(const std::filesystem::path& path);
    void createResources();
    void createPreviewPass();
    void createGVPass();
    void createPVVDepthPass();
    void createPVVPass();
    void createPVVRayPass();
    void renderPreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void createPVVRenderPass();
    void renderPVVOverlay(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void loadPreviewPath();
    void applyPreviewSample();
    void loadRenderMetadata();
    void loadRenderVolume();
    bool ensureRenderVolumeLoaded(bool forceReload);
    bool isModeRunning() const;
    void startSelectedMode();
    void stopCurrentMode();
    void startRenderPVVMode();
    void useGeneratedRenderPaths();
    std::filesystem::path resolveRenderVolumePath() const;
    std::filesystem::path getRenderPredictedPVVRoot() const;
    std::filesystem::path getRenderFrameOutputPath() const;
    std::filesystem::path getRenderFrameStagingPath() const;
    std::filesystem::path getRenderVideoOutputPath() const;
    void captureRenderFrame(const ref<Fbo>& pTargetFbo);
    void finishRenderPVVMode();
    void encodeRenderVideo(const std::string& completionPrefix);
    void startProgressiveExport();
    void processProgressiveExportSample(RenderContext* pRenderContext);
    void finishProgressiveExport();
    void exportSceneVolumes(RenderContext* pRenderContext);
    std::vector<ExportSample> buildExportSamples(const float3& sceneCenter, const float3& sceneExtent) const;
    void validateExportSamples(const std::vector<ExportSample>& samples, bool useCameraFrustum) const;
    VolumeProjectionParams makeVolumeProjection(
        const ExportSample& sample,
        float viewCellRadius,
        float nearPlane,
        float farPlane,
        float fovExpansionDegrees
    ) const;
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
    void appendCurrentCameraToCapturedPath();
    void removeLastCapturedPathSample();
    void clearCapturedPath();
    void saveCapturedPathCsv();
    void useCapturedPathForExport();
    std::vector<ExportSample> resampleCapturedPath(uint32_t targetSampleCount) const;
    void writePathCsv(const std::filesystem::path& path, const std::vector<ExportSample>& samples) const;
    std::vector<uint8_t> readGzipStoredFile(const std::filesystem::path& path) const;

    uint32_t mVolumeSize = 256;
    uint32_t mVolumeDepth = 256;
    uint32_t mRasterWidth = 2048;
    uint32_t mRasterHeight = 2048;
    uint32_t mSamplingFactor = 2;
    uint32_t mExportIndex = 0;

    std::string mScenePathText = "C:/dev/Falcor/media/Bistro/ORCA/Bistro_v5_2/BistroExterior.pyscene";
    std::string mOutputRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets";
    std::string mDatasetName = "falcor_bistro_exterior_builtin_256";

    std::filesystem::path mScenePath = mScenePathText;
    std::filesystem::path mOutputRoot = mOutputRootText;

    uint32_t mSamplesPerAxis = 5;
    float mVolumeExtentScale = 0.65f;
    float mSampleStepScale = 0.05f;
    uint32_t mExportMode = 3; // 0 = Generate GV, 1 = Generate PVV, 2 = Generate GV + PVV, 3 = Render PVV
    bool mWriteDebugProjections = false;

    uint32_t mSamplingMode = 1; // 0 = grid, 1 = path CSV
    std::string mPathCsvText = "C:/dev/Falcor/neuralpvs_paths/arcade_captured_path.csv";
    std::string mCapturedPathCsvText = "C:/dev/Falcor/neuralpvs_paths/arcade_captured_path.csv";
    uint32_t mCapturedPathOutputSamples = 256;
    uint32_t mVisibilityMode = 0; // 0 = view cell rays, 1 = Unity viewcell sample cameras
    float mCameraAspectRatio = 1.777778f;
    uint32_t mVolumeMappingMode = 0; // 0 = world AABB, 1 = Unity view-cell projection
    float mViewCellRadius = 0.3f;
    float mViewCellNearPlane = 0.3f;
    float mViewCellFarPlane = 30.0f;
    uint32_t mPVVSampleSteps = 10;
    bool mLinearZ = true;
    float mLogDepthScale = 0.01f;
    float mUnityFovExpansionDegrees = 30.0f;
    bool mHighDetail = false;
    float mMaxOrthoSize = 80.0f;
    uint32_t mGVDilationRadius = 1;

    bool mRenderScenePreview = true;
    bool mPreviewPlayback = false;
    float mPreviewFps = 12.f;
    uint32_t mPreviewSampleIndex = 0;
    double mPreviewAccumulator = 0.0;
    bool mPreviewWallClockValid = false;
    std::chrono::steady_clock::time_point mPreviewWallClockLast;
    std::vector<ExportSample> mPreviewSamples;
    std::vector<ExportSample> mCapturedPathSamples;

    std::string mRenderDatasetRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets/falcor_bistro_exterior_builtin_256";
    std::string mPredictedPVVRootText = "C:/dev/Falcor/neuralpvs_export_test/datasets/falcor_bistro_exterior_builtin_256/predicted_pvv";
    uint32_t mRenderVolumeSource = 0; // 0 = predicted PVV, 1 = dataset PVV, 2 = dataset GV, 3 = scene only
    uint32_t mRenderVolumeKind = 1; // 0 = GV, 1 = PVV, used by debug overlay coloring
    uint32_t mRenderPVVFilter = 2; // 1 = none/exact, 2 = 3x3x3 box, 3 = trilinear, 4 = radius-2 cross
    uint32_t mRenderSampleIndex = 0;
    uint32_t mRenderVolumeSize = 256;
    uint32_t mRenderVolumeDepth = 256;
    uint32_t mRenderVolumeMappingMode = 0; // Read from metadata. Missing metadata means legacy world AABB.
    float mRenderViewCellRadius = 0.3f;
    float mRenderViewCellNearPlane = 0.3f;
    float mRenderViewCellFarPlane = 30.0f;
    uint32_t mRenderPVVSampleSteps = 10;
    bool mRenderLinearZ = true;
    float mRenderLogDepthScale = 0.01f;
    float mRenderUnityFovExpansionDegrees = 30.0f;
    uint32_t mRenderSamplingFactor = 2;
    bool mRenderHighDetail = false;
    float mRenderMaxOrthoSize = 80.0f;
    bool mRenderExportFrames = false;
    uint32_t mRenderFrameExportMode = 1; // 0 = PNG image sequence, 1 = lossless MKV video.
    bool mRenderPVVActive = false;
    bool mStopRequested = false;
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
    bool mRenderVideoFramesReady = false;
    bool mRenderPVVFinished = false;
    std::filesystem::path mRenderLoadedVolumePath;

    bool mProgressiveExportActive = false;
    uint32_t mProgressiveExportIndex = 0;
    double mProgressiveExportElapsedMs = 0.0;
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
    ref<RasterPass> mpPVVDepthPass;
    ref<ComputePass> mpPVVPass;
    ref<ComputePass> mpPVVRayPass;
    ref<FullScreenPass> mpPVVRenderPass;
    ref<Texture> mpGVVolume;
    ref<Texture> mpPVVVolume;
    ref<Texture> mpRenderVolume;
    ref<Fbo> mpGVFbo;
    ref<Fbo> mpPVVDepthFbo;

    std::string mLastExportStatus = "Not exported yet.";
};
