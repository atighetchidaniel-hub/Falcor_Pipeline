#include "NeuralPVSExporter.h"
#include "Core/Platform/OS.h"
#include "Utils/Math/FalcorMath.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
    uint32_t crc32(const std::vector<uint8_t>& data)
    {
        uint32_t crc = 0xffffffffu;
        for (uint8_t byte : data)
        {
            crc ^= byte;
            for (uint32_t i = 0; i < 8; ++i)
            {
                uint32_t mask = 0u - (crc & 1u);
                crc = (crc >> 1) ^ (0xedb88320u & mask);
            }
        }
        return crc ^ 0xffffffffu;
    }

    void writeLE16(std::ofstream& out, uint16_t value)
    {
        out.put(static_cast<char>(value & 0xffu));
        out.put(static_cast<char>((value >> 8) & 0xffu));
    }

    void writeLE32(std::ofstream& out, uint32_t value)
    {
        out.put(static_cast<char>(value & 0xffu));
        out.put(static_cast<char>((value >> 8) & 0xffu));
        out.put(static_cast<char>((value >> 16) & 0xffu));
        out.put(static_cast<char>((value >> 24) & 0xffu));
    }

    void writeGzipStored(const std::filesystem::path& path, const std::vector<uint8_t>& data)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open output file " + path.string());

        const uint8_t header[10] = {0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
        out.write(reinterpret_cast<const char*>(header), sizeof(header));

        size_t offset = 0;
        while (offset < data.size())
        {
            const size_t blockSize = std::min<size_t>(65535, data.size() - offset);
            const bool finalBlock = (offset + blockSize) == data.size();

            out.put(static_cast<char>(finalBlock ? 0x01 : 0x00));
            writeLE16(out, static_cast<uint16_t>(blockSize));
            writeLE16(out, static_cast<uint16_t>(~static_cast<uint16_t>(blockSize)));
            out.write(reinterpret_cast<const char*>(data.data() + offset), blockSize);

            offset += blockSize;
        }

        writeLE32(out, crc32(data));
        writeLE32(out, static_cast<uint32_t>(data.size()));
    }

    std::string fourDigitName(uint32_t index, const std::string& suffix)
    {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << index << suffix;
        return oss.str();
    }

    std::string trim(std::string value)
    {
        auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char c) { return !isSpace(c); }));
        value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char c) { return !isSpace(c); }).base(), value.end());
        return value;
    }

    std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        return value;
    }

    std::vector<std::string> splitCsvLine(std::string line)
    {
        std::replace(line.begin(), line.end(), ';', ',');
        std::stringstream stream(line);

        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(stream, token, ','))
        {
            tokens.push_back(trim(token));
        }

        return tokens;
    }

    bool tryParseFloat(const std::string& value, float& parsed);

    std::vector<float> parseFloatArray(const std::string& text)
    {
        std::vector<float> values;
        for (const std::string& token : splitCsvLine(text))
        {
            float value = 0.f;
            if (tryParseFloat(token, value))
            {
                values.push_back(value);
            }
        }
        return values;
    }

    std::vector<float> extractFloatArrayFromLine(const std::string& line, const std::string& key)
    {
        const size_t keyPos = line.find(key);
        if (keyPos == std::string::npos)
            return {};

        const size_t begin = line.find('[', keyPos);
        const size_t end = line.find(']', begin);
        if (begin == std::string::npos || end == std::string::npos || end <= begin)
            return {};

        return parseFloatArray(line.substr(begin + 1, end - begin - 1));
    }

    bool extractFloatFromLine(const std::string& line, const std::string& key, float& value)
    {
        const size_t keyPos = line.find(key);
        if (keyPos == std::string::npos)
            return false;

        const size_t colon = line.find(':', keyPos);
        if (colon == std::string::npos)
            return false;

        size_t end = line.find_first_of(",}", colon + 1);
        if (end == std::string::npos)
            end = line.size();

        return tryParseFloat(trim(line.substr(colon + 1, end - colon - 1)), value);
    }

    bool tryParseFloat(const std::string& value, float& parsed)
    {
        try
        {
            size_t parsedChars = 0;
            parsed = std::stof(value, &parsedChars);
            return parsedChars == value.size();
        }
        catch (...)
        {
            return false;
        }
    }

    uint32_t countSetBits(const std::vector<uint8_t>& bytes)
    {
        uint32_t count = 0;
        for (uint8_t byte : bytes)
        {
            uint8_t v = byte;
            while (v != 0)
            {
                count += uint32_t(v & 1u);
                v >>= 1;
            }
        }
        return count;
    }

    float3 normalizedOrDefault(float3 value, float3 fallback)
    {
        const float len = length(value);
        return len > 0.00001f ? value / len : fallback;
    }
}

NeuralPVSExporter::NeuralPVSExporter(const SampleAppConfig& config) : SampleApp(config) {}
NeuralPVSExporter::~NeuralPVSExporter() {}

void NeuralPVSExporter::onLoad(RenderContext* pRenderContext)
{
    loadScene(mScenePath);
    createResources();
    createPreviewPass();
    createGVPass();
    createPVVPass();
    createPVVRenderPass();
}

void NeuralPVSExporter::onShutdown() {}
void NeuralPVSExporter::onResize(uint32_t width, uint32_t height) {}

void NeuralPVSExporter::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    const float4 clearColor(0.08f, 0.10f, 0.12f, 1.0f);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mProgressiveExportActive && !mProgressiveExportSamples.empty())
    {
        mPreviewSampleIndex = std::min(mProgressiveExportIndex, uint32_t(mProgressiveExportSamples.size() - 1u));
        applyPreviewSample();
    }

    if (mRenderScenePreview)
    {
        renderPreview(pRenderContext, pTargetFbo);
    }

    if (mRenderPVVOverlay)
    {
        renderPVVOverlay(pRenderContext, pTargetFbo);
    }

    if (mRenderPVVCullScene && mRenderExportFrames)
    {
        captureRenderFrame(pTargetFbo);
    }

    if (mProgressiveExportActive)
    {
        processProgressiveExportSample(pRenderContext);
    }
}

void NeuralPVSExporter::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "NeuralPVS Exporter", {760, 700}, {20, 40});
    renderGlobalUI(pGui);

    w.text("NeuralPVS Exporter");

    {
        auto group = w.group("Dataset", true);
        if (group)
        {
            w.textbox("Scene path", mScenePathText);
            w.textbox("Dataset name", mDatasetName);
            w.textbox("Output root", mOutputRootText);
            w.textbox("Predicted PVV folder", mPredictedPVVRootText);
        }
    }

    {
        auto group = w.group("Unity Pipeline", true);
        if (group)
        {
            Gui::DropdownList exportModes = {
                {0, "Generate GV"},
                {1, "Generate PVV"},
                {2, "Generate GV + PVV"},
                {3, "Render PVV"},
            };
            w.dropdown("Mode", exportModes, mExportMode);

            w.textbox("Camera path CSV", mPathCsvText);
            w.var("Camera aspect ratio", mCameraAspectRatio, 0.1f, 4.0f, 0.01f);
            w.var("View cell radius", mViewCellRadius, 0.001f, 10.0f, 0.001f);
            w.var("View cell near", mViewCellNearPlane, 0.001f, 10.0f, 0.001f);
            w.var("View cell far", mViewCellFarPlane, 0.1f, 1000.0f, 0.1f);
            if (mExportMode == 3)
            {
                w.checkbox("Export frames", mRenderExportFrames);
                w.text("Frame output: " + getRenderFrameOutputPath().string());
            }

            Gui::DropdownList pvvFilters = {
                {1, "Exact"},
                {2, "Box"},
                {3, "Trilinear"},
            };
            w.dropdown("RenderPVV filter", pvvFilters, mRenderPVVFilter);
            w.checkbox("Write debug projections", mWriteDebugProjections);
        }
    }

    {
        auto group = w.group("Path Preview", true);
        if (group)
        {
            w.checkbox("Render scene preview", mRenderScenePreview);
            w.checkbox("Play path", mPreviewPlayback);
            w.var("Playback FPS", mPreviewFps, 0.5f, 60.0f, 0.5f);

            const uint32_t maxPreviewIndex = mPreviewSamples.empty() ? 0u : uint32_t(mPreviewSamples.size() - 1u);
            w.var("Preview sample", mPreviewSampleIndex, 0u, maxPreviewIndex);

            if (w.button("Load Path Preview"))
            {
                loadPreviewPath();
            }
            if (w.button("Previous sample") && !mPreviewSamples.empty())
            {
                mPreviewPlayback = false;
                mPreviewSampleIndex = mPreviewSampleIndex == 0u ? maxPreviewIndex : mPreviewSampleIndex - 1u;
                applyPreviewSample();
            }
            if (w.button("Next sample") && !mPreviewSamples.empty())
            {
                mPreviewPlayback = false;
                mPreviewSampleIndex = (mPreviewSampleIndex + 1u) % (maxPreviewIndex + 1u);
                applyPreviewSample();
            }

            w.text("Preview samples: " + std::to_string(mPreviewSamples.size()));
            if (mProgressiveExportActive)
            {
                w.text(
                    "Exporting sample " + std::to_string(mProgressiveExportIndex) + " / " +
                    std::to_string(mProgressiveExportSamples.empty() ? 0u : uint32_t(mProgressiveExportSamples.size() - 1u))
                );
                if (w.button("Cancel export"))
                {
                    mProgressiveExportActive = false;
                    mLastExportStatus = "Canceled export at sample " + std::to_string(mProgressiveExportIndex) + ".";
                }
            }
        }
    }

    w.separator();

    if (w.button("Load Scene"))
    {
        mScenePath = std::filesystem::path(mScenePathText);
        mOutputRoot = std::filesystem::path(mOutputRootText);

        loadScene(mScenePath);
        createResources();
        createPreviewPass();
        createGVPass();
        createPVVPass();
        createPVVRenderPass();

        mLastExportStatus = "Loaded scene: " + mScenePath.string();
    }

    if (w.button("Run Mode") && !mProgressiveExportActive)
    {
        mSamplingMode = 1;
        mVisibilityMode = 1;
        mVolumeMappingMode = 1;
        mScenePath = std::filesystem::path(mScenePathText);
        mOutputRoot = std::filesystem::path(mOutputRootText);
        mRenderDatasetRootText = (mOutputRoot / mDatasetName).string();
        mRenderVolumeSource = 0;
        mRenderUseSampleCamera = true;
        mRenderKeepOutsidePVVInView = true;
        mRenderPVVOverlay = false;
        mRenderLastCapturedSampleIndex = 0xffffffffu;
        mRenderCapturedFrameCount = 0;

        loadScene(mScenePath);
        createResources();
        createPreviewPass();
        createGVPass();
        createPVVPass();
        createPVVRenderPass();

        if (mExportMode == 3)
        {
            startRenderPVVMode();
        }
        else
        {
            startProgressiveExport();
        }
    }

    w.separator();

    w.text("Scene: " + mScenePath.string());
    w.text("Output: " + (mOutputRoot / mDatasetName).string());
    if (mExportMode == 3)
    {
        w.text("Predicted PVV: " + std::filesystem::path(mPredictedPVVRootText).string());
        w.text(mRenderStatus);
    }
    w.text(mLastExportStatus);
}
bool NeuralPVSExporter::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mpScene && !mPreviewPlayback && mpScene->onKeyEvent(keyEvent);
}

bool NeuralPVSExporter::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && !mPreviewPlayback && mpScene->onMouseEvent(mouseEvent);
}

void NeuralPVSExporter::onHotReload(HotReloadFlags reloaded) {}

void NeuralPVSExporter::loadScene(const std::filesystem::path& path)
{
    mpScene = Scene::create(getDevice(), path);
    mpCamera = mpScene->getCamera();

    if (mpCamera)
    {
        const float radius = mpScene->getSceneBounds().radius();
        mpScene->setCameraSpeed(radius * 0.25f);
        mpCamera->setDepthRange(std::max(0.1f, radius / 750.0f), radius * 10.0f);
        mpCamera->setFocalLength(18.0f);
        mpCamera->setAspectRatio(float(mRasterWidth) / float(mRasterHeight));
    }
}

void NeuralPVSExporter::createResources()
{
    mpGVVolume = getDevice()->createTexture3D(
        mVolumeSize / 32, mVolumeSize, mVolumeDepth, ResourceFormat::R32Uint, 1, nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mpPVVVolume = getDevice()->createTexture3D(
        mVolumeSize / 32, mVolumeSize, mVolumeDepth, ResourceFormat::R32Uint, 1, nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    std::vector<uint8_t> emptyVolume(size_t(mVolumeSize) * size_t(mVolumeSize) * size_t(mVolumeDepth) / 8, 0);
    mpRenderVolume = getDevice()->createTexture3D(
        mVolumeSize / 32,
        mVolumeSize,
        mVolumeDepth,
        ResourceFormat::R32Uint,
        1,
        emptyVolume.data(),
        ResourceBindFlags::ShaderResource
    );
    mRenderLoadedSampleIndex = 0xffffffffu;
    mRenderLoadedVolumeSource = 0xffffffffu;
    mRenderLoadedVolumePath.clear();

    Fbo::Desc desc;
    desc.setColorTarget(0, ResourceFormat::RGBA8Unorm);
    desc.setDepthStencilTarget(ResourceFormat::D32Float);
    mpGVFbo = Fbo::create2D(getDevice(), mRasterWidth, mRasterHeight, desc);
}

void NeuralPVSExporter::createPreviewPass()
{
    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/NeuralPVSExporter/NeuralPVSPreview.3d.slang").vsEntry("vsMain").psEntry("psMain");
    desc.addTypeConformances(mpScene->getTypeConformances());

    mpPreviewPass = RasterPass::create(getDevice(), desc, mpScene->getSceneDefines());
}

void NeuralPVSExporter::createGVPass()
{
    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/NeuralPVSExporter/NeuralPVSGV.3d.slang").vsEntry("vsMain").psEntry("psMain");
    desc.addTypeConformances(mpScene->getTypeConformances());

    mpGVPass = RasterPass::create(getDevice(), desc, mpScene->getSceneDefines());
}

void NeuralPVSExporter::createPVVPass()
{
    if (!getDevice()->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        FALCOR_THROW("NeuralPVS PVV export requires DXR 1.1 / inline ray tracing support.");
    }

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/NeuralPVSExporter/NeuralPVSPVV.cs.slang").csEntry("main");
    desc.addTypeConformances(mpScene->getTypeConformances());

    mpPVVPass = ComputePass::create(getDevice(), desc, mpScene->getSceneDefines());
}

void NeuralPVSExporter::createPVVRenderPass()
{
    mpPVVRenderPass = FullScreenPass::create(getDevice(), "Samples/NeuralPVSExporter/NeuralPVSRenderPVV.ps.slang");

    BlendState::Desc blendDesc;
    blendDesc.setRtBlend(0, true).setRtParams(
        0,
        BlendState::BlendOp::Add,
        BlendState::BlendOp::Add,
        BlendState::BlendFunc::SrcAlpha,
        BlendState::BlendFunc::OneMinusSrcAlpha,
        BlendState::BlendFunc::One,
        BlendState::BlendFunc::One
    );
    mpPVVRenderPass->getState()->setBlendState(BlendState::create(blendDesc));
}

void NeuralPVSExporter::renderPreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (!mpScene || !mpPreviewPass || !mpCamera)
        return;

    if (pTargetFbo->getHeight() > 0)
    {
        mpCamera->setAspectRatio(float(pTargetFbo->getWidth()) / float(pTargetFbo->getHeight()));
    }

    if (!mPreviewSamples.empty())
    {
        if (mPreviewPlayback)
        {
            mPreviewAccumulator += getGlobalClock().getDelta();
            const double frameSeconds = 1.0 / std::max(0.5f, mPreviewFps);
            while (mPreviewAccumulator >= frameSeconds)
            {
                if (mRenderPVVCullScene && !mRenderSamples.empty())
                {
                    if (mPreviewSampleIndex + 1u >= uint32_t(mPreviewSamples.size()))
                    {
                        mPreviewPlayback = false;
                        mPreviewAccumulator = 0.0;
                        mRenderStatus =
                            "RenderPVV finished at sample " + std::to_string(mPreviewSampleIndex) +
                            (mRenderExportFrames ? ". Frames saved to " + getRenderFrameOutputPath().string() : ".");
                        mLastExportStatus = mRenderStatus;
                        break;
                    }

                    ++mPreviewSampleIndex;
                }
                else
                {
                    mPreviewSampleIndex = (mPreviewSampleIndex + 1u) % uint32_t(mPreviewSamples.size());
                }
                mPreviewAccumulator -= frameSeconds;
            }
        }

        applyPreviewSample();
    }

    uint32_t enableRenderPVV = 0u;
    float3 renderVolumeMin = float3(0.f);
    VolumeProjectionParams renderProjection;
    uint32_t renderVolumeMappingMode = 0u;
    if (mRenderPVVCullScene && !mRenderSamples.empty())
    {
        if (mRenderUseSampleCamera && !mPreviewSamples.empty())
        {
            mRenderSampleIndex = std::min(mPreviewSampleIndex, uint32_t(mRenderSamples.size() - 1u));
        }

        ensureRenderVolumeLoaded(false);
        mRenderSampleIndex = std::min(mRenderSampleIndex, uint32_t(mRenderSamples.size() - 1u));
        const ExportSample& renderSample = mRenderSamples[mRenderSampleIndex];
        renderVolumeMin = renderSample.center - mRenderVolumeExtent * 0.5f;
        renderProjection = makeVolumeProjection(
            renderSample,
            mRenderViewCellRadius,
            mRenderViewCellNearPlane,
            mRenderViewCellFarPlane
        );
        renderVolumeMappingMode = mRenderVolumeMappingMode == 1u && renderSample.hasCamera ? 1u : 0u;
        enableRenderPVV = mpRenderVolume ? 1u : 0u;
    }

    IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
    {
        FALCOR_THROW("Scene update requires shader recompilation. Reload the scene.");
    }

    auto previewRoot = mpPreviewPass->getRootVar();
    previewRoot["gRenderPVVVolume"] = mpRenderVolume;
    previewRoot["PreviewCB"]["gRenderPVVVolumeMin"] = renderVolumeMin;
    previewRoot["PreviewCB"]["gRenderPVVVolumeSize"] = mRenderVolumeSize;
    previewRoot["PreviewCB"]["gRenderPVVVolumeExtent"] = mRenderVolumeExtent;
    previewRoot["PreviewCB"]["gRenderPVVVolumeDepth"] = mRenderVolumeDepth;
    previewRoot["PreviewCB"]["gEnableRenderPVV"] = enableRenderPVV;
    previewRoot["PreviewCB"]["gPVVFilter"] = mRenderPVVFilter;
    previewRoot["PreviewCB"]["gRenderPVVSource"] = mRenderVolumeSource;
    previewRoot["PreviewCB"]["gKeepOutsidePVVInView"] = mRenderKeepOutsidePVVInView ? 1u : 0u;
    previewRoot["PreviewCB"]["gMainCamViewProj"] = mpCamera->getViewProjMatrix();
    previewRoot["PreviewCB"]["gRenderVolumeMappingMode"] = renderVolumeMappingMode;
    previewRoot["PreviewCB"]["gViewCellPosition"] = renderProjection.viewCellPosition;
    previewRoot["PreviewCB"]["gViewCellForward"] = renderProjection.forward;
    previewRoot["PreviewCB"]["gViewCellRight"] = renderProjection.right;
    previewRoot["PreviewCB"]["gViewCellUp"] = renderProjection.up;
    previewRoot["PreviewCB"]["gViewCellNearPlane"] = renderProjection.nearPlane;
    previewRoot["PreviewCB"]["gViewCellFarPlane"] = renderProjection.farPlane;
    previewRoot["PreviewCB"]["gTanHalfFovX"] = renderProjection.tanHalfFovX;
    previewRoot["PreviewCB"]["gTanHalfFovY"] = renderProjection.tanHalfFovY;

    mpPreviewPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpPreviewPass->getState().get(), mpPreviewPass->getVars().get());
}

void NeuralPVSExporter::renderPVVOverlay(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (!mpPVVRenderPass || !mpRenderVolume || !mpCamera || mRenderSamples.empty())
        return;

    if (pTargetFbo->getHeight() > 0)
    {
        mpCamera->setAspectRatio(float(pTargetFbo->getWidth()) / float(pTargetFbo->getHeight()));
    }

    mRenderSampleIndex = std::min(mRenderSampleIndex, uint32_t(mRenderSamples.size() - 1u));
    const ExportSample& sample = mRenderSamples[mRenderSampleIndex];
    const float3 volumeMin = sample.center - mRenderVolumeExtent * 0.5f;

    auto root = mpPVVRenderPass->getRootVar();
    root["gVolume"] = mpRenderVolume;
    root["RenderCB"]["gInvViewProj"] = mpCamera->getInvViewProjMatrix();
    root["RenderCB"]["gCameraPos"] = mpCamera->getPosition();
    root["RenderCB"]["gVolumeMin"] = volumeMin;
    root["RenderCB"]["gVolumeSize"] = mRenderVolumeSize;
    root["RenderCB"]["gVolumeExtent"] = mRenderVolumeExtent;
    root["RenderCB"]["gVolumeDepth"] = mRenderVolumeDepth;
    root["RenderCB"]["gSampleCenter"] = sample.center;
    root["RenderCB"]["gOpacity"] = std::clamp(mRenderOpacity, 0.01f, 1.0f);
    root["RenderCB"]["gStepScale"] = std::clamp(mRenderStepScale, 0.25f, 4.0f);
    root["RenderCB"]["gVolumeKind"] = mRenderVolumeKind;

    mpPVVRenderPass->execute(pRenderContext, pTargetFbo);
}

void NeuralPVSExporter::loadPreviewPath()
{
    mPreviewSamples = loadPathSamples(std::filesystem::path(mPathCsvText));
    mPreviewSampleIndex = 0;
    mPreviewAccumulator = 0.0;
    applyPreviewSample();
    mLastExportStatus = "Loaded " + std::to_string(mPreviewSamples.size()) + " preview path samples.";
}

void NeuralPVSExporter::applyPreviewSample()
{
    if (!mpCamera || mPreviewSamples.empty())
        return;

    mPreviewSampleIndex = std::min(mPreviewSampleIndex, uint32_t(mPreviewSamples.size() - 1u));
    const ExportSample& sample = mPreviewSamples[mPreviewSampleIndex];

    const float3 forward = normalizedOrDefault(sample.forward, float3(0.f, 0.f, -1.f));
    const float3 upHint = std::abs(forward.y) > 0.98f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
    const float3 right = normalizedOrDefault(cross(upHint, forward), float3(1.f, 0.f, 0.f));
    const float3 up = normalizedOrDefault(cross(forward, right), upHint);

    mpCamera->setPosition(sample.center);
    mpCamera->setTarget(sample.center + forward);
    mpCamera->setUpVector(up);

    const float fovYRadians = std::clamp(sample.fovYDegrees, 1.0f, 179.0f) * 3.1415926535f / 180.0f;
    mpCamera->setFocalLength(fovYToFocalLength(fovYRadians, Camera::kDefaultFrameHeight));
}

void NeuralPVSExporter::loadRenderMetadata()
{
    mRenderDatasetRoot = std::filesystem::path(mRenderDatasetRootText);
    const std::filesystem::path metadataPath = mRenderDatasetRoot / "metadata.json";

    std::ifstream metadata(metadataPath);
    if (!metadata)
    {
        FALCOR_THROW("Failed to open render metadata '{}'.", metadataPath.string());
    }

    std::vector<ExportSample> samples;
    float3 volumeExtent = float3(1.f);
    uint32_t volumeSize = mVolumeSize;
    uint32_t volumeDepth = mVolumeDepth;
    uint32_t volumeMappingMode = 0u;
    float cameraAspectRatio = mCameraAspectRatio;
    float viewCellRadius = mViewCellRadius;
    float viewCellNearPlane = mViewCellNearPlane;
    float viewCellFarPlane = mViewCellFarPlane;

    std::string line;
    while (std::getline(metadata, line))
    {
        if (line.find("\"volume_mapping\"") != std::string::npos && line.find("unity_projection") != std::string::npos)
        {
            volumeMappingMode = 1u;
        }

        extractFloatFromLine(line, "\"camera_aspect_ratio\"", cameraAspectRatio);
        extractFloatFromLine(line, "\"view_cell_radius\"", viewCellRadius);
        extractFloatFromLine(line, "\"view_cell_near\"", viewCellNearPlane);
        extractFloatFromLine(line, "\"view_cell_far\"", viewCellFarPlane);

        std::vector<float> volumeSizeValues = extractFloatArrayFromLine(line, "\"volume_size\"");
        if (volumeSizeValues.size() >= 3)
        {
            volumeSize = std::max(1u, uint32_t(std::round(volumeSizeValues[0])));
            volumeDepth = std::max(1u, uint32_t(std::round(volumeSizeValues[2])));
        }

        std::vector<float> volumeExtentValues = extractFloatArrayFromLine(line, "\"volume_extent\"");
        if (volumeExtentValues.size() >= 3)
        {
            volumeExtent = float3(volumeExtentValues[0], volumeExtentValues[1], volumeExtentValues[2]);
        }

        std::vector<float> centerValues = extractFloatArrayFromLine(line, "\"center\"");
        if (centerValues.size() >= 3)
        {
            ExportSample sample;
            sample.center = float3(centerValues[0], centerValues[1], centerValues[2]);
            sample.hasCamera = line.find("\"has_camera\": true") != std::string::npos;

            std::vector<float> forwardValues = extractFloatArrayFromLine(line, "\"forward\"");
            if (forwardValues.size() >= 3)
            {
                sample.forward = normalizedOrDefault(float3(forwardValues[0], forwardValues[1], forwardValues[2]), float3(0.f, 0.f, -1.f));
                sample.hasCamera = true;
            }

            float fovYDegrees = sample.fovYDegrees;
            if (extractFloatFromLine(line, "\"fov_y_degrees\"", fovYDegrees))
            {
                sample.fovYDegrees = fovYDegrees;
            }

            samples.push_back(sample);
        }
    }

    if (samples.empty())
    {
        FALCOR_THROW("Render metadata '{}' did not contain any samples.", metadataPath.string());
    }

    if (volumeSize % 32u != 0u)
    {
        FALCOR_THROW("Render volume size {} must be divisible by 32.", volumeSize);
    }

    mRenderSamples = std::move(samples);
    mRenderVolumeExtent = volumeExtent;
    mRenderVolumeSize = volumeSize;
    mRenderVolumeDepth = volumeDepth;
    mRenderVolumeMappingMode = volumeMappingMode;
    mCameraAspectRatio = std::max(0.1f, cameraAspectRatio);
    mRenderViewCellRadius = std::max(0.001f, viewCellRadius);
    mRenderViewCellNearPlane = std::max(0.001f, viewCellNearPlane);
    mRenderViewCellFarPlane = std::max(mRenderViewCellNearPlane + 0.001f, viewCellFarPlane);
    mRenderSampleIndex = std::min(mRenderSampleIndex, uint32_t(mRenderSamples.size() - 1u));
    mRenderLoadedSampleIndex = 0xffffffffu;
    mRenderLoadedVolumeSource = 0xffffffffu;
    mRenderLoadedVolumePath.clear();
    mRenderStatus = "Loaded metadata with " + std::to_string(mRenderSamples.size()) + " samples.";
    mLastExportStatus = mRenderStatus;
}

void NeuralPVSExporter::loadRenderVolume()
{
    ensureRenderVolumeLoaded(true);
}

void NeuralPVSExporter::ensureRenderVolumeLoaded(bool forceReload)
{
    const std::filesystem::path requestedRoot = std::filesystem::path(mRenderDatasetRootText);
    if (mRenderSamples.empty() || requestedRoot != mRenderDatasetRoot)
    {
        loadRenderMetadata();
    }

    mRenderSampleIndex = std::min(mRenderSampleIndex, uint32_t(mRenderSamples.size() - 1u));
    const std::filesystem::path volumePath = resolveRenderVolumePath();

    if (!forceReload && mpRenderVolume && mRenderLoadedSampleIndex == mRenderSampleIndex &&
        mRenderLoadedVolumeSource == mRenderVolumeSource && mRenderLoadedVolumePath == volumePath)
    {
        return;
    }

    std::vector<uint8_t> bytes = readGzipStoredFile(volumePath);
    const size_t expectedBytes = size_t(mRenderVolumeSize) * size_t(mRenderVolumeSize) * size_t(mRenderVolumeDepth) / 8;
    if (bytes.size() != expectedBytes)
    {
        FALCOR_THROW("Render volume '{}' has {} bytes, expected {}.", volumePath.string(), bytes.size(), expectedBytes);
    }

    mpRenderVolume = getDevice()->createTexture3D(
        mRenderVolumeSize / 32,
        mRenderVolumeSize,
        mRenderVolumeDepth,
        ResourceFormat::R32Uint,
        1,
        bytes.data(),
        ResourceBindFlags::ShaderResource
    );

    mRenderLoadedSampleIndex = mRenderSampleIndex;
    mRenderLoadedVolumeSource = mRenderVolumeSource;
    mRenderLoadedVolumePath = volumePath;
    mRenderVolumeKind = mRenderVolumeSource == 2u ? 0u : 1u;

    if (mRenderUseSampleCamera)
    {
        mPreviewSamples = mRenderSamples;
        mPreviewSampleIndex = mRenderSampleIndex;
        applyPreviewSample();
    }

    mRenderScenePreview = true;
    mRenderStatus =
        "Loaded " + std::string(mRenderVolumeSource == 2u ? "GV" : (mRenderVolumeSource == 1u ? "PVV" : "predicted PVV")) +
        " sample " + std::to_string(mRenderSampleIndex) + " from " + volumePath.string();
    mLastExportStatus = mRenderStatus;
}

void NeuralPVSExporter::startRenderPVVMode()
{
    mRenderDatasetRootText = (std::filesystem::path(mOutputRootText) / mDatasetName).string();
    mRenderVolumeSource = 0u;
    loadRenderMetadata();

    mRenderPVVCullScene = true;
    mRenderKeepOutsidePVVInView = true;
    mRenderPVVOverlay = false;
    mRenderUseSampleCamera = true;
    mRenderScenePreview = true;
    mRenderSampleIndex = 0;
    mRenderLastCapturedSampleIndex = 0xffffffffu;
    mRenderCapturedFrameCount = 0;
    mPreviewSamples = mRenderSamples;
    mPreviewSampleIndex = 0;
    mPreviewAccumulator = 0.0;
    mPreviewPlayback = true;

    if (mRenderExportFrames)
    {
        std::filesystem::create_directories(getRenderFrameOutputPath());
    }

    applyPreviewSample();
    ensureRenderVolumeLoaded(true);

    mRenderStatus =
        "RenderPVV mode running through predicted PVV samples" +
        std::string(mRenderExportFrames ? ". Saving frames to " + getRenderFrameOutputPath().string() : ".");
    mLastExportStatus = mRenderStatus;
}

std::filesystem::path NeuralPVSExporter::resolveRenderVolumePath() const
{
    if (mRenderSamples.empty())
    {
        FALCOR_THROW("Render metadata must be loaded before resolving a volume file.");
    }

    const uint32_t sampleIndex = std::min(mRenderSampleIndex, uint32_t(mRenderSamples.size() - 1u));

    if (mRenderVolumeSource == 0u)
    {
        const std::filesystem::path predictedRoot = std::filesystem::path(mPredictedPVVRootText);
        const std::filesystem::path unpaddedPath = predictedRoot / (std::to_string(sampleIndex) + "_predicted_pvv.bin.gz");
        if (std::filesystem::exists(unpaddedPath))
            return unpaddedPath;

        const std::filesystem::path paddedPath = predictedRoot / fourDigitName(sampleIndex, "_predicted_pvv.bin.gz");
        if (std::filesystem::exists(paddedPath))
            return paddedPath;

        FALCOR_THROW(
            "Could not find predicted PVV sample {}. Tried '{}' and '{}'.",
            sampleIndex,
            unpaddedPath.string(),
            paddedPath.string()
        );
    }

    if (mRenderVolumeSource == 1u)
    {
        return mRenderDatasetRoot / "pvv" / fourDigitName(sampleIndex, "_pvv.bin.gz");
    }

    return mRenderDatasetRoot / "gv" / fourDigitName(sampleIndex, "_gv.bin.gz");
}

std::filesystem::path NeuralPVSExporter::getRenderFrameOutputPath() const
{
    return std::filesystem::path(mPredictedPVVRootText) / "00_color";
}

void NeuralPVSExporter::captureRenderFrame(const ref<Fbo>& pTargetFbo)
{
    if (!pTargetFbo || mRenderSamples.empty() || mRenderSampleIndex >= mRenderSamples.size())
        return;

    if (mRenderLastCapturedSampleIndex == mRenderSampleIndex)
        return;

    const std::filesystem::path outputPath = getRenderFrameOutputPath();
    std::filesystem::create_directories(outputPath);

    std::ostringstream filename;
    filename << std::setw(4) << std::setfill('0') << mRenderSampleIndex << ".png";
    const std::filesystem::path framePath = outputPath / filename.str();

    pTargetFbo->getColorTexture(0)->captureToFile(0, 0, framePath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);

    mRenderLastCapturedSampleIndex = mRenderSampleIndex;
    ++mRenderCapturedFrameCount;
    mRenderStatus =
        "Saved RenderPVV frame " + std::to_string(mRenderSampleIndex) + " / " +
        std::to_string(mRenderSamples.size() - 1u) + ": " + framePath.string();
    mLastExportStatus = mRenderStatus;
}

std::vector<NeuralPVSExporter::ExportSample> NeuralPVSExporter::buildExportSamples(const float3& sceneCenter, const float3& sceneExtent) const
{
    const float3 sampleStep = sceneExtent * mSampleStepScale;
    std::vector<ExportSample> samples;

    if (mSamplingMode == 1)
    {
        return loadPathSamples(std::filesystem::path(mPathCsvText));
    }

    const uint32_t samplesPerAxis = mSamplesPerAxis < 1u ? 1u : mSamplesPerAxis;
    samples.reserve(size_t(samplesPerAxis) * size_t(samplesPerAxis) * size_t(samplesPerAxis));

    const float centerOffset = 0.5f * float(samplesPerAxis - 1u);
    for (uint32_t z = 0; z < samplesPerAxis; ++z)
    {
        for (uint32_t y = 0; y < samplesPerAxis; ++y)
        {
            for (uint32_t x = 0; x < samplesPerAxis; ++x)
            {
                const float3 offset = float3(float(x) - centerOffset, float(y) - centerOffset, float(z) - centerOffset);
                ExportSample sample;
                sample.center = sceneCenter + offset * sampleStep;
                samples.push_back(sample);
            }
        }
    }

    return samples;
}

void NeuralPVSExporter::validateExportSamples(const std::vector<ExportSample>& samples, bool useCameraFrustum) const
{
    if (samples.empty())
    {
        FALCOR_THROW("Export did not produce any sample centers.");
    }

    if (!useCameraFrustum)
        return;

    if (mSamplingMode != 1)
    {
        FALCOR_THROW("Camera frustum visibility requires Path CSV sampling.");
    }

    const bool allSamplesHaveCamera = std::all_of(samples.begin(), samples.end(), [](const ExportSample& sample) { return sample.hasCamera; });
    if (!allSamplesHaveCamera)
    {
        FALCOR_THROW("Camera frustum visibility requires CSV columns forward_x, forward_y, forward_z, and fov.");
    }
}

NeuralPVSExporter::VolumeProjectionParams NeuralPVSExporter::makeVolumeProjection(
    const ExportSample& sample,
    float viewCellRadius,
    float nearPlane,
    float farPlane
) const
{
    VolumeProjectionParams params;

    const float fovYRadians = std::clamp(sample.fovYDegrees, 1.0f, 179.0f) * 3.1415926535f / 180.0f;
    const float aspectRatio = std::max(0.1f, mCameraAspectRatio);
    const float tanHalfOffsetFov = std::max(0.0001f, std::tan(0.5f * fovYRadians / aspectRatio));
    const float viewCellOffset = std::max(0.0f, viewCellRadius) / tanHalfOffsetFov;

    params.forward = normalizedOrDefault(sample.forward, float3(0.f, 0.f, -1.f));
    const float3 upHint = std::abs(params.forward.y) > 0.98f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
    params.right = normalizedOrDefault(cross(upHint, params.forward), float3(1.f, 0.f, 0.f));
    params.up = normalizedOrDefault(cross(params.forward, params.right), upHint);

    params.viewCellPosition = sample.center - params.forward * viewCellOffset;
    params.nearPlane = std::max(0.001f, nearPlane);
    params.farPlane = std::max(params.nearPlane + 0.001f, farPlane + 2.0f * viewCellOffset);
    params.tanHalfFovY = std::max(0.0001f, std::tan(0.5f * fovYRadians));
    params.tanHalfFovX = params.tanHalfFovY * aspectRatio;

    return params;
}

void NeuralPVSExporter::startProgressiveExport()
{
    mProgressiveSceneBounds = mpScene->getSceneBounds();
    mProgressiveSceneCenter = mProgressiveSceneBounds.center();
    mProgressiveSceneExtent = max(mProgressiveSceneBounds.extent(), float3(0.0001f));
    mProgressiveVolumeExtent = mProgressiveSceneExtent * mVolumeExtentScale;
    mProgressiveSampleStep = mProgressiveSceneExtent * mSampleStepScale;
    mProgressiveViewCellRadius = std::max(0.001f, mViewCellRadius);
    mProgressiveUseCameraFrustum = mVisibilityMode == 1;

    mProgressiveExportSamples = buildExportSamples(mProgressiveSceneCenter, mProgressiveSceneExtent);
    validateExportSamples(mProgressiveExportSamples, mProgressiveUseCameraFrustum);

    mProgressiveGVBitCounts.clear();
    mProgressivePVVBitCounts.clear();
    mProgressiveGVBitCounts.reserve(mProgressiveExportSamples.size());
    mProgressivePVVBitCounts.reserve(mProgressiveExportSamples.size());
    mProgressiveExportIndex = 0;
    mProgressiveExportActive = true;

    mPreviewSamples = mProgressiveExportSamples;
    mPreviewSampleIndex = 0;
    mPreviewPlayback = false;
    mPreviewAccumulator = 0.0;
    applyPreviewSample();

    mLastExportStatus = "Started preview export of " + std::to_string(mProgressiveExportSamples.size()) + " samples.";
}

void NeuralPVSExporter::processProgressiveExportSample(RenderContext* pRenderContext)
{
    if (!mProgressiveExportActive)
        return;

    if (mProgressiveExportIndex >= mProgressiveExportSamples.size())
    {
        finishProgressiveExport();
        return;
    }

    mPreviewSampleIndex = mProgressiveExportIndex;
    applyPreviewSample();

    uint64_t gvBitCount = 0;
    uint64_t pvvBitCount = 0;
    exportOneSample(
        pRenderContext,
        mProgressiveExportSamples[mProgressiveExportIndex],
        mProgressiveExportIndex,
        mProgressiveVolumeExtent,
        mProgressiveViewCellRadius,
        gvBitCount,
        pvvBitCount
    );

    mProgressiveGVBitCounts.push_back(gvBitCount);
    mProgressivePVVBitCounts.push_back(pvvBitCount);

    mLastExportStatus =
        "Exported sample " + std::to_string(mProgressiveExportIndex) + " / " +
        std::to_string(mProgressiveExportSamples.size() - 1) + " with live preview.";

    ++mProgressiveExportIndex;
    if (mProgressiveExportIndex >= mProgressiveExportSamples.size())
    {
        finishProgressiveExport();
    }
}

void NeuralPVSExporter::finishProgressiveExport()
{
    writeExportMetadata(
        mProgressiveExportSamples,
        mProgressiveGVBitCounts,
        mProgressivePVVBitCounts,
        mProgressiveSceneBounds,
        mProgressiveSceneCenter,
        mProgressiveSceneExtent,
        mProgressiveVolumeExtent,
        mProgressiveSampleStep,
        mProgressiveUseCameraFrustum
    );

    mProgressiveExportActive = false;
    mLastExportStatus = "Exported " + std::to_string(mProgressiveExportSamples.size()) + " " + mDatasetName + " samples with live preview.";
}

void NeuralPVSExporter::exportOneSample(
    RenderContext* pRenderContext,
    const ExportSample& sample,
    uint32_t index,
    const float3& volumeExtent,
    float viewCellRadius,
    uint64_t& gvBitCount,
    uint64_t& pvvBitCount
)
{
    const float3 viewCellCenter = sample.center;
    const float3 volumeMin = viewCellCenter - volumeExtent * 0.5f;
    const bool useCameraFrustum = mVisibilityMode == 1;
    const bool useProjectionVolume = mVolumeMappingMode == 1 && sample.hasCamera;
    const VolumeProjectionParams volumeProjection =
        makeVolumeProjection(sample, viewCellRadius, mViewCellNearPlane, mViewCellFarPlane);

    pRenderContext->clearUAV(mpGVVolume->getUAV().get(), uint4(0, 0, 0, 0));
    pRenderContext->clearUAV(mpPVVVolume->getUAV().get(), uint4(0, 0, 0, 0));
    pRenderContext->clearFbo(mpGVFbo.get(), float4(0, 0, 0, 0), 1.0f, 0, FboAttachmentType::All);

    auto gvRoot = mpGVPass->getRootVar();
    gvRoot["gGeometryVolume"] = mpGVVolume;
    gvRoot["ExporterCB"]["gSceneMin"] = volumeMin;
    gvRoot["ExporterCB"]["gSceneExtent"] = volumeExtent;
    gvRoot["ExporterCB"]["gVolumeSize"] = mVolumeSize;
    gvRoot["ExporterCB"]["gVolumeDepth"] = mVolumeDepth;
    gvRoot["ExporterCB"]["gUseProjectionVolume"] = useProjectionVolume ? 1u : 0u;
    gvRoot["ExporterCB"]["gViewCellPosition"] = volumeProjection.viewCellPosition;
    gvRoot["ExporterCB"]["gViewCellForward"] = volumeProjection.forward;
    gvRoot["ExporterCB"]["gViewCellRight"] = volumeProjection.right;
    gvRoot["ExporterCB"]["gViewCellUp"] = volumeProjection.up;
    gvRoot["ExporterCB"]["gViewCellNearPlane"] = volumeProjection.nearPlane;
    gvRoot["ExporterCB"]["gViewCellFarPlane"] = volumeProjection.farPlane;
    gvRoot["ExporterCB"]["gTanHalfFovX"] = volumeProjection.tanHalfFovX;
    gvRoot["ExporterCB"]["gTanHalfFovY"] = volumeProjection.tanHalfFovY;

    mpGVPass->getState()->setFbo(mpGVFbo);
    mpScene->rasterize(pRenderContext, mpGVPass->getState().get(), mpGVPass->getVars().get());

    auto pvvRoot = mpPVVPass->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, pvvRoot["gScene"]);

    pvvRoot["gGeometryVolume"] = mpGVVolume;
    pvvRoot["gPVVVolume"] = mpPVVVolume;
    pvvRoot["PVVCB"]["gSceneMin"] = volumeMin;
    pvvRoot["PVVCB"]["gSceneExtent"] = volumeExtent;
    pvvRoot["PVVCB"]["gVolumeSize"] = mVolumeSize;
    pvvRoot["PVVCB"]["gVolumeDepth"] = mVolumeDepth;
    pvvRoot["PVVCB"]["gViewCellCenter"] = viewCellCenter;
    pvvRoot["PVVCB"]["gViewCellRadius"] = viewCellRadius;
    pvvRoot["PVVCB"]["gSampleCount"] = useCameraFrustum ? 1u : 9u;
    pvvRoot["PVVCB"]["gUseCameraFrustum"] = useCameraFrustum ? 1u : 0u;
    pvvRoot["PVVCB"]["gCameraFovYRadians"] = std::clamp(sample.fovYDegrees, 1.0f, 179.0f) * 3.1415926535f / 180.0f;
    pvvRoot["PVVCB"]["gCameraAspectRatio"] = std::max(0.1f, mCameraAspectRatio);
    pvvRoot["PVVCB"]["gCameraForward"] = normalizedOrDefault(sample.forward, float3(0.f, 0.f, -1.f));
    pvvRoot["PVVCB"]["gUseProjectionVolume"] = useProjectionVolume ? 1u : 0u;
    pvvRoot["PVVCB"]["gViewCellPosition"] = volumeProjection.viewCellPosition;
    pvvRoot["PVVCB"]["gViewCellForward"] = volumeProjection.forward;
    pvvRoot["PVVCB"]["gViewCellRight"] = volumeProjection.right;
    pvvRoot["PVVCB"]["gViewCellUp"] = volumeProjection.up;
    pvvRoot["PVVCB"]["gViewCellNearPlane"] = volumeProjection.nearPlane;
    pvvRoot["PVVCB"]["gViewCellFarPlane"] = volumeProjection.farPlane;
    pvvRoot["PVVCB"]["gTanHalfFovX"] = volumeProjection.tanHalfFovX;
    pvvRoot["PVVCB"]["gTanHalfFovY"] = volumeProjection.tanHalfFovY;

    mpPVVPass->execute(pRenderContext, mVolumeSize / 32, mVolumeSize, mVolumeDepth);
    pRenderContext->submit(true);

    std::vector<uint8_t> gvBytes = pRenderContext->readTextureSubresource(mpGVVolume.get(), 0);
    std::vector<uint8_t> pvvBytes = pRenderContext->readTextureSubresource(mpPVVVolume.get(), 0);

    writeVolumePair(gvBytes, pvvBytes, mDatasetName, index);
    if (mWriteDebugProjections)
    {
        const std::filesystem::path datasetRoot = mOutputRoot / mDatasetName;

        if (mExportMode == 0 || mExportMode == 2)
        {
            writeDebugProjections(datasetRoot, gvBytes, "gv", index);
        }

        if (mExportMode == 1 || mExportMode == 2)
        {
            writeDebugProjections(datasetRoot, pvvBytes, "pvv", index);
        }
    }

    gvBitCount = countSetBits(gvBytes);
    pvvBitCount = countSetBits(pvvBytes);
}

void NeuralPVSExporter::writeExportMetadata(
    const std::vector<ExportSample>& samples,
    const std::vector<uint64_t>& gvBitCounts,
    const std::vector<uint64_t>& pvvBitCounts,
    const AABB& sceneBounds,
    const float3& sceneCenter,
    const float3& sceneExtent,
    const float3& volumeExtent,
    const float3& sampleStep,
    bool useCameraFrustum
)
{
    const std::filesystem::path datasetRoot = mOutputRoot / mDatasetName;
    std::filesystem::create_directories(datasetRoot);

    std::ofstream metadata(datasetRoot / "metadata.json");
    metadata << "{\n";
    metadata << "  \"dataset_name\": \"" << mDatasetName << "\",\n";
    metadata << "  \"scene_path\": \"" << mScenePath.generic_string() << "\",\n";
    metadata << "  \"sampling_mode\": \"" << (mSamplingMode == 1 ? "path_csv" : "grid") << "\",\n";
    metadata << "  \"visibility_mode\": \"" << (useCameraFrustum ? "camera_frustum" : "view_cell") << "\",\n";
    metadata << "  \"volume_mapping\": \"" << (mVolumeMappingMode == 1 ? "unity_projection" : "world_aabb") << "\",\n";
    metadata << "  \"path_csv\": \"" << (mSamplingMode == 1 ? std::filesystem::path(mPathCsvText).generic_string() : "") << "\",\n";
    metadata << "  \"camera_aspect_ratio\": " << mCameraAspectRatio << ",\n";
    metadata << "  \"view_cell_radius\": " << mViewCellRadius << ",\n";
    metadata << "  \"view_cell_near\": " << mViewCellNearPlane << ",\n";
    metadata << "  \"view_cell_far\": " << mViewCellFarPlane << ",\n";
    metadata << "  \"sample_count\": " << samples.size() << ",\n";
    metadata << "  \"volume_size\": [" << mVolumeSize << ", " << mVolumeSize << ", " << mVolumeDepth << "],\n";
    metadata << "  \"scene_bounds_min\": [" << sceneBounds.minPoint.x << ", " << sceneBounds.minPoint.y << ", " << sceneBounds.minPoint.z << "],\n";
    metadata << "  \"scene_bounds_max\": [" << sceneBounds.maxPoint.x << ", " << sceneBounds.maxPoint.y << ", " << sceneBounds.maxPoint.z << "],\n";
    metadata << "  \"scene_center\": [" << sceneCenter.x << ", " << sceneCenter.y << ", " << sceneCenter.z << "],\n";
    metadata << "  \"scene_extent\": [" << sceneExtent.x << ", " << sceneExtent.y << ", " << sceneExtent.z << "],\n";
    metadata << "  \"volume_extent\": [" << volumeExtent.x << ", " << volumeExtent.y << ", " << volumeExtent.z << "],\n";
    metadata << "  \"sample_step\": [" << sampleStep.x << ", " << sampleStep.y << ", " << sampleStep.z << "],\n";
    metadata << "  \"samples\": [\n";

    for (size_t i = 0; i < samples.size(); ++i)
    {
        metadata << "    {";
        metadata << "\"index\": " << i << ", ";
        metadata << "\"center\": [" << samples[i].center.x << ", " << samples[i].center.y << ", " << samples[i].center.z << "], ";
        metadata << "\"has_camera\": " << (samples[i].hasCamera ? "true" : "false") << ", ";
        metadata << "\"forward\": [" << samples[i].forward.x << ", " << samples[i].forward.y << ", " << samples[i].forward.z << "], ";
        metadata << "\"fov_y_degrees\": " << samples[i].fovYDegrees << ", ";
        metadata << "\"gv_file\": \"gv/" << std::setw(4) << std::setfill('0') << i << "_gv.bin.gz\", ";
        metadata << "\"pvv_file\": \"pvv/" << std::setw(4) << std::setfill('0') << i << "_pvv.bin.gz\", ";
        metadata << "\"gv_set_bits\": " << (i < gvBitCounts.size() ? gvBitCounts[i] : 0u) << ", ";
        metadata << "\"pvv_set_bits\": " << (i < pvvBitCounts.size() ? pvvBitCounts[i] : 0u);
        metadata << "}";
        if (i + 1 < samples.size()) metadata << ",";
        metadata << "\n";
    }

    metadata << "  ]\n";
    metadata << "}\n";
}

void NeuralPVSExporter::exportSceneVolumes(RenderContext* pRenderContext)
{
    const AABB sceneBounds = mpScene->getSceneBounds();
    const float3 sceneCenter = sceneBounds.center();
    const float3 sceneExtent = max(sceneBounds.extent(), float3(0.0001f));

    const float3 volumeExtent = sceneExtent * mVolumeExtentScale;
    const float viewCellRadius = std::max(0.001f, mViewCellRadius);
    const float3 sampleStep = sceneExtent * mSampleStepScale;

    const bool useCameraFrustum = mVisibilityMode == 1;
    const std::vector<ExportSample> samples = buildExportSamples(sceneCenter, sceneExtent);
    validateExportSamples(samples, useCameraFrustum);

    std::vector<uint64_t> gvBitCounts;
    std::vector<uint64_t> pvvBitCounts;
    gvBitCounts.reserve(samples.size());
    pvvBitCounts.reserve(samples.size());

    for (uint32_t sampleIndex = 0; sampleIndex < static_cast<uint32_t>(samples.size()); ++sampleIndex)
    {
        uint64_t gvBitCount = 0;
        uint64_t pvvBitCount = 0;
        exportOneSample(pRenderContext, samples[sampleIndex], sampleIndex, volumeExtent, viewCellRadius, gvBitCount, pvvBitCount);
        gvBitCounts.push_back(gvBitCount);
        pvvBitCounts.push_back(pvvBitCount);

        mLastExportStatus = "Exported sample " + std::to_string(sampleIndex) + " / " + std::to_string(samples.size() - 1);
    }

    writeExportMetadata(samples, gvBitCounts, pvvBitCounts, sceneBounds, sceneCenter, sceneExtent, volumeExtent, sampleStep, useCameraFrustum);
    mLastExportStatus = "Exported " + std::to_string(samples.size()) + " " + mDatasetName + " samples with metadata.";
}

void NeuralPVSExporter::writeVolumePair(
    const std::vector<uint8_t>& gvBytes,
    const std::vector<uint8_t>& pvvBytes,
    const std::string& datasetName,
    uint32_t index
)
{
    const size_t expectedBytes = size_t(mVolumeSize) * size_t(mVolumeSize) * size_t(mVolumeDepth) / 8;
    if (gvBytes.size() != expectedBytes)
        throw std::runtime_error("Unexpected GV byte count: " + std::to_string(gvBytes.size()));
    if (pvvBytes.size() != expectedBytes)
        throw std::runtime_error("Unexpected PVV byte count: " + std::to_string(pvvBytes.size()));

    const std::filesystem::path datasetRoot = mOutputRoot / datasetName;

    if (mExportMode == 0 || mExportMode == 2)
    {
        writeVolumeFile(datasetRoot / "gv" / fourDigitName(index, "_gv.bin.gz"), gvBytes);
    }

    if (mExportMode == 1 || mExportMode == 2)
    {
        writeVolumeFile(datasetRoot / "pvv" / fourDigitName(index, "_pvv.bin.gz"), pvvBytes);
    }
}

void NeuralPVSExporter::writeVolumeFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    writeGzipStored(path, bytes);
}

void NeuralPVSExporter::writeDebugProjections(
    const std::filesystem::path& datasetRoot,
    const std::vector<uint8_t>& bytes,
    const std::string& label,
    uint32_t index
)
{
    const std::filesystem::path debugRoot = datasetRoot / "debug";
    std::filesystem::create_directories(debugRoot);

    auto getBit = [&](uint32_t x, uint32_t y, uint32_t z) -> bool
    {
        const uint64_t bitIndex =
            uint64_t(z) * uint64_t(mVolumeSize) * uint64_t(mVolumeSize) + uint64_t(y) * uint64_t(mVolumeSize) + uint64_t(x);

        const uint64_t byteIndex = bitIndex / 8;
        const uint32_t bitInByte = uint32_t(bitIndex % 8);
        return (bytes[byteIndex] & (1u << bitInByte)) != 0;
    };

    auto writePgm = [](const std::filesystem::path& path, const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height)
    {
        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            FALCOR_THROW("Failed to open debug image '{}'.", path.string());
        }

        file << "P5\n" << width << " " << height << "\n255\n";
        file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    };

    auto makeName = [&](const std::string& axis) -> std::filesystem::path
    {
        std::ostringstream stream;
        stream << label << "_" << std::setw(4) << std::setfill('0') << index << "_" << axis << ".pgm";
        return debugRoot / stream.str();
    };

    std::vector<uint8_t> xy(size_t(mVolumeSize) * size_t(mVolumeSize), 0);
    std::vector<uint8_t> xz(size_t(mVolumeSize) * size_t(mVolumeDepth), 0);
    std::vector<uint8_t> yz(size_t(mVolumeSize) * size_t(mVolumeDepth), 0);

    for (uint32_t z = 0; z < mVolumeDepth; ++z)
    {
        for (uint32_t y = 0; y < mVolumeSize; ++y)
        {
            for (uint32_t x = 0; x < mVolumeSize; ++x)
            {
                if (!getBit(x, y, z))
                    continue;

                xy[size_t(y) * size_t(mVolumeSize) + x] = 255;
                xz[size_t(z) * size_t(mVolumeSize) + x] = 255;
                yz[size_t(z) * size_t(mVolumeSize) + y] = 255;
            }
        }
    }

    writePgm(makeName("xy"), xy, mVolumeSize, mVolumeSize);
    writePgm(makeName("xz"), xz, mVolumeSize, mVolumeDepth);
    writePgm(makeName("yz"), yz, mVolumeSize, mVolumeDepth);
}

std::vector<uint8_t> NeuralPVSExporter::readGzipStoredFile(const std::filesystem::path& path) const
{
    const std::string decompressed = decompressFile(path);
    return std::vector<uint8_t>(decompressed.begin(), decompressed.end());
}

std::vector<NeuralPVSExporter::ExportSample> NeuralPVSExporter::loadPathSamples(const std::filesystem::path& path) const
{
    std::ifstream file(path);
    if (!file)
    {
        FALCOR_THROW("Failed to open path CSV '{}'.", path.string());
    }

    std::vector<ExportSample> samples;
    std::string line;
    std::vector<std::string> header;

    auto findColumn = [&](const std::string& name) -> int
    {
        for (size_t i = 0; i < header.size(); ++i)
        {
            if (toLower(header[i]) == name)
                return int(i);
        }
        return -1;
    };

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;

        std::vector<std::string> tokens = splitCsvLine(line);
        if (tokens.size() < 3)
            continue;

        float x = 0.f;
        float y = 0.f;
        float z = 0.f;
        if (!tryParseFloat(tokens[0], x) || !tryParseFloat(tokens[1], y) || !tryParseFloat(tokens[2], z))
        {
            header = tokens;
            continue;
        }

        ExportSample sample;
        sample.center = float3(x, y, z);

        int forwardX = findColumn("forward_x");
        int forwardY = findColumn("forward_y");
        int forwardZ = findColumn("forward_z");
        int fov = findColumn("fov");

        if (forwardX < 0 && tokens.size() >= 11)
        {
            forwardX = 7;
            forwardY = 8;
            forwardZ = 9;
            fov = 10;
        }

        if (forwardX >= 0 && forwardY >= 0 && forwardZ >= 0 && fov >= 0 &&
            size_t(std::max(std::max(forwardX, forwardY), std::max(forwardZ, fov))) < tokens.size())
        {
            float fx = 0.f;
            float fy = 0.f;
            float fz = -1.f;
            float fovYDegrees = 60.f;
            if (tryParseFloat(tokens[forwardX], fx) && tryParseFloat(tokens[forwardY], fy) &&
                tryParseFloat(tokens[forwardZ], fz) && tryParseFloat(tokens[fov], fovYDegrees))
            {
                sample.forward = normalizedOrDefault(float3(fx, fy, fz), float3(0.f, 0.f, -1.f));
                sample.fovYDegrees = fovYDegrees;
                sample.hasCamera = true;
            }
        }

        samples.push_back(sample);
    }

    if (samples.empty())
    {
        FALCOR_THROW("Path CSV '{}' did not contain any sample centers.", path.string());
    }

    return samples;
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "NeuralPVS Exporter";
    config.windowDesc.resizableWindow = true;

    NeuralPVSExporter app(config);
    return app.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
