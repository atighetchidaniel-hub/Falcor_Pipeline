#include "NeuralPVSExporter.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
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
}

NeuralPVSExporter::NeuralPVSExporter(const SampleAppConfig& config) : SampleApp(config) {}
NeuralPVSExporter::~NeuralPVSExporter() {}

void NeuralPVSExporter::onLoad(RenderContext* pRenderContext)
{
    loadScene(mScenePath);
    createResources();
    createGVPass();
    createPVVPass();
}

void NeuralPVSExporter::onShutdown() {}
void NeuralPVSExporter::onResize(uint32_t width, uint32_t height) {}

void NeuralPVSExporter::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    const float4 clearColor(0.08f, 0.10f, 0.12f, 1.0f);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
}

void NeuralPVSExporter::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "NeuralPVS Exporter", {680, 560}, {20, 40});
    renderGlobalUI(pGui);

    w.text("NeuralPVS Exporter");

    {
        auto group = w.group("Dataset", true);
        if (group)
        {
            w.textbox("Scene path", mScenePathText);
            w.textbox("Dataset name", mDatasetName);
            w.textbox("Output root", mOutputRootText);
        }
    }

    {
        auto group = w.group("Sampling", true);
        if (group)
        {
            w.var("Samples per axis", mSamplesPerAxis, 1u, 16u);
            w.var("Volume extent scale", mVolumeExtentScale, 0.05f, 2.0f, 0.01f);
            w.var("Sample step scale", mSampleStepScale, 0.001f, 0.5f, 0.001f);

            Gui::DropdownList exportModes = {
                {0, "GV only"},
                {1, "PVV only"},
                {2, "GV + PVV"},
                {3, "Metadata only"},
            };
            w.dropdown("Export mode", exportModes, mExportMode);
            w.checkbox("Write debug projections", mWriteDebugProjections);

            uint64_t totalSamples = uint64_t(mSamplesPerAxis) * uint64_t(mSamplesPerAxis) * uint64_t(mSamplesPerAxis);
            w.text("Total samples: " + std::to_string(totalSamples));
        }
    }

    w.separator();

    if (w.button("Load Scene"))
    {
        mScenePath = std::filesystem::path(mScenePathText);
        mOutputRoot = std::filesystem::path(mOutputRootText);

        loadScene(mScenePath);
        createResources();
        createGVPass();
        createPVVPass();

        mLastExportStatus = "Loaded scene: " + mScenePath.string();
    }

    if (w.button("Export Selected Mode"))
    {
        mScenePath = std::filesystem::path(mScenePathText);
        mOutputRoot = std::filesystem::path(mOutputRootText);

        loadScene(mScenePath);
        createResources();
        createGVPass();
        createPVVPass();

        exportSceneVolumes(getRenderContext());
    }

    w.separator();

    w.text("Scene: " + mScenePath.string());
    w.text("Output: " + (mOutputRoot / mDatasetName).string());
    w.text(mLastExportStatus);
}
bool NeuralPVSExporter::onKeyEvent(const KeyboardEvent& keyEvent) { return false; }
bool NeuralPVSExporter::onMouseEvent(const MouseEvent& mouseEvent) { return false; }
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

    Fbo::Desc desc;
    desc.setColorTarget(0, ResourceFormat::RGBA8Unorm);
    desc.setDepthStencilTarget(ResourceFormat::D32Float);
    mpGVFbo = Fbo::create2D(getDevice(), mRasterWidth, mRasterHeight, desc);
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

void NeuralPVSExporter::exportSceneVolumes(RenderContext* pRenderContext)
{
    const AABB sceneBounds = mpScene->getSceneBounds();
    const float3 sceneCenter = sceneBounds.center();
    const float3 sceneExtent = max(sceneBounds.extent(), float3(0.0001f));

    const float3 volumeExtent = sceneExtent * mVolumeExtentScale;
    const float viewCellRadius = std::max(0.1f, sceneBounds.radius() * 0.02f);
    const uint32_t samplesPerAxis = mSamplesPerAxis < 1u ? 1u : mSamplesPerAxis;

    std::vector<float3> centers;
    centers.reserve(size_t(samplesPerAxis) * size_t(samplesPerAxis) * size_t(samplesPerAxis));

    const float3 sampleStep = sceneExtent * mSampleStepScale;
    const float centerOffset = 0.5f * float(samplesPerAxis - 1u);

    for (uint32_t z = 0; z < samplesPerAxis; ++z)
    {
        for (uint32_t y = 0; y < samplesPerAxis; ++y)
        {
            for (uint32_t x = 0; x < samplesPerAxis; ++x)
            {
                const float3 offset = float3(float(x) - centerOffset, float(y) - centerOffset, float(z) - centerOffset);
                centers.push_back(sceneCenter + offset * sampleStep);
            }
        }
    }
    auto countSetBits = [](const std::vector<uint8_t>& bytes) -> uint64_t
    {
        uint64_t count = 0;
        for (uint8_t byte : bytes)
        {
            uint8_t v = byte;
            while (v != 0)
            {
                count += uint64_t(v & 1u);
                v >>= 1;
            }
        }
        return count;
    };

    std::vector<uint64_t> gvBitCounts;
    std::vector<uint64_t> pvvBitCounts;
    gvBitCounts.reserve(centers.size());
    pvvBitCounts.reserve(centers.size());
    for (uint32_t sampleIndex = 0; sampleIndex < static_cast<uint32_t>(centers.size()); ++sampleIndex)
    {
        const float3 viewCellCenter = centers[sampleIndex];
        const float3 volumeMin = viewCellCenter - volumeExtent * 0.5f;

        pRenderContext->clearUAV(mpGVVolume->getUAV().get(), uint4(0, 0, 0, 0));
        pRenderContext->clearUAV(mpPVVVolume->getUAV().get(), uint4(0, 0, 0, 0));
        pRenderContext->clearFbo(mpGVFbo.get(), float4(0, 0, 0, 0), 1.0f, 0, FboAttachmentType::All);

        auto gvRoot = mpGVPass->getRootVar();
        gvRoot["gGeometryVolume"] = mpGVVolume;
        gvRoot["ExporterCB"]["gSceneMin"] = volumeMin;
        gvRoot["ExporterCB"]["gSceneExtent"] = volumeExtent;
        gvRoot["ExporterCB"]["gVolumeSize"] = mVolumeSize;
        gvRoot["ExporterCB"]["gVolumeDepth"] = mVolumeDepth;

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
        pvvRoot["PVVCB"]["gSampleCount"] = 9u;

        mpPVVPass->execute(pRenderContext, mVolumeSize / 32, mVolumeSize, mVolumeDepth);
        pRenderContext->submit(true);

        std::vector<uint8_t> gvBytes = pRenderContext->readTextureSubresource(mpGVVolume.get(), 0);
        std::vector<uint8_t> pvvBytes = pRenderContext->readTextureSubresource(mpPVVVolume.get(), 0);

        writeVolumePair(gvBytes, pvvBytes, mDatasetName, sampleIndex);
        if (mWriteDebugProjections)
        {
            const std::filesystem::path datasetRoot = mOutputRoot / mDatasetName;

            if (mExportMode == 0 || mExportMode == 2)
            {
                writeDebugProjections(datasetRoot, gvBytes, "gv", sampleIndex);
            }

            if (mExportMode == 1 || mExportMode == 2)
            {
                writeDebugProjections(datasetRoot, pvvBytes, "pvv", sampleIndex);
            }
        }
        gvBitCounts.push_back(countSetBits(gvBytes));
        pvvBitCounts.push_back(countSetBits(pvvBytes));

        mLastExportStatus = "Exported sample " + std::to_string(sampleIndex) + " / " + std::to_string(centers.size() - 1);
    }
    const std::filesystem::path datasetRoot = mOutputRoot / mDatasetName;
    std::filesystem::create_directories(datasetRoot);

    std::ofstream metadata(datasetRoot / "metadata.json");
    metadata << "{\n";
    metadata << "  \"dataset_name\": \"" << mDatasetName << "\",\n";
    metadata << "  \"scene_path\": \"" << mScenePath.generic_string() << "\",\n";
    metadata << "  \"sample_count\": " << centers.size() << ",\n";
    metadata << "  \"volume_size\": [" << mVolumeSize << ", " << mVolumeSize << ", " << mVolumeDepth << "],\n";
    metadata << "  \"scene_bounds_min\": [" << sceneBounds.minPoint.x << ", " << sceneBounds.minPoint.y << ", " << sceneBounds.minPoint.z << "],\n";
    metadata << "  \"scene_bounds_max\": [" << sceneBounds.maxPoint.x << ", " << sceneBounds.maxPoint.y << ", " << sceneBounds.maxPoint.z << "],\n";
    metadata << "  \"scene_center\": [" << sceneCenter.x << ", " << sceneCenter.y << ", " << sceneCenter.z << "],\n";
    metadata << "  \"scene_extent\": [" << sceneExtent.x << ", " << sceneExtent.y << ", " << sceneExtent.z << "],\n";
    metadata << "  \"volume_extent\": [" << volumeExtent.x << ", " << volumeExtent.y << ", " << volumeExtent.z << "],\n";
    metadata << "  \"sample_step\": [" << sampleStep.x << ", " << sampleStep.y << ", " << sampleStep.z << "],\n";
    metadata << "  \"samples\": [\n";

    for (size_t i = 0; i < centers.size(); ++i)
    {
        metadata << "    {";
        metadata << "\"index\": " << i << ", ";
        metadata << "\"center\": [" << centers[i].x << ", " << centers[i].y << ", " << centers[i].z << "], ";
        metadata << "\"gv_file\": \"gv/" << std::setw(4) << std::setfill('0') << i << "_gv.bin.gz\", ";
        metadata << "\"pvv_file\": \"pvv/" << std::setw(4) << std::setfill('0') << i << "_pvv.bin.gz\", ";
        metadata << "\"gv_set_bits\": " << gvBitCounts[i] << ", ";
        metadata << "\"pvv_set_bits\": " << pvvBitCounts[i];
        metadata << "}";
        if (i + 1 < centers.size()) metadata << ",";
        metadata << "\n";
    }

    metadata << "  ]\n";
    metadata << "}\n";

    mLastExportStatus = "Exported " + std::to_string(centers.size()) + " " + mDatasetName + " samples with metadata.";
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









