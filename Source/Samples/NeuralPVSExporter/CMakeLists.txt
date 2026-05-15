add_falcor_executable(NeuralPVSExporter)

target_sources(NeuralPVSExporter PRIVATE
    NeuralPVSExporter.cpp
    NeuralPVSExporter.h
    NeuralPVSPreview.3d.slang
    NeuralPVSGV.3d.slang
    NeuralPVSPVV.cs.slang
    NeuralPVSRenderPVV.ps.slang
)

target_copy_shaders(NeuralPVSExporter Samples/NeuralPVSExporter)

target_source_group(NeuralPVSExporter "Samples")
