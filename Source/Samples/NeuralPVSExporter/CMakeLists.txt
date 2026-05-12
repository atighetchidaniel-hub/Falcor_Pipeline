add_falcor_executable(NeuralPVSExporter)

target_sources(NeuralPVSExporter PRIVATE
    NeuralPVSExporter.cpp
    NeuralPVSExporter.h
    NeuralPVSGV.3d.slang
    NeuralPVSPVV.cs.slang
)

target_copy_shaders(NeuralPVSExporter Samples/NeuralPVSExporter)

target_source_group(NeuralPVSExporter "Samples")
