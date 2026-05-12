import Scene.Raster;

RWTexture3D<uint> gGeometryVolume;

cbuffer ExporterCB
{
    float3 gSceneMin;
    uint gVolumeSize;
    float3 gSceneExtent;
    uint gVolumeDepth;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    float3 p = (vsOut.posW - gSceneMin) / gSceneExtent;

    if (all(p >= 0.0f) && all(p <= 1.0f))
    {
        uint x = min((uint)floor(p.x * (float)gVolumeSize), gVolumeSize - 1);
        uint y = min((uint)floor(p.y * (float)gVolumeSize), gVolumeSize - 1);
        uint z = min((uint)floor(p.z * (float)gVolumeDepth), gVolumeDepth - 1);

        uint wordX = x / 32;
        uint bitInWord = x % 32;
        uint bitMask = 1u << (31u - bitInWord);

        InterlockedOr(gGeometryVolume[int3(wordX, y, z)], bitMask);
    }

    return float4(p, 1.0f);
}
