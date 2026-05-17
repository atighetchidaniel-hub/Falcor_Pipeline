import Scene.Raster;

RWTexture3D<uint> gGeometryVolume;

cbuffer ExporterCB
{
    float3 gSceneMin;
    uint gVolumeSize;
    float3 gSceneExtent;
    uint gVolumeDepth;
    uint gUseProjectionVolume;
    float gViewCellNearPlane;
    float gViewCellFarPlane;
    float gTanHalfFovX;
    float3 gViewCellPosition;
    float gTanHalfFovY;
    float3 gViewCellForward;
    uint gPadding0;
    float3 gViewCellRight;
    uint gPadding1;
    float3 gViewCellUp;
    uint gPadding2;
    uint gLinearZ;
    float gLogDepthScale;
    uint2 gPadding3;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

bool worldToVolume01(float3 worldPos, out float3 p)
{
    if (gUseProjectionVolume == 0u)
    {
        p = (worldPos - gSceneMin) / gSceneExtent;
        return all(p >= 0.0f) && all(p <= 1.0f);
    }

    float3 rel = worldPos - gViewCellPosition;
    float depth = dot(rel, normalize(gViewCellForward));
    if (depth < gViewCellNearPlane || depth > gViewCellFarPlane)
        return false;

    float xNdc = dot(rel, normalize(gViewCellRight)) / max(1e-6f, depth * gTanHalfFovX);
    float yNdc = dot(rel, normalize(gViewCellUp)) / max(1e-6f, depth * gTanHalfFovY);
    float z = (depth - gViewCellNearPlane) / (gViewCellFarPlane - gViewCellNearPlane);
    if (gLinearZ == 0u)
    {
        z = log2(gLogDepthScale * (depth / gViewCellNearPlane) + 1.0f) /
            log2(gLogDepthScale * (gViewCellFarPlane / gViewCellNearPlane) + 1.0f);
    }

    p = float3(xNdc * 0.5f + 0.5f, yNdc * 0.5f + 0.5f, z);

    return all(p >= 0.0f) && all(p <= 1.0f);
}

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    float3 p = float3(0.0f);
    if (worldToVolume01(vsOut.posW, p))
    {
        float3 dims = float3(float(gVolumeSize), float(gVolumeSize), float(gVolumeDepth)) - 1.0f;
        uint x = (uint)trunc(p.x * dims.x);
        uint y = (uint)trunc(p.y * dims.y);
        uint z = (uint)trunc(p.z * dims.z);

        uint wordX = x / 32;
        uint bitInWord = x % 32;
        uint bitMask = 1u << (31u - bitInWord);

        InterlockedOr(gGeometryVolume[int3(wordX, y, z)], bitMask);
    }

    return float4(p, 1.0f);
}
