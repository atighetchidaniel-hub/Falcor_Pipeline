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
    // Linear-Z: z = (depth - near) / (far - near).
    // Maps depth==near → 0, depth==far → 1. Consistent with the inverse lerp(near,far,z)
    // used in the ray/render path. Using (depth + near) would push far-plane voxels above 1.0
    // and cause them to be silently rejected by the all(p <= 1.0f) guard.
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
        // Use floor(p * size) clamped to [0, size-1] — consistent with PVV write and preview read.
        // The old trunc(p * (size-1)) formula shifted voxel addresses by ~half a voxel at mid-range,
        // causing a spatial mismatch between GV writes and PVV/preview reads.
        uint x = min((uint)floor(p.x * float(gVolumeSize)), gVolumeSize - 1u);
        uint y = min((uint)floor(p.y * float(gVolumeSize)), gVolumeSize - 1u);
        uint z = min((uint)floor(p.z * float(gVolumeDepth)), gVolumeDepth - 1u);

        uint wordX = x / 32;
        uint bitInWord = x % 32;
        uint bitMask = 1u << (31u - bitInWord);

        InterlockedOr(gGeometryVolume[int3(wordX, y, z)], bitMask);
    }

    return float4(p, 1.0f);
}
