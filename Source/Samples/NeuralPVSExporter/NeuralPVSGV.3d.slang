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
    uint gGVDilationRadius;
    uint gPadding3;
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

float3 applyNeuralPVSVolumeConvention(float3 p)
{
    return float3(p.y, p.x, p.z);
}

bool worldToVolume01(float3 worldPos, out float3 p)
{
    if (gUseProjectionVolume == 0u)
    {
        p = (worldPos - gSceneMin) / gSceneExtent;
        if (!all(p >= 0.0f) || !all(p <= 1.0f))
            return false;

        p = applyNeuralPVSVolumeConvention(p);
        return true;
    }

    float3 rel = worldPos - gViewCellPosition;
    float depth = dot(rel, normalize(gViewCellForward));
    if (depth < gViewCellNearPlane || depth > gViewCellFarPlane)
        return false;

    float xNdc = dot(rel, normalize(gViewCellRight)) / max(1e-6f, depth * gTanHalfFovX);
    float yNdc = dot(rel, normalize(gViewCellUp)) / max(1e-6f, depth * gTanHalfFovY);
    // Linear-Z: z = (depth + near) / (far - near).
    // Matches Unity's GetVolumeProjectionMatrix which sets P[2,3] = +n/(f-n),
    // making clip.z (used directly, without /w) equal to (d+n)/(f-n).
    float z = (depth + gViewCellNearPlane) / (gViewCellFarPlane - gViewCellNearPlane);
    if (gLinearZ == 0u)
    {
        z = log2(gLogDepthScale * (depth / gViewCellNearPlane) + 1.0f) /
            log2(gLogDepthScale * (gViewCellFarPlane / gViewCellNearPlane) + 1.0f);
    }

    p = float3(xNdc * 0.5f + 0.5f, yNdc * 0.5f + 0.5f, z);
    if (!all(p >= 0.0f) || !all(p <= 1.0f))
        return false;

    p = applyNeuralPVSVolumeConvention(p);
    return true;
}

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    float3 p = float3(0.0f);
    if (worldToVolume01(vsOut.posW, p))
    {
        // Use floor(p * size) clamped to [0, size-1] — consistent with PVV write and preview read.
        // The old trunc(p * (size-1)) formula shifted voxel addresses by ~half a voxel at mid-range,
        // causing a spatial mismatch between GV writes and PVV/preview reads.
        int x = int(min((uint)floor(p.x * float(gVolumeSize)), gVolumeSize - 1u));
        int y = int(min((uint)floor(p.y * float(gVolumeSize)), gVolumeSize - 1u));
        int z = int(min((uint)floor(p.z * float(gVolumeDepth)), gVolumeDepth - 1u));
        int radius = int(min(gGVDilationRadius, 1u));

        for (int dz = -radius; dz <= radius; ++dz)
        {
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int vx = x + dx;
                    int vy = y + dy;
                    int vz = z + dz;
                    if (vx < 0 || vy < 0 || vz < 0 ||
                        vx >= int(gVolumeSize) || vy >= int(gVolumeSize) || vz >= int(gVolumeDepth))
                    {
                        continue;
                    }

                    uint wordX = uint(vx) / 32u;
                    uint bitInWord = uint(vx) % 32u;
                    uint bitMask = 1u << (31u - bitInWord);

                    InterlockedOr(gGeometryVolume[int3(int(wordX), vy, vz)], bitMask);
                }
            }
        }
    }

    return float4(p, 1.0f);
}
