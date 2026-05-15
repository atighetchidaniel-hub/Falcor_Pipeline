import Scene.Raster;

Texture3D<uint> gRenderPVVVolume;

cbuffer PreviewCB
{
    float4x4 gMainCamViewProj;
    float3 gRenderPVVVolumeMin;
    uint gRenderPVVVolumeSize;
    float3 gRenderPVVVolumeExtent;
    uint gRenderPVVVolumeDepth;
    uint gEnableRenderPVV;
    uint gPVVFilter;
    uint gRenderPVVSource;
    uint gKeepOutsidePVVInView;
    uint gRenderVolumeMappingMode;
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
};

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

bool loadPVVBit(uint3 voxel)
{
    if (voxel.x >= gRenderPVVVolumeSize || voxel.y >= gRenderPVVVolumeSize || voxel.z >= gRenderPVVVolumeDepth)
        return false;

    uint wordX = voxel.x / 32u;
    uint bitInWord = voxel.x % 32u;
    uint bitMask = 1u << (31u - bitInWord);

    return (gRenderPVVVolume.Load(int4(int(wordX), int(voxel.y), int(voxel.z), 0)) & bitMask) != 0u;
}

bool loadPVVBitClamped(int3 voxel)
{
    int maxX = int(gRenderPVVVolumeSize) - 1;
    int maxY = int(gRenderPVVVolumeSize) - 1;
    int maxZ = int(gRenderPVVVolumeDepth) - 1;
    if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 || voxel.x > maxX || voxel.y > maxY || voxel.z > maxZ)
        return false;

    return loadPVVBit(uint3(uint(voxel.x), uint(voxel.y), uint(voxel.z)));
}

bool isInsideMainCameraFrustum(float3 worldPos)
{
    float4 clip = mul(gMainCamViewProj, float4(worldPos, 1.0f));
    if (clip.w <= 0.0f)
        return false;

    float3 ndc = clip.xyz / clip.w;
    return ndc.x >= -1.0f && ndc.x <= 1.0f &&
           ndc.y >= -1.0f && ndc.y <= 1.0f &&
           ndc.z >= 0.0f && ndc.z <= 1.0f;
}

bool worldToVolume01(float3 worldPos, out float3 p)
{
    if (gRenderVolumeMappingMode == 0u)
    {
        p = (worldPos - gRenderPVVVolumeMin) / gRenderPVVVolumeExtent;
        return all(p >= 0.0f) && all(p <= 1.0f);
    }

    float3 rel = worldPos - gViewCellPosition;
    float depth = dot(rel, normalize(gViewCellForward));
    if (depth < gViewCellNearPlane || depth > gViewCellFarPlane)
        return false;

    float xNdc = dot(rel, normalize(gViewCellRight)) / max(1e-6f, depth * gTanHalfFovX);
    float yNdc = dot(rel, normalize(gViewCellUp)) / max(1e-6f, depth * gTanHalfFovY);
    p = float3(xNdc * 0.5f + 0.5f, yNdc * 0.5f + 0.5f, (depth - gViewCellNearPlane) / (gViewCellFarPlane - gViewCellNearPlane));

    return all(p >= 0.0f) && all(p <= 1.0f);
}

bool isVisibleInPVV(float3 worldPos)
{
    if (gEnableRenderPVV == 0u)
        return true;

    float3 p = float3(0.0f);
    if (!worldToVolume01(worldPos, p))
        return gKeepOutsidePVVInView != 0u && isInsideMainCameraFrustum(worldPos);

    float3 gridPos = p * float3(float(gRenderPVVVolumeSize), float(gRenderPVVVolumeSize), float(gRenderPVVVolumeDepth));
    uint x = min((uint)floor(gridPos.x), gRenderPVVVolumeSize - 1u);
    uint y = min((uint)floor(gridPos.y), gRenderPVVVolumeSize - 1u);
    uint z = min((uint)floor(gridPos.z), gRenderPVVVolumeDepth - 1u);

    if (gPVVFilter <= 1u)
        return loadPVVBit(uint3(x, y, z));

    if (gPVVFilter == 2u)
    {
        int3 center = int3(int(x), int(y), int(z));
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (loadPVVBitClamped(center + int3(dx, dy, dz)))
                        return true;
                }
            }
        }
        return false;
    }

    int3 baseVoxel = int3(floor(gridPos - 0.5f));
    for (int dz = 0; dz <= 1; ++dz)
    {
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                if (loadPVVBitClamped(baseVoxel + int3(dx, dy, dz)))
                    return true;
            }
        }
    }

    return false;
}

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    if (!isVisibleInPVV(vsOut.posW))
        discard;

    float3 n = normalize(vsOut.normalW);
    float3 lightDir = normalize(float3(-0.35f, 0.7f, -0.55f));
    float shade = saturate(dot(n, lightDir)) * 0.75f + 0.18f;
    float3 normalTint = abs(n) * 0.25f;

    return float4(float3(shade, shade, shade) + normalTint, 1.0f);
}
