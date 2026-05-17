Texture2D<float> gVisibilityDepthBuffer;
RWTexture3D<uint> gPVVVolume;

cbuffer PVVCB
{
    float4x4 gSampleInvViewProj;
    float3 gSceneMin;
    uint gVolumeSize;
    float3 gSceneExtent;
    uint gVolumeDepth;
    uint2 gDepthBufferSize;
    uint gUseProjectionVolume;
    uint gLinearZ;
    float3 gViewCellPosition;
    float gViewCellNearPlane;
    float3 gViewCellForward;
    float gViewCellFarPlane;
    float3 gViewCellRight;
    float gTanHalfFovX;
    float3 gViewCellUp;
    float gTanHalfFovY;
    float gLogDepthScale;
    uint3 gPadding0;
};

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
    float z = (depth - gViewCellNearPlane) / max(1e-6f, gViewCellFarPlane - gViewCellNearPlane);
    if (gLinearZ == 0u)
    {
        z = log2(gLogDepthScale * (depth / gViewCellNearPlane) + 1.0f) /
            log2(gLogDepthScale * (gViewCellFarPlane / gViewCellNearPlane) + 1.0f);
    }

    p = float3(xNdc * 0.5f + 0.5f, yNdc * 0.5f + 0.5f, z);
    return all(p >= 0.0f) && all(p <= 1.0f);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gDepthBufferSize.x || id.y >= gDepthBufferSize.y)
        return;

    float depth = gVisibilityDepthBuffer.Load(int3(int(id.x), int(id.y), 0));
    if (depth >= 1.0f || depth <= 0.0f)
        return;

    float ndcX = (float(id.x) + 0.5f) * 2.0f / float(gDepthBufferSize.x) - 1.0f;
    float ndcY = (float(id.y) + 0.5f) * 2.0f / float(gDepthBufferSize.y) - 1.0f;
    float4 worldPosH = mul(gSampleInvViewProj, float4(ndcX, ndcY, depth, 1.0f));
    float3 worldPos = worldPosH.xyz / worldPosH.w;

    float3 p = float3(0.0f);
    if (!worldToVolume01(worldPos, p))
        return;

    float volXf = p.x * float(gVolumeSize);
    float volYf = p.y * float(gVolumeSize);
    float volZf = p.z * float(gVolumeDepth);

    if (volXf < 0.0f || volXf >= float(gVolumeSize) ||
        volYf < 0.0f || volYf >= float(gVolumeSize) ||
        volZf < 0.0f || volZf >= float(gVolumeDepth))
    {
        return;
    }

    uint x = uint(trunc(volXf));
    uint y = uint(trunc(volYf));
    uint z = uint(trunc(volZf));
    uint wordX = x / 32u;
    uint bitInWord = x % 32u;
    uint bitMask = 1u << (31u - bitInWord);

    InterlockedOr(gPVVVolume[int3(wordX, y, z)], bitMask);
}
