import Scene.RaytracingInline;

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

Texture3D<uint> gRayGeometryVolume;
RWTexture3D<uint> gRayPVVVolume;

cbuffer PVVRayCB
{
    float3 gRaySceneMin;
    uint gRayVolumeSize;
    float3 gRaySceneExtent;
    uint gRayVolumeDepth;
    float3 gRayViewCellCenter;
    float gRayViewCellRadius;
    uint gRaySampleCount;
    uint gRayUseCameraFrustum;
    float gRayCameraFovYRadians;
    float gRayCameraAspectRatio;
    float3 gRayCameraForward;
    uint gRayUseProjectionVolume;
    float gRayViewCellNearPlane;
    float gRayViewCellFarPlane;
    float gRayTanHalfFovX;
    float gRayTanHalfFovY;
    float3 gRayViewCellPosition;
    uint gRayPadding0;
    float3 gRayViewCellForward;
    uint gRayPadding1;
    float3 gRayViewCellRight;
    uint gRayPadding2;
    float3 gRayViewCellUp;
    uint gRayPadding3;
};

float3 rayNormalizedOrFallback(float3 v, float3 fallback)
{
    float len = length(v);
    return len > 1e-6f ? v / len : fallback;
}

float3 getRaySampleOffset(uint sampleIndex)
{
    if (gRaySampleCount <= 1u)
        return float3(0.f);

    float r = gRayViewCellRadius;
    float3 local = float3(0.f);
    switch (sampleIndex)
    {
    case 0: local = float3(0.f, 0.f, 0.f); break;
    case 1: local = float3( r, 0.f, 0.f); break;
    case 2: local = float3(-r, 0.f, 0.f); break;
    case 3: local = float3(0.f,  r, 0.f); break;
    case 4: local = float3(0.f, -r, 0.f); break;
    case 5: local = float3(0.f, 0.f,  r); break;
    case 6: local = float3(0.f, 0.f, -r); break;
    case 7: local = float3(0.70710678f * r, 0.70710678f * r, 0.f); break;
    default: local = float3(-0.70710678f * r, -0.70710678f * r, 0.f); break;
    }

    const float3 right = rayNormalizedOrFallback(gRayViewCellRight, float3(1.f, 0.f, 0.f));
    const float3 up = rayNormalizedOrFallback(gRayViewCellUp, float3(0.f, 1.f, 0.f));
    const float3 forward = rayNormalizedOrFallback(gRayViewCellForward, float3(0.f, 0.f, -1.f));
    return right * local.x + up * local.y + forward * local.z;
}

bool isInsideRayCameraFrustum(float3 origin, float3 voxelWorld)
{
    if (gRayUseCameraFrustum == 0u)
        return true;

    float3 toVoxel = voxelWorld - origin;
    float distance = length(toVoxel);
    if (distance <= 1e-5f)
        return false;

    float3 dir = toVoxel / distance;
    float3 forward = rayNormalizedOrFallback(gRayCameraForward, rayNormalizedOrFallback(gRayViewCellForward, float3(0.f, 0.f, -1.f)));
    float z = dot(dir, forward);
    if (z <= 1e-5f)
        return false;

    float3 right = rayNormalizedOrFallback(gRayViewCellRight, float3(1.f, 0.f, 0.f));
    float3 up = rayNormalizedOrFallback(gRayViewCellUp, float3(0.f, 1.f, 0.f));
    float tanHalfY = tan(0.5f * gRayCameraFovYRadians);
    float tanHalfX = tanHalfY * max(0.1f, gRayCameraAspectRatio);

    float x = dot(dir, right) / z;
    float y = dot(dir, up) / z;
    return abs(x) <= tanHalfX && abs(y) <= tanHalfY;
}

float3 rayVoxelToWorld(uint x, uint y, uint z)
{
    float3 voxel01 = (float3(float(x), float(y), float(z)) + 0.5f) /
        float3(float(gRayVolumeSize), float(gRayVolumeSize), float(gRayVolumeDepth));

    if (gRayUseProjectionVolume == 0u)
        return gRaySceneMin + voxel01 * gRaySceneExtent;

    float depth = lerp(gRayViewCellNearPlane, gRayViewCellFarPlane, voxel01.z);
    float xNdc = voxel01.x * 2.0f - 1.0f;
    float yNdc = voxel01.y * 2.0f - 1.0f;

    return gRayViewCellPosition +
        rayNormalizedOrFallback(gRayViewCellForward, float3(0.f, 0.f, -1.f)) * depth +
        rayNormalizedOrFallback(gRayViewCellRight, float3(1.f, 0.f, 0.f)) * (xNdc * depth * gRayTanHalfFovX) +
        rayNormalizedOrFallback(gRayViewCellUp, float3(0.f, 1.f, 0.f)) * (yNdc * depth * gRayTanHalfFovY);
}

[numthreads(4, 4, 4)]
void rayMain(uint3 id : SV_DispatchThreadID)
{
    uint wordsX = gRayVolumeSize / 32u;
    if (id.x >= wordsX || id.y >= gRayVolumeSize || id.z >= gRayVolumeDepth)
        return;

    uint word = gRayGeometryVolume[int3(id)];
    if (word == 0u)
        return;

    SceneRayQuery<1> rayQuery;

    for (uint bit = 0u; bit < 32u; ++bit)
    {
        uint bitMask = 1u << (31u - bit);
        if ((word & bitMask) == 0u)
            continue;

        uint x = id.x * 32u + bit;
        float3 voxelWorld = rayVoxelToWorld(x, id.y, id.z);

        bool visible = false;
        for (uint sampleIndex = 0u; sampleIndex < gRaySampleCount && !visible; ++sampleIndex)
        {
            float3 origin = gRayViewCellCenter + getRaySampleOffset(sampleIndex);
            float3 toVoxel = voxelWorld - origin;
            float distance = length(toVoxel);
            if (distance <= 0.001f)
                continue;

            if (!isInsideRayCameraFrustum(origin, voxelWorld))
                continue;

            Ray ray = Ray(origin, toVoxel / distance, 0.01f, distance * 0.98f);
            visible = rayQuery.traceVisibilityRay(ray);
        }

        if (visible)
            InterlockedOr(gRayPVVVolume[int3(id)], bitMask);
    }
}
