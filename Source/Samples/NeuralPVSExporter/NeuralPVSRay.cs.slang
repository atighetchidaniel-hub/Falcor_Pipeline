import Scene.RaytracingInline;

Texture3D<uint> gGeometryVolume;
RWTexture3D<uint> gPVVVolume;

cbuffer PVVRayCB
{
    float3 gSceneMin;
    uint gVolumeSize;
    float3 gSceneExtent;
    uint gVolumeDepth;
    float3 gViewCellCenter;
    float gViewCellRadius;
    uint gSampleCount;
    uint gUseCameraFrustum;
    float gCameraFovYRadians;
    float gCameraAspectRatio;
    float3 gCameraForward;
    uint gUseProjectionVolume;
    float gViewCellNearPlane;
    float gViewCellFarPlane;
    float gTanHalfFovX;
    float gTanHalfFovY;
    float3 gViewCellPosition;
    uint gPadding0;
    float3 gViewCellForward;
    uint gPadding1;
    float3 gViewCellRight;
    uint gPadding2;
    float3 gViewCellUp;
    uint gPadding3;
};

float3 normalizedOrFallback(float3 v, float3 fallback)
{
    float len = length(v);
    return len > 1e-6f ? v / len : fallback;
}

float3 getSampleOffset(uint sampleIndex)
{
    if (gSampleCount <= 1u)
        return float3(0.f);

    float r = gViewCellRadius;
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

    const float3 right = normalizedOrFallback(gViewCellRight, float3(1.f, 0.f, 0.f));
    const float3 up = normalizedOrFallback(gViewCellUp, float3(0.f, 1.f, 0.f));
    const float3 forward = normalizedOrFallback(gViewCellForward, float3(0.f, 0.f, -1.f));
    return right * local.x + up * local.y + forward * local.z;
}

bool isInsideCameraFrustum(float3 origin, float3 voxelWorld)
{
    if (gUseCameraFrustum == 0u)
        return true;

    float3 toVoxel = voxelWorld - origin;
    float distance = length(toVoxel);
    if (distance <= 1e-5f)
        return false;

    float3 dir = toVoxel / distance;
    float3 forward = normalizedOrFallback(gCameraForward, normalizedOrFallback(gViewCellForward, float3(0.f, 0.f, -1.f)));
    float z = dot(dir, forward);
    if (z <= 1e-5f)
        return false;

    float3 right = normalizedOrFallback(gViewCellRight, float3(1.f, 0.f, 0.f));
    float3 up = normalizedOrFallback(gViewCellUp, float3(0.f, 1.f, 0.f));
    float tanHalfY = tan(0.5f * gCameraFovYRadians);
    float tanHalfX = tanHalfY * max(0.1f, gCameraAspectRatio);

    float x = dot(dir, right) / z;
    float y = dot(dir, up) / z;
    return abs(x) <= tanHalfX && abs(y) <= tanHalfY;
}

float3 voxelToWorld(uint x, uint y, uint z)
{
    float3 voxel01 = (float3(float(x), float(y), float(z)) + 0.5f) /
        float3(float(gVolumeSize), float(gVolumeSize), float(gVolumeDepth));

    if (gUseProjectionVolume == 0u)
        return gSceneMin + voxel01 * gSceneExtent;

    float depth = lerp(gViewCellNearPlane, gViewCellFarPlane, voxel01.z);
    float xNdc = voxel01.x * 2.0f - 1.0f;
    float yNdc = voxel01.y * 2.0f - 1.0f;

    return gViewCellPosition +
        normalizedOrFallback(gViewCellForward, float3(0.f, 0.f, -1.f)) * depth +
        normalizedOrFallback(gViewCellRight, float3(1.f, 0.f, 0.f)) * (xNdc * depth * gTanHalfFovX) +
        normalizedOrFallback(gViewCellUp, float3(0.f, 1.f, 0.f)) * (yNdc * depth * gTanHalfFovY);
}

[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint wordsX = gVolumeSize / 32u;
    if (id.x >= wordsX || id.y >= gVolumeSize || id.z >= gVolumeDepth)
        return;

    uint word = gGeometryVolume[int3(id)];
    if (word == 0u)
        return;

    SceneRayQuery<1> rayQuery;

    for (uint bit = 0u; bit < 32u; ++bit)
    {
        uint bitMask = 1u << (31u - bit);
        if ((word & bitMask) == 0u)
            continue;

        uint x = id.x * 32u + bit;
        float3 voxelWorld = voxelToWorld(x, id.y, id.z);

        bool visible = false;
        for (uint sampleIndex = 0u; sampleIndex < gSampleCount && !visible; ++sampleIndex)
        {
            float3 origin = gViewCellCenter + getSampleOffset(sampleIndex);
            float3 toVoxel = voxelWorld - origin;
            float distance = length(toVoxel);
            if (distance <= 0.001f)
                continue;

            if (!isInsideCameraFrustum(origin, voxelWorld))
                continue;

            Ray ray = Ray(origin, toVoxel / distance, 0.01f, distance * 0.98f);
            visible = rayQuery.traceVisibilityRay(ray);
        }

        if (visible)
            InterlockedOr(gPVVVolume[int3(id)], bitMask);
    }
}
