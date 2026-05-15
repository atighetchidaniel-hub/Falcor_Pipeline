import Scene.RaytracingInline;

Texture3D<uint> gGeometryVolume;
RWTexture3D<uint> gPVVVolume;

cbuffer PVVCB
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
    uint gPadding0;
    uint gUseProjectionVolume;
    float gViewCellNearPlane;
    float gViewCellFarPlane;
    float gTanHalfFovX;
    float3 gViewCellPosition;
    float gTanHalfFovY;
    float3 gViewCellForward;
    uint gPadding1;
    float3 gViewCellRight;
    uint gPadding2;
    float3 gViewCellUp;
    uint gPadding3;
};

float3 getSampleOffset(uint sampleIndex)
{
    float r = gViewCellRadius;
    switch (sampleIndex)
    {
    case 0: return float3(0, 0, 0);
    case 1: return float3( r, 0, 0);
    case 2: return float3(-r, 0, 0);
    case 3: return float3(0,  r, 0);
    case 4: return float3(0, -r, 0);
    case 5: return float3(0, 0,  r);
    case 6: return float3(0, 0, -r);
    case 7: return float3( r,  r, 0);
    default: return float3(-r, -r, 0);
    }
}

bool isInsideCameraFrustum(float3 origin, float3 voxelWorld)
{
    if (gUseCameraFrustum == 0u) return true;

    float3 toVoxel = voxelWorld - origin;
    float distance = length(toVoxel);
    if (distance <= 0.001f) return false;

    float3 forward = normalize(gCameraForward);
    float3 dir = toVoxel / distance;
    float depth = dot(dir, forward);
    if (depth <= 0.0f) return false;

    float3 upHint = abs(forward.y) > 0.98f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 right = normalize(cross(upHint, forward));
    float3 up = cross(forward, right);

    float tanHalfY = tan(0.5f * gCameraFovYRadians);
    float tanHalfX = tanHalfY * gCameraAspectRatio;
    float horizontal = abs(dot(dir, right) / depth);
    float vertical = abs(dot(dir, up) / depth);

    return horizontal <= tanHalfX && vertical <= tanHalfY;
}

float3 voxelToWorld(uint x, uint y, uint z)
{
    float3 voxel01 = (float3(float(x), float(y), float(z)) + 0.5f) / float3(float(gVolumeSize), float(gVolumeSize), float(gVolumeDepth));
    if (gUseProjectionVolume == 0u)
        return gSceneMin + voxel01 * gSceneExtent;

    float depth = lerp(gViewCellNearPlane, gViewCellFarPlane, voxel01.z);
    float xNdc = voxel01.x * 2.0f - 1.0f;
    float yNdc = voxel01.y * 2.0f - 1.0f;

    return gViewCellPosition +
        normalize(gViewCellForward) * depth +
        normalize(gViewCellRight) * (xNdc * depth * gTanHalfFovX) +
        normalize(gViewCellUp) * (yNdc * depth * gTanHalfFovY);
}

[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint wordsX = gVolumeSize / 32;
    if (id.x >= wordsX || id.y >= gVolumeSize || id.z >= gVolumeDepth) return;

    uint word = gGeometryVolume[int3(id)];
    if (word == 0u) return;

    SceneRayQuery<1> rayQuery;

    for (uint bit = 0; bit < 32; ++bit)
    {
        uint bitMask = 1u << (31u - bit);
        if ((word & bitMask) == 0u) continue;

        uint x = id.x * 32u + bit;
        float3 voxelWorld = voxelToWorld(x, id.y, id.z);

        bool visible = false;
        for (uint sampleIndex = 0; sampleIndex < gSampleCount && !visible; ++sampleIndex)
        {
            float3 origin = gViewCellCenter + getSampleOffset(sampleIndex);
            float3 toVoxel = voxelWorld - origin;
            float distance = length(toVoxel);
            if (distance <= 0.001f) continue;
            if (!isInsideCameraFrustum(origin, voxelWorld)) continue;

            Ray ray = Ray(origin, toVoxel / distance, 0.01f, distance * 0.98f);
            visible = rayQuery.traceVisibilityRay(ray);
        }

        if (visible) InterlockedOr(gPVVVolume[int3(id)], bitMask);
    }
}
