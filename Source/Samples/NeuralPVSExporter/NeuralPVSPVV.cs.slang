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
        float3 voxel01 = (float3(float(x), float(id.y), float(id.z)) + 0.5f) / float3(float(gVolumeSize), float(gVolumeSize), float(gVolumeDepth));
        float3 voxelWorld = gSceneMin + voxel01 * gSceneExtent;

        bool visible = false;
        for (uint sampleIndex = 0; sampleIndex < gSampleCount && !visible; ++sampleIndex)
        {
            float3 origin = gViewCellCenter + getSampleOffset(sampleIndex);
            float3 toVoxel = voxelWorld - origin;
            float distance = length(toVoxel);
            if (distance <= 0.001f) continue;

            Ray ray = Ray(origin, toVoxel / distance, 0.01f, distance * 0.98f);
            visible = rayQuery.traceVisibilityRay(ray);
        }

        if (visible) InterlockedOr(gPVVVolume[int3(id)], bitMask);
    }
}
