Texture3D<uint> gVolume;

cbuffer RenderCB
{
    float4x4 gInvViewProj;
    float3 gCameraPos;
    uint gVolumeSize;
    float3 gVolumeMin;
    uint gVolumeDepth;
    float3 gVolumeExtent;
    float gOpacity;
    float3 gSampleCenter;
    float gStepScale;
    uint gVolumeKind;
};

float safeRcp(float value)
{
    return abs(value) > 1e-6f ? 1.0f / value : (value < 0.0f ? -1e20f : 1e20f);
}

bool intersectBox(float3 origin, float3 direction, float3 boxMin, float3 boxMax, out float tNear, out float tFar)
{
    float3 invDir = float3(safeRcp(direction.x), safeRcp(direction.y), safeRcp(direction.z));
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);

    tNear = max(max(tMin.x, tMin.y), tMin.z);
    tFar = min(min(tMax.x, tMax.y), tMax.z);
    return tFar >= max(tNear, 0.0f);
}

bool isOccupied(float3 worldPos)
{
    float3 p = (worldPos - gVolumeMin) / gVolumeExtent;
    if (any(p < 0.0f) || any(p > 1.0f))
        return false;

    uint x = min((uint)floor(p.x * (float)gVolumeSize), gVolumeSize - 1u);
    uint y = min((uint)floor(p.y * (float)gVolumeSize), gVolumeSize - 1u);
    uint z = min((uint)floor(p.z * (float)gVolumeDepth), gVolumeDepth - 1u);
    uint wordX = x / 32u;
    uint bitInWord = x % 32u;
    uint bitMask = 1u << (31u - bitInWord);

    return (gVolume.Load(int4(int(wordX), int(y), int(z), 0)) & bitMask) != 0u;
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET
{
    float2 clip = float2(texC.x * 2.0f - 1.0f, 1.0f - texC.y * 2.0f);
    float4 farH = mul(gInvViewProj, float4(clip, 1.0f, 1.0f));
    float3 farW = farH.xyz / farH.w;
    float3 rayDir = normalize(farW - gCameraPos);

    float3 boxMin = gVolumeMin;
    float3 boxMax = gVolumeMin + gVolumeExtent;
    float tNear;
    float tFar;
    if (!intersectBox(gCameraPos, rayDir, boxMin, boxMax, tNear, tFar))
        return float4(0.0f);

    tNear = max(tNear, 0.0f);
    float distance = tFar - tNear;
    if (distance <= 0.0f)
        return float4(0.0f);

    float maxExtent = max(max(gVolumeExtent.x, gVolumeExtent.y), gVolumeExtent.z);
    float stepLength = max(1e-4f, maxExtent / max(1.0f, float(gVolumeSize)) * max(0.25f, gStepScale));
    uint stepCount = min(2048u, max(1u, (uint)ceil(distance / stepLength)));
    stepLength = distance / float(stepCount);

    float hitWeight = 0.0f;
    float firstHit = distance;

    for (uint i = 0u; i < stepCount; ++i)
    {
        float t = tNear + (float(i) + 0.5f) * stepLength;
        float3 p = gCameraPos + rayDir * t;

        if (isOccupied(p))
        {
            float normalizedDepth = saturate((t - tNear) / distance);
            hitWeight += lerp(1.0f, 0.35f, normalizedDepth);
            firstHit = min(firstHit, t - tNear);

            if (hitWeight > 32.0f)
                break;
        }
    }

    if (hitWeight <= 0.0f)
        return float4(0.0f);

    float density = saturate(1.0f - exp(-hitWeight * 0.18f));
    float depth = saturate(firstHit / distance);

    float3 pvvColor = lerp(float3(0.10f, 0.95f, 1.0f), float3(0.02f, 0.28f, 1.0f), depth);
    float3 gvColor = lerp(float3(1.0f, 0.72f, 0.10f), float3(1.0f, 0.24f, 0.02f), depth);
    float3 color = gVolumeKind == 1u ? pvvColor : gvColor;
    float alpha = saturate(gOpacity * density);

    return float4(color, alpha);
}
