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
    uint gVolumeMappingMode;  // 0 = world AABB, 1 = projection frustum
    uint gLinearZ;
    float gLogDepthScale;
    uint gPadding0;
    float gViewCellNearPlane;
    float gViewCellFarPlane;
    float gTanHalfFovX;
    float gTanHalfFovY;
    float3 gViewCellPosition;
    uint gPadding1;
    float3 gViewCellForward;
    uint gPadding2;
    float3 gViewCellRight;
    uint gPadding3;
    float3 gViewCellUp;
    uint gPadding4;
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

// Returns the world-space AABB of the view frustum for ray-march bounds in projection mode.
void getFrustumAABB(out float3 boxMin, out float3 boxMax)
{
    float3 fwd   = normalize(gViewCellForward);
    float3 right = normalize(gViewCellRight);
    float3 up    = normalize(gViewCellUp);
    float3 pos   = gViewCellPosition;
    float nearZ  = gViewCellNearPlane;
    float farZ   = gViewCellFarPlane;
    float tx     = gTanHalfFovX;
    float ty     = gTanHalfFovY;

    // 8 frustum corners
    float3 corners[8];
    corners[0] = pos + fwd * nearZ + right * ( nearZ * tx) + up * ( nearZ * ty);
    corners[1] = pos + fwd * nearZ + right * ( nearZ * tx) + up * (-nearZ * ty);
    corners[2] = pos + fwd * nearZ + right * (-nearZ * tx) + up * ( nearZ * ty);
    corners[3] = pos + fwd * nearZ + right * (-nearZ * tx) + up * (-nearZ * ty);
    corners[4] = pos + fwd * farZ  + right * ( farZ  * tx) + up * ( farZ  * ty);
    corners[5] = pos + fwd * farZ  + right * ( farZ  * tx) + up * (-farZ  * ty);
    corners[6] = pos + fwd * farZ  + right * (-farZ  * tx) + up * ( farZ  * ty);
    corners[7] = pos + fwd * farZ  + right * (-farZ  * tx) + up * (-farZ  * ty);

    boxMin = corners[0];
    boxMax = corners[0];
    for (int i = 1; i < 8; ++i)
    {
        boxMin = min(boxMin, corners[i]);
        boxMax = max(boxMax, corners[i]);
    }
}

// Maps a world position to [0,1]^3 volume coordinates.
// Mirrors worldToVolume01 in NeuralPVSGV.3d.slang and NeuralPVSPVV.cs.slang.
bool worldToVolume01(float3 worldPos, out float3 p)
{
    if (gVolumeMappingMode == 0u)
    {
        p = (worldPos - gVolumeMin) / gVolumeExtent;
        return all(p >= 0.0f) && all(p <= 1.0f);
    }

    float3 rel = worldPos - gViewCellPosition;
    float depth = dot(rel, normalize(gViewCellForward));
    if (depth < gViewCellNearPlane || depth > gViewCellFarPlane)
    {
        p = float3(0.0f);
        return false;
    }

    float xNdc = dot(rel, normalize(gViewCellRight)) / max(1e-6f, depth * gTanHalfFovX);
    float yNdc = dot(rel, normalize(gViewCellUp))    / max(1e-6f, depth * gTanHalfFovY);

    // Linear-Z: z = (depth - near) / (far - near). Keep in sync with GV and PVV shaders.
    float z = (depth - gViewCellNearPlane) / max(1e-6f, gViewCellFarPlane - gViewCellNearPlane);
    if (gLinearZ == 0u)
    {
        z = log2(gLogDepthScale * (depth / gViewCellNearPlane) + 1.0f) /
            log2(gLogDepthScale * (gViewCellFarPlane / gViewCellNearPlane) + 1.0f);
    }

    p = float3(xNdc * 0.5f + 0.5f, yNdc * 0.5f + 0.5f, z);
    return all(p >= 0.0f) && all(p <= 1.0f);
}

bool isOccupied(float3 worldPos)
{
    float3 p = float3(0.0f);
    if (!worldToVolume01(worldPos, p))
        return false;

    // Use floor(p * size) clamped — consistent with GV write (after fix) and PVV/preview reads.
    uint x = min((uint)floor(p.x * float(gVolumeSize)), gVolumeSize - 1u);
    uint y = min((uint)floor(p.y * float(gVolumeSize)), gVolumeSize - 1u);
    uint z = min((uint)floor(p.z * float(gVolumeDepth)), gVolumeDepth - 1u);
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

    float3 boxMin;
    float3 boxMax;
    if (gVolumeMappingMode == 0u)
    {
        boxMin = gVolumeMin;
        boxMax = gVolumeMin + gVolumeExtent;
    }
    else
    {
        getFrustumAABB(boxMin, boxMax);
    }

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
