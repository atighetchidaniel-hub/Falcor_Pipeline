import Scene.Raster;

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

float4 psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID) : SV_TARGET
{
    float3 n = normalize(vsOut.normalW);
    float3 lightDir = normalize(float3(-0.35f, 0.7f, -0.55f));
    float shade = saturate(dot(n, lightDir)) * 0.75f + 0.18f;
    float3 normalTint = abs(n) * 0.25f;

    return float4(float3(shade, shade, shade) + normalTint, 1.0f);
}
