// URP: оператор Собеля (ті самі ваги, що в ЛР 7), шість вибірок на кожен напрямок.
Shader "Lab11/URP/Sobel"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        ZWrite Off Cull Off ZTest Always

        Pass
        {
            Name "Sobel"

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            // Яскравість за тими самими коефіцієнтами, що й у ЛР 5.
            float lum(float2 uv)
            {
                float3 c = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv).rgb;
                return dot(c, float3(0.299, 0.587, 0.114));
            }

            half4 Frag (Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float2 uv = input.texcoord;
                // x = 1/ширина, y = 1/висота (аналог _MainTex_TexelSize з Built-in).
                float2 t = _BlitTexture_TexelSize.xy;

                // Напрямок X: шість вибірок (стовпець з нулями пропущено).
                float gx = -1 * lum(uv + float2(-t.x, -t.y))
                           -2 * lum(uv + float2(-t.x,  0))
                           -1 * lum(uv + float2(-t.x,  t.y))
                           +1 * lum(uv + float2( t.x, -t.y))
                           +2 * lum(uv + float2( t.x,  0))
                           +1 * lum(uv + float2( t.x,  t.y));

                // Напрямок Y: також шість вибірок.
                float gy = -1 * lum(uv + float2(-t.x, -t.y))
                           -2 * lum(uv + float2( 0,   -t.y))
                           -1 * lum(uv + float2( t.x, -t.y))
                           +1 * lum(uv + float2(-t.x,  t.y))
                           +2 * lum(uv + float2( 0,    t.y))
                           +1 * lum(uv + float2( t.x,  t.y));

                // Повна сила краю.
                float e = sqrt(gx * gx + gy * gy);
                return half4(e, e, e, 1);
            }
            ENDHLSL
        }
    }
}
