// URP: роздільне розмиття Гауса 9x9 (сигма 1,8) двома проходами.
//   Pass Index 0 = горизонталь, Pass Index 1 = вертикаль.
// У Renderer Features додаються ДВА Full Screen Pass з одним матеріалом:
// перший з Pass Index 0, другий (нижче в списку) з Pass Index 1.
Shader "Lab11/URP/Blur"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        ZWrite Off Cull Off ZTest Always

        // Спільна частина для обох проходів.
        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

        // Ядро Гауса, сума ваг = 1, тому яскравість не змінюється.
        static const float W[9] = {
            0.01898, 0.05590, 0.12092, 0.19212, 0.22417,
            0.19212, 0.12092, 0.05590, 0.01898 };

        half4 blur(float2 uv, float2 dir)
        {
            half4 sum = 0;
            [unroll]
            for (int k = -4; k <= 4; k++)
                sum += W[k + 4] * SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + dir * k);
            return sum;
        }

        half4 FragH (Varyings input) : SV_Target
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            return blur(input.texcoord, float2(_BlitTexture_TexelSize.x, 0));
        }

        half4 FragV (Varyings input) : SV_Target
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            return blur(input.texcoord, float2(0, _BlitTexture_TexelSize.y));
        }
        ENDHLSL

        // Прохід 0: горизонталь
        Pass
        {
            Name "BlurH"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragH
            ENDHLSL
        }

        // Прохід 1: вертикаль
        Pass
        {
            Name "BlurV"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragV
            ENDHLSL
        }
    }
}
