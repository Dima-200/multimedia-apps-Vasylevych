using UnityEngine;

// Постобробка у вбудованому конвеєрі (Built-in): повісити на камеру.
// Один скрипт для всіх матеріалів цієї роботи:
//   - шейдер з одним проходом (Passthrough, Sobel, BlurOnePass) -> один Blit;
//   - шейдер з двома проходами (Blur) -> горизонталь у тимчасову текстуру, потім вертикаль.
[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class PostEffect : MonoBehaviour
{
    public Material material;

    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        // Поки матеріал не призначено, просто віддаємо кадр як є (без чорного екрана).
        if (material == null)
        {
            Graphics.Blit(src, dst);
            return;
        }

        if (material.passCount < 2)
        {
            Graphics.Blit(src, dst, material);
            return;
        }

        // Проміжна текстура береться з пулу Unity й обов'язково повертається.
        RenderTexture tmp = RenderTexture.GetTemporary(
            src.width, src.height, 0, src.format);

        Graphics.Blit(src, tmp, material, 0); // прохід 0: горизонталь
        Graphics.Blit(tmp, dst, material, 1); // прохід 1: вертикаль

        RenderTexture.ReleaseTemporary(tmp);
    }
}
