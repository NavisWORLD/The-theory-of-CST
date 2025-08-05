using UnityEngine;
using System.Linq;
#if HAS_FASTNOISELITE
using FastNoiseLite;
#endif

public class ProceduralMaterialGenerator : MonoBehaviour
{
    public MicAnalyzer micAnalyzer;
    private const float PHI = 1.6180339887f;
    private const int TEXTURE_SIZE = 256;

    void Start()
    {
        if (micAnalyzer == null)
        {
            micAnalyzer = FindFirstObjectByType<MicAnalyzer>();
            if (micAnalyzer == null)
            {
                Debug.LogError("⛔ ProceduralMaterialGenerator: MicAnalyzer not found!");
            }
        }
    }

    public Material GenerateMaterial(ShaderParams shaderParams, TextureParams textureParams, float psi = 0f, float lyapunov = 0f, float[] biomeMap = null)
    {
        Shader shader = Shader.Find("Standard");
        if (shader == null)
        {
            Debug.LogError("⛔ ProceduralMaterialGenerator: Standard shader not found!");
            return null;
        }

        try
        {
            Material material = new Material(shader);

            float psiNorm = Mathf.Clamp01(psi / 1e-10f);
            float lyapunovNorm = Mathf.Clamp01(lyapunov / 0.1f);
            float audioMod = micAnalyzer != null && float.IsFinite(micAnalyzer.rmsValue) ? micAnalyzer.rmsValue : 0f;
            float freqNorm = textureParams.freq_scale / 10.0f; // Normalized from freq_scale

            // Enhanced color variation
            Color baseColor = new Color(
                Mathf.Clamp(shaderParams.base_color[0] + freqNorm * 0.25f, 0f, 1f),
                Mathf.Clamp(shaderParams.base_color[1] + freqNorm * 0.15f, 0f, 1f),
                Mathf.Clamp(shaderParams.base_color[2] + freqNorm * 0.2f, 0f, 1f)
            );
            baseColor *= (1f + audioMod * 0.5f + psiNorm * 0.3f);
            baseColor = Color.Lerp(baseColor, Color.HSVToRGB(freqNorm, 0.9f, 1f), lyapunovNorm * 0.3f);

            material.SetColor("_Color", baseColor);
            material.SetColor("_EmissionColor", baseColor * (shaderParams.emission_power + audioMod + psiNorm));
            material.EnableKeyword("_EMISSION");
            material.SetFloat("_Metallic", Mathf.Lerp(0f, 0.5f, textureParams.entropy_scale + freqNorm * 0.2f));
            material.SetFloat("_Glossiness", Mathf.Lerp(0.3f, 1f, lyapunovNorm + audioMod * 0.2f));

            Texture2D texture = GenerateProceduralTexture(textureParams, psiNorm, lyapunovNorm, biomeMap, freqNorm);
            if (texture != null)
            {
                material.SetTexture("_MainTex", texture);
            }

            Debug.Log($"🖌 Generated material with color=({baseColor.r},{baseColor.g},{baseColor.b}), freqNorm={freqNorm}");
            return material;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ ProceduralMaterialGenerator: Failed to generate material: {e.Message}");
            return new Material(shader);
        }
    }

    private Texture2D GenerateProceduralTexture(TextureParams textureParams, float psiNorm, float lyapunovNorm, float[] biomeMap, float freqNorm)
    {
        Texture2D texture = new Texture2D(TEXTURE_SIZE, TEXTURE_SIZE);
        Color[] colors = new Color[TEXTURE_SIZE * TEXTURE_SIZE];

        bool isPlanetary = textureParams.noise_type == "planetary";
        float freqScale = textureParams.freq_scale * (1f + freqNorm);
        float entropyScale = textureParams.entropy_scale;
        float audioMod = micAnalyzer != null && float.IsFinite(micAnalyzer.rmsValue) ? micAnalyzer.rmsValue : 0f;

#if HAS_FASTNOISELITE
        FastNoiseLite noise = new FastNoiseLite();
        noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise.SetFractalType(FastNoiseLite.FractalType.FBm);
        noise.SetFractalOctaves(4);

        FastNoiseLite ridgedNoise = new FastNoiseLite();
        ridgedNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        ridgedNoise.SetFractalType(FastNoiseLite.FractalType.Ridged);
        ridgedNoise.SetFractalOctaves(3);
#endif

        for (int y = 0; y < TEXTURE_SIZE; y++)
        {
            for (int x = 0; x < TEXTURE_SIZE; x++)
            {
                int index = y * TEXTURE_SIZE + x;
                float u = (float)x / TEXTURE_SIZE;
                float v = (float)y / TEXTURE_SIZE;

                float n = 0f, r = 0f;
                float amplitude = 1f;
                float frequency = freqScale * (1f + psiNorm);
                float totalAmplitude = 0f;

                for (int octave = 0; octave < 4; octave++)
                {
#if HAS_FASTNOISELITE
                    noise.SetFrequency(frequency);
                    ridgedNoise.SetFrequency(frequency * 0.5f);
                    n += noise.GetNoise(u * TEXTURE_SIZE, v * TEXTURE_SIZE) * amplitude;
                    r += (1f - Mathf.Abs(ridgedNoise.GetNoise(u * TEXTURE_SIZE, v * TEXTURE_SIZE))) * amplitude * 0.3f;
#else
                    n += Mathf.PerlinNoise(u * TEXTURE_SIZE * frequency, v * TEXTURE_SIZE * frequency) * amplitude;
                    r += (1f - Mathf.Abs(Mathf.PerlinNoise(u * TEXTURE_SIZE * frequency * 0.5f, v * TEXTURE_SIZE * frequency * 0.5f))) * amplitude * 0.3f;
#endif
                    totalAmplitude += amplitude;
                    amplitude *= 0.5f;
                    frequency *= 2f;
                }
                float noiseValue = (n * 0.7f + r * 0.3f) / totalAmplitude;

                float biomeValue = biomeMap != null && index < biomeMap.Length ? biomeMap[index] : noiseValue;
                Color pixelColor;

                if (isPlanetary)
                {
                    float landThreshold = 0.5f + entropyScale * 0.1f - audioMod * 0.2f + psiNorm * 0.1f;
                    float hue;
                    if (noiseValue > landThreshold)
                    {
                        if (freqNorm < 0.25f) // Desert
                            hue = 0.1f + entropyScale * 0.05f + freqNorm * 0.1f;
                        else if (freqNorm < 0.5f) // Forest
                            hue = 0.3f + entropyScale * 0.1f + freqNorm * 0.15f;
                        else if (freqNorm < 0.75f) // Ocean
                            hue = 0.5f + entropyScale * 0.08f + freqNorm * 0.12f;
                        else // Tundra
                            hue = 0.6f + entropyScale * 0.05f + freqNorm * 0.1f;
                        pixelColor = Color.HSVToRGB(hue, 0.6f + biomeValue * 0.2f, noiseValue);
                    }
                    else
                    {
                        hue = 0.5f + audioMod * 0.1f + psiNorm * 0.05f + freqNorm * 0.2f;
                        pixelColor = Color.HSVToRGB(hue, 0.8f, noiseValue * 0.8f);
                    }
                    if (audioMod > 0.3f)
                    {
                        float cloudNoise = Mathf.PerlinNoise(u * TEXTURE_SIZE * 2f, v * TEXTURE_SIZE * 2f);
                        if (cloudNoise > 0.7f)
                            pixelColor = Color.Lerp(pixelColor, Color.white, audioMod * 0.5f);
                    }
                }
                else
                {
                    float hue = noiseValue * (0.1f + entropyScale * 0.2f + lyapunovNorm + freqNorm * 0.3f);
                    float value = Mathf.Clamp01(noiseValue * (1f + entropyScale * 0.5f));
                    pixelColor = Color.HSVToRGB(hue, 0.8f, value);
                }

                pixelColor *= Mathf.Lerp(0.5f, 1.5f, (noiseValue * PHI) % 1f);
                colors[index] = pixelColor;
            }
        }

        texture.SetPixels(colors);
        texture.Apply();
        return texture;
    }
}