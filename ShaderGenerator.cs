using UnityEngine;
#if HAS_FASTNOISELITE
using FastNoiseLite;
#endif

public class ShaderGenerator : MonoBehaviour
{
    private const float PHI = 1.6180339887f;
    public Shader starShader;
    public Shader blackHoleShader;
    public Shader nebulaShader;
    private Shader fallbackShader;

    void Start()
    {
        fallbackShader = Shader.Find("Standard");
        if (fallbackShader == null)
        {
            Debug.LogError("⛔ ShaderGenerator: Standard shader not found!");
            return;
        }

        starShader = ValidateShader(starShader, "Custom/Star", "Star");
        blackHoleShader = ValidateShader(blackHoleShader, "Custom/BlackHole", "BlackHole");
        nebulaShader = ValidateShader(nebulaShader, "Custom/Nebula", "Nebula");

        bool anyMissing = starShader == fallbackShader && blackHoleShader == fallbackShader && nebulaShader == fallbackShader;
        if (anyMissing)
        {
            Debug.LogWarning("⚠ ShaderGenerator: Using Standard shader for all entity types");
        }
        else
        {
            Debug.Log("✅ ShaderGenerator: Custom shaders initialized");
        }
    }

    private Shader ValidateShader(Shader shader, string shaderName, string logName)
    {
        if (shader != null)
        {
            Debug.Log($"[ShaderGenerator] Using assigned {logName} shader");
            return shader;
        }
        shader = Shader.Find(shaderName);
        if (shader == null)
        {
            Debug.LogWarning($"⚠ ShaderGenerator: {logName} shader ({shaderName}) not found, using Standard");
            return fallbackShader;
        }
        Debug.Log($"[ShaderGenerator] Loaded {logName} shader: {shaderName}");
        return shader;
    }

    public Material GenerateMaterial(ShaderParams shaderParams, TextureParams textureParams, float psi = 0f, float lyapunov = 0f, float synapticStrength = 0f, int entityType = 0)
    {
        if (shaderParams == null || textureParams == null)
        {
            Debug.LogWarning("⛔ ShaderGenerator: Invalid shader or texture params, using fallback");
            return new Material(fallbackShader);
        }

        try
        {
            Shader selectedShader;
            switch (entityType)
            {
                case 0: // Star
                    selectedShader = starShader ?? fallbackShader;
                    break;
                case 2: // Black Hole
                    selectedShader = blackHoleShader ?? fallbackShader;
                    break;
                case 3: // Nebula
                    selectedShader = nebulaShader ?? fallbackShader;
                    break;
                default: // Planets and others
                    selectedShader = fallbackShader;
                    break;
            }

            Material material = new Material(selectedShader);

            float psiNorm = Mathf.Clamp01(psi / 1e-10f);
            float lyapunovNorm = Mathf.Clamp01(lyapunov / 0.1f);
            float omegaNorm = Mathf.Clamp01(synapticStrength / 1e5f);
            float audioMod = FindFirstObjectByType<MicAnalyzer>()?.rmsValue ?? 0f;

            // Enhanced color variation based on frequency
            float freqNorm = textureParams.freq_scale / 10.0f; // Normalized from freq_scale
            Color baseColor = new Color(
                Mathf.Clamp(shaderParams.base_color[0] + freqNorm * 0.2f, 0f, 1f),
                Mathf.Clamp(shaderParams.base_color[1] + freqNorm * 0.1f, 0f, 1f),
                Mathf.Clamp(shaderParams.base_color[2] + freqNorm * 0.15f, 0f, 1f)
            );
            baseColor *= (1f + psiNorm * 0.3f + audioMod * 0.4f);
            baseColor = Color.Lerp(baseColor, Color.HSVToRGB(freqNorm, 0.8f, 1f), lyapunovNorm * 0.25f);

            material.SetColor("_BaseColor", baseColor);
            material.SetFloat("_EmissionStrength", shaderParams.emission_power * (1f + psiNorm + audioMod) * PHI);
            material.SetFloat("_NoiseScale", shaderParams.noise_scale * (1f + lyapunovNorm + freqNorm * 0.5f));
            material.SetFloat("_PulseSpeed", shaderParams.pulse_speed * (1f + omegaNorm));
            material.EnableKeyword("_EMISSION");

            if (entityType == 2) // Black Hole
            {
                material.SetFloat("_AccretionDiskRadius", 1f + omegaNorm * 0.5f + freqNorm * 0.2f);
                material.SetFloat("_EventHorizonDistortion", lyapunovNorm * 0.3f + audioMod * 0.1f);
            }
            else if (entityType == 0) // Star
            {
                material.SetFloat("_SolarFlareIntensity", psiNorm * 0.5f + freqNorm * 0.3f);
            }
            else if (entityType == 3) // Nebula
            {
                material.SetFloat("_VolumetricDensity", textureParams.entropy_scale * 2f + audioMod * 0.5f);
            }

            Texture2D texture = GenerateProceduralTexture(textureParams, psiNorm, lyapunovNorm, freqNorm);
            if (texture != null)
            {
                material.SetTexture("_MainTex", texture);
            }

            Debug.Log($"🖌 ShaderGenerator: Generated material for entity type {entityType} with psi={psi}, freqNorm={freqNorm}");
            return material;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ ShaderGenerator: Failed to generate material: {e.Message}");
            return new Material(fallbackShader);
        }
    }

    private Texture2D GenerateProceduralTexture(TextureParams textureParams, float psiNorm, float lyapunovNorm, float freqNorm)
    {
        const int TEXTURE_SIZE = 256;
        Texture2D texture = new Texture2D(TEXTURE_SIZE, TEXTURE_SIZE);
        Color[] colors = new Color[TEXTURE_SIZE * TEXTURE_SIZE];

        bool isPlanetary = textureParams.noise_type == "planetary";
        float freqScale = textureParams.freq_scale * (1f + freqNorm);
        float entropyScale = textureParams.entropy_scale;
        float audioMod = FindFirstObjectByType<MicAnalyzer>()?.rmsValue ?? 0f;

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

                Color pixelColor;
                if (isPlanetary)
                {
                    float landThreshold = 0.5f + entropyScale * 0.1f - audioMod * 0.2f + psiNorm * 0.1f;
                    float hue;
                    if (noiseValue > landThreshold)
                    {
                        if (freqNorm < 0.25f) // Desert
                            hue = 0.1f + entropyScale * 0.05f + freqNorm * 0.1f; // Sandy yellow
                        else if (freqNorm < 0.5f) // Forest
                            hue = 0.3f + entropyScale * 0.1f + freqNorm * 0.15f; // Green
                        else if (freqNorm < 0.75f) // Ocean
                            hue = 0.5f + entropyScale * 0.08f + freqNorm * 0.12f; // Blue
                        else // Tundra
                            hue = 0.6f + entropyScale * 0.05f + freqNorm * 0.1f; // Gray-blue
                        pixelColor = Color.HSVToRGB(hue, 0.6f, noiseValue);
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
                colors[y * TEXTURE_SIZE + x] = pixelColor;
            }
        }

        texture.SetPixels(colors);
        texture.Apply();
        return texture;
    }
}