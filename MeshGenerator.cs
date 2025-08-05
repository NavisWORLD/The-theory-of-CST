using UnityEngine;
using System.Collections.Generic;
#if HAS_FASTNOISELITE
using FastNoiseLite;
#endif

public class MeshGenerator : MonoBehaviour
{
    private const float PHI = 1.6180339887f;
    private const int TERRAIN_SIZE = 256;

    public Mesh GenerateMesh(MeshParams meshParams, float entropy = 0.5f, float frequency = 440f, float psi = 0f, float lyapunov = 0f)
    {
        if (meshParams == null)
        {
            Debug.LogWarning("⛔ MeshGenerator: MeshParams is null, using default sphere");
            meshParams = new MeshParams { type = "sphere", radius = 1f, segments = 16 };
        }

        float psiNorm = Mathf.Clamp01(psi / 1e-10f);
        float lyapunovNorm = Mathf.Clamp01(lyapunov / 0.1f);

        try
        {
            switch (meshParams.type.ToLower())
            {
                case "sphere":
                    return GenerateSphere(meshParams.radius * (1f + psiNorm * 0.1f), meshParams.segments, entropy, frequency, lyapunovNorm);
                case "cube":
                    return GenerateCube(meshParams.radius * (1f + psiNorm * 0.1f));
                case "cluster":
                    return GenerateCluster(meshParams.radius * (1f + psiNorm * 0.2f), meshParams.segments, entropy);
                case "blackhole":
                    return GenerateDistortedSphere(meshParams.radius * (1f + psiNorm * 0.15f), meshParams.segments, lyapunovNorm);
                default:
                    Debug.LogWarning($"⛔ MeshGenerator: Unsupported mesh type '{meshParams.type}', using default sphere");
                    return GenerateSphere(meshParams.radius, meshParams.segments, entropy, frequency, lyapunovNorm);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ MeshGenerator: Failed to generate mesh for type '{meshParams.type}': {e.Message}");
            return GenerateSphere(1f, 16, entropy, frequency, lyapunovNorm);
        }
    }

    public Mesh GenerateSphere(float radius, int segments, float entropy, float frequency, float lyapunovNorm)
    {
        Mesh mesh = new Mesh();
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uv = new List<Vector2>();

        float deltaTheta = Mathf.PI / segments;
        float deltaPhi = 2 * Mathf.PI / segments;

#if HAS_FASTNOISELITE
        FastNoiseLite noise = new FastNoiseLite();
        noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise.SetFractalType(FastNoiseLite.FractalType.FBm);
        noise.SetFractalOctaves(6);
        noise.SetFrequency(0.01f * (1f + entropy + lyapunovNorm));
#endif

        for (int i = 0; i <= segments; i++)
        {
            float theta = i * deltaTheta;
            float sinTheta = Mathf.Sin(theta);
            float cosTheta = Mathf.Cos(theta);

            for (int j = 0; j <= segments; j++)
            {
                float phi = j * deltaPhi;
                float sinPhi = Mathf.Sin(phi);
                float cosPhi = Mathf.Cos(phi);

                Vector3 vertex = new Vector3(
                    sinTheta * cosPhi,
                    cosTheta,
                    sinTheta * sinPhi
                );

                float height;
#if HAS_FASTNOISELITE
                height = noise.GetNoise(vertex.x * 100f, vertex.y * 100f, vertex.z * 100f);
                height = Mathf.Clamp01(height * 0.5f + 0.5f) * (0.2f + entropy * 0.4f);
#else
                float baseHeight = Mathf.PerlinNoise(vertex.x * 10f + entropy, vertex.z * 10f + lyapunovNorm);
                float ridgedHeight = 1f - Mathf.Abs(Mathf.PerlinNoise(vertex.x * 20f, vertex.z * 20f));
                height = baseHeight * 0.6f + ridgedHeight * 0.4f * (0.2f + entropy * 0.4f);
#endif
                float biomeFactor = frequency < 500f ? 1.5f : frequency < 2000f ? 1.2f : frequency < 8000f ? 0.8f : 0.6f;
                vertex *= radius * (1f + height * biomeFactor * (1f + lyapunovNorm * 0.3f));

                vertices.Add(vertex);
                uv.Add(new Vector2((float)j / segments, (float)i / segments));
            }
        }

        for (int i = 0; i < segments; i++)
        {
            for (int j = 0; j < segments; j++)
            {
                int a = i * (segments + 1) + j;
                int b = a + 1;
                int c = (i + 1) * (segments + 1) + j;
                int d = c + 1;

                triangles.Add(a); triangles.Add(c); triangles.Add(b);
                triangles.Add(b); triangles.Add(c); triangles.Add(d);
            }
        }

        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uv.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private Mesh GenerateCube(float size)
    {
        Mesh mesh = new Mesh();
        float halfSize = size * 0.5f;

        Vector3[] vertices = new Vector3[]
        {
            new Vector3(-halfSize, -halfSize, -halfSize),
            new Vector3(halfSize, -halfSize, -halfSize),
            new Vector3(halfSize, halfSize, -halfSize),
            new Vector3(-halfSize, halfSize, -halfSize),
            new Vector3(-halfSize, -halfSize, halfSize),
            new Vector3(halfSize, -halfSize, halfSize),
            new Vector3(halfSize, halfSize, halfSize),
            new Vector3(-halfSize, halfSize, halfSize),
            new Vector3(-halfSize, halfSize, -halfSize),
            new Vector3(halfSize, halfSize, -halfSize),
            new Vector3(halfSize, halfSize, halfSize),
            new Vector3(-halfSize, halfSize, halfSize),
            new Vector3(-halfSize, -halfSize, -halfSize),
            new Vector3(halfSize, -halfSize, -halfSize),
            new Vector3(halfSize, -halfSize, halfSize),
            new Vector3(-halfSize, -halfSize, halfSize),
            new Vector3(-halfSize, -halfSize, -halfSize),
            new Vector3(-halfSize, halfSize, -halfSize),
            new Vector3(-halfSize, halfSize, halfSize),
            new Vector3(-halfSize, -halfSize, halfSize),
            new Vector3(halfSize, -halfSize, -halfSize),
            new Vector3(halfSize, halfSize, -halfSize),
            new Vector3(halfSize, halfSize, halfSize),
            new Vector3(halfSize, -halfSize, halfSize)
        };

        int[] triangles = new int[]
        {
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
            12, 14, 13, 12, 15, 14,
            16, 17, 18, 16, 18, 19,
            20, 22, 21, 20, 23, 22
        };

        Vector2[] uv = new Vector2[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            uv[i] = new Vector2(vertices[i].x / size + 0.5f, vertices[i].y / size + 0.5f);
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uv;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private Mesh GenerateCluster(float radius, int segments, float entropy)
    {
        Mesh mesh = new Mesh();
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uv = new List<Vector2>();

        int numStars = Mathf.Max(5, segments);
        float clusterRadius = radius * 2f * (1f + entropy * 0.1f);

#if HAS_FASTNOISELITE
        FastNoiseLite noise = new FastNoiseLite();
        noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise.SetFrequency(0.05f);
#endif

        for (int star = 0; star < numStars; star++)
        {
            Vector3 starCenter = Random.insideUnitSphere * clusterRadius;
            float starRadius = radius * Random.Range(0.2f, 0.5f) * PHI;
            int starSegments = Mathf.Max(4, segments / 2);

            float deltaTheta = Mathf.PI / starSegments;
            float deltaPhi = 2 * Mathf.PI / starSegments;
            int baseVertexIndex = vertices.Count;

            for (int i = 0; i <= starSegments; i++)
            {
                float theta = i * deltaTheta;
                float sinTheta = Mathf.Sin(theta);
                float cosTheta = Mathf.Cos(theta);

                for (int j = 0; j <= starSegments; j++)
                {
                    float phi = j * deltaPhi;
                    float sinPhi = Mathf.Sin(phi);
                    float cosPhi = Mathf.Cos(phi);

                    Vector3 vertex = new Vector3(
                        starRadius * sinTheta * cosPhi,
                        starRadius * cosTheta,
                        starRadius * sinTheta * sinPhi
                    );
#if HAS_FASTNOISELITE
                    float offset = noise.GetNoise(vertex.x * 100f, vertex.y * 100f, vertex.z * 100f) * entropy * 0.1f;
#else
                    float offset = Mathf.PerlinNoise(vertex.x * 10f + entropy, vertex.z * 10f) * entropy * 0.1f;
#endif
                    vertex += starCenter + vertex.normalized * offset;

                    vertices.Add(vertex);
                    uv.Add(new Vector2((float)j / starSegments, (float)i / starSegments));
                }
            }

            for (int i = 0; i < starSegments; i++)
            {
                for (int j = 0; j < starSegments; j++)
                {
                    int a = baseVertexIndex + i * (starSegments + 1) + j;
                    int b = a + 1;
                    int c = baseVertexIndex + (i + 1) * (starSegments + 1) + j;
                    int d = c + 1;

                    triangles.Add(a); triangles.Add(c); triangles.Add(b);
                    triangles.Add(b); triangles.Add(c); triangles.Add(d);
                }
            }
        }

        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uv.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private Mesh GenerateDistortedSphere(float radius, int segments, float lyapunovNorm)
    {
        Mesh mesh = new Mesh();
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uv = new List<Vector2>();

        float deltaTheta = Mathf.PI / segments;
        float deltaPhi = 2 * Mathf.PI / segments;

#if HAS_FASTNOISELITE
        FastNoiseLite noise = new FastNoiseLite();
        noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise.SetFrequency(0.02f * (1f + lyapunovNorm));
#endif

        for (int i = 0; i <= segments; i++)
        {
            float theta = i * deltaTheta;
            float sinTheta = Mathf.Sin(theta);
            float cosTheta = Mathf.Cos(theta);

            for (int j = 0; j <= segments; j++)
            {
                float phi = j * deltaPhi;
                float sinPhi = Mathf.Sin(phi);
                float cosPhi = Mathf.Cos(phi);

                float distortion = 1f - Mathf.Abs(cosTheta) * 0.5f * (1f + lyapunovNorm);
                float distortedRadius = radius * distortion;

                Vector3 vertex = new Vector3(
                    distortedRadius * sinTheta * cosPhi,
                    distortedRadius * cosTheta,
                    distortedRadius * sinTheta * sinPhi
                );
#if HAS_FASTNOISELITE
                float noiseOffset = noise.GetNoise(vertex.x * 100f, vertex.y * 100f, vertex.z * 100f) * 0.1f * lyapunovNorm;
#else
                float noiseOffset = Mathf.PerlinNoise(vertex.x * 10f + lyapunovNorm, vertex.z * 10f) * 0.1f * lyapunovNorm;
#endif
                vertex *= (1f + noiseOffset);

                vertices.Add(vertex);
                uv.Add(new Vector2((float)j / segments, (float)i / segments));
            }
        }

        for (int i = 0; i < segments; i++)
        {
            for (int j = 0; j < segments; j++)
            {
                int a = i * (segments + 1) + j;
                int b = a + 1;
                int c = (i + 1) * (segments + 1) + j;
                int d = c + 1;

                triangles.Add(a); triangles.Add(c); triangles.Add(b);
                triangles.Add(b); triangles.Add(c); triangles.Add(d);
            }
        }

        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uv.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }
}