using UnityEngine;
using System.Linq;
using System.Collections.Generic;

public class PlanetFactory : MonoBehaviour
{
    public ProceduralMaterialGenerator materialGenerator;
    public MeshGenerator meshGenerator;
    public MicAnalyzer micAnalyzer;
    public AtmosphereGenerator atmosphereGenerator;

    private const float PHI = 1.6180339887f;
    private const int TERRAIN_SIZE = 256;

    void Start()
    {
        if (materialGenerator == null)
        {
            materialGenerator = FindFirstObjectByType<ProceduralMaterialGenerator>();
            if (materialGenerator == null)
            {
                Debug.LogError("⛔ PlanetFactory: ProceduralMaterialGenerator not found!");
                enabled = false;
                return;
            }
            Debug.Log("✅ PlanetFactory: ProceduralMaterialGenerator auto-assigned");
        }

        if (meshGenerator == null)
        {
            meshGenerator = FindFirstObjectByType<MeshGenerator>();
            if (meshGenerator == null)
            {
                Debug.LogError("⛔ PlanetFactory: MeshGenerator not found!");
                enabled = false;
                return;
            }
            Debug.Log("✅ PlanetFactory: MeshGenerator auto-assigned");
        }

        if (micAnalyzer == null)
        {
            micAnalyzer = FindFirstObjectByType<MicAnalyzer>();
            if (micAnalyzer == null)
            {
                Debug.LogError("⛔ PlanetFactory: MicAnalyzer not found!");
                enabled = false;
                return;
            }
            Debug.Log("✅ PlanetFactory: MicAnalyzer auto-assigned");
        }

        if (atmosphereGenerator == null)
        {
            atmosphereGenerator = FindFirstObjectByType<AtmosphereGenerator>();
            if (atmosphereGenerator == null)
            {
                Debug.LogError("⛔ PlanetFactory: AtmosphereGenerator not found!");
                enabled = false;
                return;
            }
            Debug.Log("✅ PlanetFactory: AtmosphereGenerator auto-assigned");
        }
    }

    public GameObject CreatePlanet(CSTEntity entity)
    {
        if (materialGenerator == null || meshGenerator == null || micAnalyzer == null || atmosphereGenerator == null)
        {
            Debug.LogWarning("⚠ PlanetFactory: Missing references, cannot create planet");
            return null;
        }

        float loudness = micAnalyzer.rmsValue;
        float pitch = Mathf.Clamp(micAnalyzer.pitchValue, 20f, 20000f);
        float entropyNorm = Mathf.Clamp01(entity.entropy);
        float psiNorm = Mathf.Clamp01(entity.psi / 1e-10f);
        float freq = (float)entity.frequency;

        float baseSize = entity.mesh_params.radius;
        float size = baseSize * (1f + loudness * PHI) * (1f + entropyNorm * 0.5f);
        size = Mathf.Clamp(size, 0.5f, 10f);

        GameObject planet = new GameObject($"Planet_{entity.id}");
        planet.transform.position = entity.positionVector / 1e9f;
        planet.transform.localScale = Vector3.one * size;

        MeshFilter meshFilter = planet.AddComponent<MeshFilter>();
        MeshRenderer renderer = planet.AddComponent<MeshRenderer>();
        Mesh terrainMesh = GenerateTerrainMesh(entity.ecosystem_data, entity.mesh_params, entropyNorm, freq, psiNorm);
        meshFilter.mesh = terrainMesh;

        ShaderParams shaderParams = entity.shader_params;
        shaderParams.emission_power += loudness * 2f + psiNorm * 0.5f;
        TextureParams textureParams = entity.texture_params;
        Material material = materialGenerator.GenerateMaterial(shaderParams, textureParams, entity.psi, entity.lyapunov_exponent);
        renderer.material = material;

        AddEcosystemElements(planet, entity.ecosystem_data, size, freq, entropyNorm, psiNorm);

        atmosphereGenerator.AddAtmosphere(planet, entity.ecosystem_data, freq, entropyNorm);

        float spin = freq * 0.0001f * PHI;
        planet.AddComponent<Rotator>().speed = spin;

        Debug.Log($"🌍 Created planet {entity.id} at {planet.transform.position} with freq={freq}, entropy={entity.entropy}, size={size}");
        return planet;
    }

    private Mesh GenerateTerrainMesh(EcosystemData ecoData, MeshParams meshParams, float entropyNorm, float frequency, float psiNorm)
    {
        Mesh mesh = new Mesh();
        Vector3[] vertices = new Vector3[TERRAIN_SIZE * TERRAIN_SIZE];
        Vector2[] uv = new Vector2[TERRAIN_SIZE * TERRAIN_SIZE];
        int[] triangles = new int[(TERRAIN_SIZE - 1) * (TERRAIN_SIZE - 1) * 6];

        float radius = meshParams.radius;
        float[] heightmap = ecoData.heightmap != null ? FlattenArray(ecoData.heightmap) : new float[TERRAIN_SIZE * TERRAIN_SIZE];
        float[] biomeMap = ecoData.biome_map != null ? FlattenArray(ecoData.biome_map) : new float[TERRAIN_SIZE * TERRAIN_SIZE];

        for (int y = 0; y < TERRAIN_SIZE; y++)
        {
            for (int x = 0; x < TERRAIN_SIZE; x++)
            {
                int index = y * TERRAIN_SIZE + x;
                float u = (float)x / (TERRAIN_SIZE - 1);
                float v = (float)y / (TERRAIN_SIZE - 1);
                float theta = v * Mathf.PI;
                float phi = u * 2 * Mathf.PI;

                float height = heightmap[index] * (1f + entropyNorm * 0.5f);
                float biomeFactor = biomeMap[index];
                float biomeHeight = height * (frequency < 2000f ? 1.2f : frequency < 8000f ? 0.8f : 1.0f);

                Vector3 vertex = new Vector3(
                    Mathf.Sin(theta) * Mathf.Cos(phi),
                    Mathf.Cos(theta),
                    Mathf.Sin(theta) * Mathf.Sin(phi)
                );
                vertex *= radius * (1f + biomeHeight * 0.2f);

                vertices[index] = vertex;
                uv[index] = new Vector2(u, v);
            }
        }

        int triIndex = 0;
        for (int y = 0; y < TERRAIN_SIZE - 1; y++)
        {
            for (int x = 0; x < TERRAIN_SIZE - 1; x++)
            {
                int a = y * TERRAIN_SIZE + x;
                int b = a + 1;
                int c = (y + 1) * TERRAIN_SIZE + x;
                int d = c + 1;

                triangles[triIndex++] = a; triangles[triIndex++] = c; triangles[triIndex++] = b;
                triangles[triIndex++] = b; triangles[triIndex++] = c; triangles[triIndex++] = d;
            }
        }

        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private void AddEcosystemElements(GameObject planet, EcosystemData ecoData, float size, float freq, float entropyNorm, float psiNorm)
    {
        foreach (var lifeform in ecoData.lifeforms)
        {
            Vector3 pos = planet.transform.position + new Vector3(lifeform.position[0], lifeform.position[1], lifeform.position[2]) * size / 100f;
            GameObject lifeObj = new GameObject(lifeform.type);
            lifeObj.transform.SetParent(planet.transform);
            lifeObj.transform.position = pos;

            MeshFilter meshFilter = lifeObj.AddComponent<MeshFilter>();
            MeshRenderer renderer = lifeObj.AddComponent<MeshRenderer>();
            Mesh mesh = GenerateLifeformMesh(lifeform.mesh_params, freq, entropyNorm);
            meshFilter.mesh = mesh;

            Material material = new Material(Shader.Find("Standard"));
            if (lifeform.type.Contains("creature"))
                material.color = new Color(0.6f, 0.3f, 0.2f);
            else if (lifeform.type.Contains("coral"))
                material.color = new Color(0.8f, 0.2f, 0.4f);
            else if (lifeform.type.Contains("shrub"))
                material.color = new Color(0.4f, 0.6f, 0.2f);
            else
                material.color = new Color(0.3f, 0.7f, 0.1f);
            renderer.material = material;

            NPCBehavior behavior = lifeObj.AddComponent<NPCBehavior>();
            behavior.SetBehavior(lifeform.behavior_vector, ecoData.terrain_height, psiNorm);
        }

        if (ecoData.water_coverage > 0.2f)
        {
            GameObject water = new GameObject("Water");
            water.transform.SetParent(planet.transform);
            water.transform.position = planet.transform.position;
            MeshFilter waterMesh = water.AddComponent<MeshFilter>();
            MeshRenderer waterRenderer = water.AddComponent<MeshRenderer>();
            Mesh waterMeshData = meshGenerator.GenerateSphere(size * ecoData.water_coverage * 0.95f, 16, entropyNorm, freq, 0.1f);
            waterMesh.mesh = waterMeshData;
            Material waterMaterial = new Material(Shader.Find("Standard"));
            waterMaterial.SetColor("_Color", new Color(0.1f, 0.4f, 0.8f, 0.8f));
            waterRenderer.material = waterMaterial;
        }
    }

    private Mesh GenerateLifeformMesh(MeshParamsData meshParams, float frequency, float entropyNorm)
    {
        string type = meshParams.type;
        float height = meshParams.height;
        float complexity = meshParams.complexity;

        Mesh mesh = new Mesh();
        Vector3[] vertices;
        int[] triangles;

        if (type == "tree")
        {
            vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), new Vector3(0, height, 0),
                new Vector3(-0.3f * complexity, height, 0), new Vector3(0.3f * complexity, height, 0),
                new Vector3(0, height * 1.5f, 0)
            };
            triangles = new int[] { 0, 1, 2, 1, 3, 4 };
        }
        else if (type == "creature")
        {
            vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), new Vector3(-0.2f * complexity, height * 0.5f, 0),
                new Vector3(0.2f * complexity, height * 0.5f, 0), new Vector3(0, height, 0)
            };
            triangles = new int[] { 0, 1, 2, 1, 2, 3 };
        }
        else if (type == "shrub")
        {
            vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), new Vector3(-0.2f * complexity, height * 0.8f, 0),
                new Vector3(0.2f * complexity, height * 0.8f, 0), new Vector3(0, height, 0),
                new Vector3(-0.1f * complexity, height * 0.9f, 0.1f * complexity)
            };
            triangles = new int[] { 0, 1, 2, 1, 2, 3, 0, 3, 4 };
        }
        else if (type == "coral")
        {
            vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), new Vector3(-0.15f * complexity, height * 0.7f, 0),
                new Vector3(0.15f * complexity, height * 0.7f, 0), new Vector3(0, height, 0),
                new Vector3(0, height * 0.8f, 0.1f * complexity)
            };
            triangles = new int[] { 0, 1, 2, 1, 2, 3, 0, 3, 4 };
        }
        else // grass
        {
            vertices = new Vector3[]
            {
                new Vector3(0, 0, 0), new Vector3(0, height, 0),
                new Vector3(-0.1f * complexity, height, 0), new Vector3(0.1f * complexity, height, 0)
            };
            triangles = new int[] { 0, 1, 2, 1, 2, 3 };
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        return mesh;
    }

    private float[] FlattenArray(float[][] jaggedArray)
    {
        if (jaggedArray == null) return new float[TERRAIN_SIZE * TERRAIN_SIZE];
        int totalLength = 0;
        foreach (var row in jaggedArray)
        {
            if (row != null) totalLength += row.Length;
        }
        float[] result = new float[totalLength];
        int index = 0;
        foreach (var row in jaggedArray)
        {
            if (row != null)
            {
                for (int i = 0; i < row.Length; i++)
                {
                    result[index++] = row[i];
                }
            }
        }
        return result;
    }
}

public class Rotator : MonoBehaviour
{
    public float speed = 1f;
    void Update() => transform.Rotate(Vector3.up * speed * Time.deltaTime);
}