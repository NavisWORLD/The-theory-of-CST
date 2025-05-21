using System.Collections.Generic;
using UnityEngine;

public class CosmicEngine : MonoBehaviour
{
    public CSTClient cstClient;
    public MicAnalyzer micAnalyzer;
    public MeshGenerator meshGenerator;
    public ShaderGenerator shaderGenerator;
    public ProceduralMaterialGenerator materialGenerator;
    public PlanetFactory planetFactory;
    public int poolSize = 10000;
    public float wobbleAmplitude = 1f;
    public int maxEntitiesPerFrame = 200;
    public float positionScale = 1e9f;
    public float gravityScale = 1e5f;
    private const float PHI = 1.6180339887f;

    private Dictionary<int, GameObject> activeObjects = new Dictionary<int, GameObject>();
    private Queue<GameObject> pool = new Queue<GameObject>();
    private Material fallbackMaterial;
    private Dictionary<string, GameObject> prefabCache = new Dictionary<string, GameObject>();
    private Material blackHoleMaterial;

    void Awake()
    {
        if (FindObjectsByType<AudioListener>(FindObjectsSortMode.None).Length == 0)
        {
            gameObject.AddComponent<AudioListener>();
            Debug.Log("✅ Added AudioListener to CosmicEngine");
        }

        fallbackMaterial = new Material(Shader.Find("Standard"));
        if (fallbackMaterial != null)
        {
            fallbackMaterial.color = Color.gray;
        }
    }

    void Start()
    {
        if (cstClient == null) cstClient = FindFirstObjectByType<CSTClient>();
        if (micAnalyzer == null) micAnalyzer = FindFirstObjectByType<MicAnalyzer>();
        if (meshGenerator == null) meshGenerator = FindFirstObjectByType<MeshGenerator>();
        if (shaderGenerator == null) shaderGenerator = FindFirstObjectByType<ShaderGenerator>();
        if (materialGenerator == null) materialGenerator = FindFirstObjectByType<ProceduralMaterialGenerator>();
        if (planetFactory == null) planetFactory = FindFirstObjectByType<PlanetFactory>();

        if (cstClient == null || micAnalyzer == null || meshGenerator == null || shaderGenerator == null || materialGenerator == null || planetFactory == null)
        {
            Debug.LogError("⛔ CosmicEngine: Missing required components!");
            enabled = false;
            return;
        }

        InitializePool();
        InitializeProceduralAssets();
    }

    void InitializePool()
    {
        for (int i = 0; i < poolSize; i++)
        {
            GameObject obj = new GameObject($"Entity_{i}");
            obj.transform.SetParent(transform);
            obj.AddComponent<MeshFilter>();
            obj.AddComponent<MeshRenderer>();
            obj.AddComponent<CSTEntityData>();
            var rb = obj.AddComponent<Rigidbody>();
            rb.useGravity = false;
            rb.mass = 1f;
            rb.constraints = RigidbodyConstraints.None;
            var trail = obj.AddComponent<TrailRenderer>();
            trail.startWidth = 0.1f;
            trail.endWidth = 0.01f;
            trail.time = 1f;
            trail.material = new Material(Shader.Find("Standard"));
            trail.startColor = Color.white;
            trail.endColor = Color.clear;
            trail.enabled = false;
            obj.AddComponent<MemoryRift>();
            obj.SetActive(false);
            pool.Enqueue(obj);
        }
        Debug.Log($"✅ Initialized {poolSize} entity objects in pool");
    }

    void InitializeProceduralAssets()
    {
        Shader blackHoleShader = Shader.Find("Standard");
        blackHoleMaterial = new Material(blackHoleShader);
        blackHoleMaterial.SetColor("_EmissionColor", Color.black);
        blackHoleMaterial.EnableKeyword("_EMISSION");
        blackHoleMaterial.SetFloat("_Metallic", 0.8f);
        blackHoleMaterial.SetFloat("_Glossiness", 0.9f);

        GameObject creaturePrefab = new GameObject("CreaturePrefab");
        MeshFilter creatureMesh = creaturePrefab.AddComponent<MeshFilter>();
        MeshRenderer creatureRenderer = creaturePrefab.AddComponent<MeshRenderer>();
        Mesh creatureMeshData = GenerateCreatureMesh();
        creatureMesh.mesh = creatureMeshData;
        Material creatureMaterial = new Material(Shader.Find("Standard"));
        creatureMaterial.SetColor("_Color", new Color(0.6f, 0.3f, 0.2f));
        creatureRenderer.material = creatureMaterial;
        creaturePrefab.AddComponent<NPCBehavior>().moveSpeed = 1f;
        creaturePrefab.tag = "Creature";
        prefabCache["Creature"] = creaturePrefab;
        creaturePrefab.SetActive(false);

        Debug.Log("✅ Initialized procedural assets: BlackHoleMaterial, CreaturePrefab");
    }

    Mesh GenerateCreatureMesh()
    {
        Mesh mesh = new Mesh();
        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0, 0, 0), new Vector3(-0.3f, 0.5f, 0), new Vector3(0.3f, 0.5f, 0),
            new Vector3(0, 1, 0)
        };
        int[] triangles = new int[] { 0, 1, 2, 1, 2, 3 };
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        return mesh;
    }

    void Update()
    {
        var entities = cstClient?.GetEntities();
        if (entities == null || entities.Count == 0)
        {
            Debug.LogWarning("⚠ CosmicEngine: No entities received from CSTClient");
            return;
        }

        float loudness = micAnalyzer != null && float.IsFinite(micAnalyzer.rmsValue) ? micAnalyzer.rmsValue : 0f;
        HashSet<int> usedIds = new HashSet<int>();
        int processedCount = 0;

        foreach (var entity in entities)
        {
            if (processedCount >= maxEntitiesPerFrame)
            {
                Debug.LogWarning($"⚠ CosmicEngine: Reached max entities per frame ({maxEntitiesPerFrame})");
                break;
            }
            processedCount++;

            if (entity == null)
            {
                Debug.LogWarning($"⛔ Null entity detected, skipping.");
                continue;
            }

            Vector3 scaledPosition = entity.positionVector / positionScale;
            float distance = Vector3.Distance(scaledPosition, Camera.main.transform.position);
            if (distance > 1e7f || !IsInFrustum(scaledPosition))
            {
                continue;
            }

            if (!IsValidVector3(scaledPosition) || scaledPosition == Vector3.zero)
            {
                scaledPosition = new Vector3((entity.id % 10) * 10f - 50f, 0f, (entity.id / 10) * 10f - 50f);
            }

            if (entity.mesh_params == null || entity.shader_params == null || entity.texture_params == null)
            {
                Debug.LogWarning($"⛔ Missing parameters for entity {entity.id}, skipping.");
                continue;
            }

            float safePsi = float.IsFinite(entity.psi) ? entity.psi : 0f;
            float psiWobble = Mathf.Sin(Time.time * 0.5f + entity.id) * (safePsi / 1e-10f) * (1f + loudness * PHI) * wobbleAmplitude;
            if (!float.IsFinite(psiWobble))
            {
                psiWobble = 0f;
            }

            usedIds.Add(entity.id);
            Vector3 pos = scaledPosition + new Vector3(0, psiWobble, 0);
            if (!IsValidVector3(pos))
            {
                Debug.LogWarning($"⛔ Invalid adjusted position for entity {entity.id}, pos={pos}, skipping.");
                continue;
            }

            GameObject go;
            if (!activeObjects.TryGetValue(entity.id, out go))
            {
                if (entity.entity_type == 1) // Planet
                {
                    go = planetFactory.CreatePlanet(entity);
                    if (go == null)
                    {
                        Debug.LogWarning($"⚠ PlanetFactory failed for entity {entity.id}, skipping.");
                        continue;
                    }
                    go.AddComponent<CSTEntityData>().cstEntity = entity;
                    activeObjects[entity.id] = go;
                    Debug.Log($"[+] Generated planet entity {entity.id} at pos={pos}");
                    continue;
                }

                if (pool.Count == 0)
                {
                    Debug.LogWarning($"⚠ Pool exhausted for entity {entity.id}, skipping.");
                    continue;
                }

                go = pool.Dequeue();
                go.SetActive(true);
                go.name = $"Entity_{entity.id}";

                MeshFilter meshFilter = go.GetComponent<MeshFilter>();
                MeshRenderer renderer = go.GetComponent<MeshRenderer>();
                CSTEntityData entityData = go.GetComponent<CSTEntityData>();
                Rigidbody rb = go.GetComponent<Rigidbody>();
                MemoryRift rift = go.GetComponent<MemoryRift>();

                try
                {
                    float omegaNorm = Mathf.Clamp01(entity.synaptic_strength / 1e5f);
                    if (entity.entity_type == 2)
                    {
                        entity.mesh_params.radius *= 0.5f * (1f + omegaNorm);
                    }
                    else if (entity.entity_type == 4)
                    {
                        entity.mesh_params.radius *= 2f * (1f + omegaNorm);
                    }

                    entity.mesh_params.segments = distance > 1e6f ? 8 : distance > 1000f ? 16 : 32;
                    Mesh mesh = meshGenerator.GenerateMesh(entity.mesh_params, entity.entropy, (float)entity.frequency, entity.psi, entity.lyapunov_exponent);
                    if (mesh == null)
                    {
                        Debug.LogWarning($"⚠ Mesh generation failed for entity {entity.id}, using fallback sphere");
                        GameObject tempSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                        mesh = tempSphere.GetComponent<MeshFilter>().sharedMesh;
                        Destroy(tempSphere);
                    }
                    meshFilter.sharedMesh = mesh;

                    if (entity.entity_type == 0)
                    {
                        entity.shader_params.emission_power *= 2f * (1f + entity.lyapunov_exponent);
                        LensFlare flare = go.GetComponent<LensFlare>();
                        if (flare == null)
                        {
                            flare = go.AddComponent<LensFlare>();
                        }
                        flare.brightness = entity.entropy * 2f;
                    }
                    else if (entity.entity_type == 2)
                    {
                        renderer.material = blackHoleMaterial;
                        entity.shader_params.noise_scale *= 1.5f * (1f + omegaNorm);
                    }
                    else if (entity.entity_type == 3)
                    {
                        if (meshFilter != null)
                        {
                            Destroy(meshFilter);
                        }
                        var ps = go.GetComponent<ParticleSystem>();
                        if (ps == null)
                        {
                            ps = go.AddComponent<ParticleSystem>();
                            var main = ps.main;
                            main.startSize = new ParticleSystem.MinMaxCurve(0.1f, 0.5f);
                            main.startLifetime = new ParticleSystem.MinMaxCurve(1f, 2f);
                            var shape = ps.shape;
                            shape.shapeType = ParticleSystemShapeType.Sphere;
                            shape.radius = 1f;
                        }
                        try
                        {
                            var main = ps.main;
                            main.startColor = new Color(
                                entity.shader_params.base_color[0],
                                entity.shader_params.base_color[1],
                                entity.shader_params.base_color[2],
                                0.8f
                            );
                            var emission = ps.emission;
                            emission.rateOverTime = entity.entropy * 100f * (1f + loudness);
                        }
                        catch (System.Exception e)
                        {
                            Debug.LogWarning($"⚠ Failed to configure ParticleSystem for nebula entity {entity.id}: {e.Message}");
                        }
                    }

                    entity.shader_params.emission_power += loudness * 2f + safePsi * 1e-10f;
                    entity.shader_params.noise_scale += entity.entropy * 2f + entity.lyapunov_exponent;
                    entity.texture_params.freq_scale += micAnalyzer != null && float.IsFinite(micAnalyzer.pitchValue) ? micAnalyzer.pitchValue / 20000f * 5f : 0f;
                    float[] biomeMap = entity.ecosystem_data != null && entity.ecosystem_data.biome_map != null ? FlattenArray(entity.ecosystem_data.biome_map) : null;
                    Material material = materialGenerator.GenerateMaterial(entity.shader_params, entity.texture_params, entity.psi, entity.lyapunov_exponent, biomeMap);
                    if (material == null)
                    {
                        Debug.LogWarning($"⚠ Material generation failed for entity {entity.id}, using fallback");
                        material = fallbackMaterial;
                    }
                    renderer.material = material;

                    entityData.cstEntity = entity;
                    rb.mass = entity.mass / 1e30f;

                    if (entity.entity_type == 1 && loudness > 0.5f && micAnalyzer != null && micAnalyzer.pitchValue > 500f && micAnalyzer.pitchValue < 2000f)
                    {
                        ParticleSystem rain = go.GetComponent<ParticleSystem>() ?? go.AddComponent<ParticleSystem>();
                        ParticleSystem.EmissionModule emission = rain.emission;
                        emission.rateOverTime = loudness * 100f * (1f + omegaNorm);
                        var main = rain.main;
                        main.startColor = Color.cyan;
                    }

                    if (rift != null)
                    {
                        try
                        {
                            rift.Activate(entity.entropy, safePsi, (float)entity.frequency);
                        }
                        catch (System.Exception e)
                        {
                            Debug.LogWarning($"⚠ Failed to activate MemoryRift for entity {entity.id}: {e.Message}");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"⚠ MemoryRift component missing for entity {entity.id}");
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"❌ Failed to generate entity {entity.id}: {e.Message}\nStackTrace: {e.StackTrace}");
                    go.SetActive(false);
                    pool.Enqueue(go);
                    continue;
                }

                activeObjects[entity.id] = go;
                Debug.Log($"[+] Generated entity {entity.id} (Type={entity.entity_type}) at pos={pos}");
            }

            go.transform.position = pos;

            float entropyNorm = Mathf.Clamp01(entity.entropy);
            float freqNorm = ((float)entity.frequency - 20f) / (20000f - 20f);
            float size = 0.5f + entropyNorm * 2.0f + loudness * 0.5f + entity.ecosystem_level * 1.0f + freqNorm * 1.0f;
            if (!float.IsFinite(size)) size = 0.5f;
            float apparentSize = Mathf.Lerp(0.1f, 1000f, Mathf.InverseLerp(1e9f, 100f, distance));
            go.transform.localScale = Vector3.one * size * apparentSize * (1f + safePsi * 1e-10f);

            float pitch = micAnalyzer != null && float.IsFinite(micAnalyzer.pitchValue) ? Mathf.Clamp(micAnalyzer.pitchValue, 20f, 20000f) : 440f;
            Vector3 rot = Vector3.up * Time.deltaTime * (safePsi * 1e10f + pitch * 0.001f + freqNorm * 0.5f) * PHI;
            if (IsValidVector3(rot))
                go.transform.Rotate(rot);

            ApplyGravitationalForces(go, entity);
        }

        List<int> toRemove = new List<int>();
        foreach (var kvp in activeObjects)
        {
            if (!usedIds.Contains(kvp.Key))
            {
                kvp.Value.SetActive(false);
                pool.Enqueue(kvp.Value);
                toRemove.Add(kvp.Key);
                Debug.Log($"[-] Deactivated entity {kvp.Key}");
            }
        }
        foreach (var id in toRemove)
        {
            activeObjects.Remove(id);
        }
    }

    void ApplyGravitationalForces(GameObject go, CSTEntity entity)
    {
        Rigidbody rb = go.GetComponent<Rigidbody>();
        if (rb == null) return;

        foreach (var otherPair in activeObjects)
        {
            if (otherPair.Key == entity.id) continue;

            GameObject otherGo = otherPair.Value;
            CSTEntityData otherData = otherGo.GetComponent<CSTEntityData>();
            if (otherData == null || otherData.cstEntity == null) continue;

            CSTEntity otherEntity = otherData.cstEntity;
            Vector3 delta = otherGo.transform.position - go.transform.position;
            float distance = delta.magnitude;
            if (distance < 1e-5f) continue;

            float forceMagnitude = (6.67430e-11f * (entity.mass / 1e30f) * (otherEntity.mass / 1e30f)) / (distance * distance) * gravityScale;
            Vector3 force = delta.normalized * forceMagnitude;

            rb.AddForce(force, ForceMode.Force);
        }
    }

    bool IsValidVector3(Vector3 v)
    {
        return !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                 float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));
    }

    bool IsInFrustum(Vector3 pos)
    {
        Vector3 viewportPoint = Camera.main.WorldToViewportPoint(pos);
        return viewportPoint.x >= 0 && viewportPoint.x <= 1 &&
               viewportPoint.y >= 0 && viewportPoint.y <= 1 &&
               viewportPoint.z > 0;
    }

    Vector3 RandomPointOnSphere(float radius)
    {
        float theta = UnityEngine.Random.Range(0f, Mathf.PI);
        float phi = UnityEngine.Random.Range(0f, 2 * Mathf.PI);
        return new Vector3(
            radius * Mathf.Sin(theta) * Mathf.Cos(phi),
            radius * Mathf.Cos(theta),
            radius * Mathf.Sin(theta) * Mathf.Sin(phi)
        );
    }

    float[] FlattenArray(float[][] jaggedArray)
    {
        if (jaggedArray == null) return null;
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

    void OnDestroy()
    {
        if (fallbackMaterial != null)
        {
            Destroy(fallbackMaterial);
        }
        foreach (var prefab in prefabCache.Values)
        {
            Destroy(prefab);
        }
    }
}