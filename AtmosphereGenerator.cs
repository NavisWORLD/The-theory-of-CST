using UnityEngine;

public class AtmosphereGenerator : MonoBehaviour
{
    private const float PHI = 1.6180339887f;

    public void AddAtmosphere(GameObject planet, EcosystemData ecoData, float frequency, float entropy)
    {
        if (ecoData.atmosphere == null)
        {
            Debug.LogWarning("⚠ AtmosphereGenerator: No atmosphere data provided");
            return;
        }

        float radius = planet.transform.localScale.x;
        float cloudDensity = ecoData.atmosphere.cloud_density;
        float hazeThickness = ecoData.atmosphere.haze_thickness;
        Color atmColor = new Color(
            ecoData.atmosphere.color[0],
            ecoData.atmosphere.color[1],
            ecoData.atmosphere.color[2]
        );

        // Create atmosphere sphere
        GameObject atmosphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        atmosphere.transform.SetParent(planet.transform);
        atmosphere.transform.localPosition = Vector3.zero;
        atmosphere.transform.localScale = Vector3.one * (radius * 1.1f);

        // Remove collider
        Destroy(atmosphere.GetComponent<SphereCollider>());

        // Create material
        Material atmMaterial = new Material(Shader.Find("Standard"));
        atmMaterial.SetColor("_Color", atmColor * (1f + entropy * 0.2f));
        atmMaterial.SetFloat("_Metallic", 0f);
        atmMaterial.SetFloat("_Glossiness", 0.9f);
        atmMaterial.SetColor("_EmissionColor", atmColor * 0.2f);
        atmMaterial.EnableKeyword("_EMISSION");
        atmosphere.GetComponent<MeshRenderer>().material = atmMaterial;

        // Add clouds if density is high
        if (cloudDensity > 0.3f)
        {
            ParticleSystem clouds = atmosphere.AddComponent<ParticleSystem>();
            var main = clouds.main;
            main.startSize = new ParticleSystem.MinMaxCurve(0.5f, 1f);
            main.startLifetime = new ParticleSystem.MinMaxCurve(5f, 10f);
            main.startColor = Color.white;

            var emission = clouds.emission;
            emission.rateOverTime = cloudDensity * 50f;

            var shape = clouds.shape;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = radius * 1.05f;

            var velocity = clouds.velocityOverLifetime;
            velocity.enabled = true;
            velocity.space = ParticleSystemSimulationSpace.World;
            velocity.x = new ParticleSystem.MinMaxCurve(-0.1f, 0.1f);
            velocity.y = new ParticleSystem.MinMaxCurve(-0.1f, 0.1f);
            velocity.z = new ParticleSystem.MinMaxCurve(-0.1f, 0.1f);
        }

        Debug.Log($"🌫 Added atmosphere to planet at {planet.transform.position} with cloudDensity={cloudDensity}, hazeThickness={hazeThickness}");
    }
}