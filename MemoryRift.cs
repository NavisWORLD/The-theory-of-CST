using UnityEngine;

public class MemoryRift : MonoBehaviour
{
    private ParticleSystem ps;
    private TrailRenderer trail;
    public float baseIntensity = 1f;
    public float entropyScale = 2f;
    public float psiScale = 1e-10f;
    public float frequencyScale = 0.01f;

    void Awake()
    {
        ps = GetComponent<ParticleSystem>();
        trail = GetComponent<TrailRenderer>();

        // Initialize TrailRenderer if missing
        if (ps == null && trail == null)
        {
            trail = gameObject.AddComponent<TrailRenderer>();
            trail.startWidth = 0.1f;
            trail.endWidth = 0.01f;
            trail.time = 1f;
            trail.material = new Material(Shader.Find("Standard"));
            trail.startColor = Color.white;
            trail.endColor = Color.clear;
            trail.enabled = false;
            Debug.Log($"[MemoryRift] Added TrailRenderer to {gameObject.name}");
        }
    }

    public void Activate(float entropy, float psi, float frequency)
    {
        try
        {
            if (ps != null)
            {
                var main = ps.main;
                main.startColor = new Color(
                    Mathf.Clamp01(entropy * entropyScale),
                    Mathf.Clamp01(psi * psiScale),
                    Mathf.Clamp01(frequency * frequencyScale)
                );
                if (!ps.isPlaying)
                {
                    ps.Play();
                }
            }
            else if (trail != null)
            {
                trail.startColor = new Color(
                    Mathf.Clamp01(entropy * entropyScale),
                    Mathf.Clamp01(psi * psiScale),
                    Mathf.Clamp01(frequency * frequencyScale),
                    baseIntensity
                );
                trail.enabled = true;
                trail.emitting = true;
            }
            else
            {
                Debug.LogWarning($"[MemoryRift] No ParticleSystem or TrailRenderer on {gameObject.name}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"⚠ MemoryRift: Failed to activate for {gameObject.name}: {e.Message}");
        }
    }

    void OnDisable()
    {
        if (ps != null && ps.isPlaying)
        {
            ps.Stop();
        }
        if (trail != null)
        {
            trail.emitting = false;
            trail.enabled = false;
        }
    }
}