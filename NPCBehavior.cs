using UnityEngine;
using System.Linq;

public class NPCBehavior : MonoBehaviour
{
    private float[] behaviorVector;
    public float moveSpeed = 1f;
    public float rotationSpeed = 90f;
    private float lyapunov = 0f;
    private float psi = 0f;

    public void SetBehavior(float[] behavior, float lyapunovExponent, float psiValue)
    {
        if (behavior != null && behavior.Length >= 12)
        {
            behaviorVector = behavior;
            lyapunov = Mathf.Clamp01(lyapunovExponent / 0.1f);
            psi = Mathf.Clamp01(psiValue / 1e-10f);
            Debug.Log($"[NPCBehavior] Set behavior for {gameObject.name}, lyapunov={lyapunov}, psi={psi}");
        }
        else
        {
            Debug.LogWarning($"[NPCBehavior] Invalid behavior vector for {gameObject.name}");
            behaviorVector = new float[12];
        }
    }

    void Update()
    {
        if (behaviorVector == null) return;

        float[] newBehavior = new float[behaviorVector.Length];
        float sigma = 10f, beta = 8f / 3f;
        for (int i = 0; i < 11; i++)
        {
            newBehavior[i] = behaviorVector[i] + Time.deltaTime * sigma * (behaviorVector[i + 1] - behaviorVector[i]);
        }
        float squaredSum = behaviorVector.Take(11).Sum(x => x * x);
        newBehavior[11] = behaviorVector[11] + Time.deltaTime * (-beta * behaviorVector[11] + squaredSum);
        behaviorVector = newBehavior;

        if (gameObject.CompareTag("Creature"))
        {
            Vector3 moveDir = new Vector3(behaviorVector[0], 0, behaviorVector[1]).normalized;
            moveSpeed *= (1f + lyapunov * 0.5f + psi * 0.2f);
            transform.position += moveDir * moveSpeed * Time.deltaTime;
            if (moveDir != Vector3.zero)
            {
                Quaternion targetRotation = Quaternion.LookRotation(moveDir);
                transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
        }
        else if (gameObject.CompareTag("Plant"))
        {
            float sway = Mathf.Sin(Time.time * behaviorVector[2] * (1f + lyapunov)) * 0.1f;
            transform.localRotation = Quaternion.Euler(0, 0, sway);
        }
    }
}