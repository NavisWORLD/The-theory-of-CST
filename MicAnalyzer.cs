using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class MicAnalyzer : MonoBehaviour
{
    public float rmsValue { get; private set; }
    public float pitchValue { get; private set; }
    [Range(1f, 50f)]
    public float sensitivity = 15.0f;
    public float updateInterval = 0.05f;
    public float smoothingFactor = 0.05f; // Reduced for more responsiveness

    private AudioSource source;
    private const int SAMPLE_SIZE = 512;
    private float[] spectrum = new float[SAMPLE_SIZE];
    private float[] samples = new float[SAMPLE_SIZE];
    private float sampleRate;
    private float timer;
    private bool isMicActive;

    void Awake()
    {
        sampleRate = AudioSettings.outputSampleRate;
        rmsValue = 0.3f;
        pitchValue = 440f;
        isMicActive = false;
    }

    void Start()
    {
        source = GetComponent<AudioSource>();
        if (Microphone.devices.Length > 0)
        {
            try
            {
                string micDevice = Microphone.devices[0];
                Debug.Log($"[MicAnalyzer] Initializing microphone: {micDevice}, SampleRate: {sampleRate}");
                source.clip = Microphone.Start(micDevice, true, 1, (int)sampleRate);
                source.loop = true;
                while (!(Microphone.GetPosition(micDevice) > 0)) { }
                source.Play();
                isMicActive = true;
                Debug.Log("[MicAnalyzer] Microphone initialized successfully");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[MicAnalyzer] Failed to initialize microphone: {e.Message}");
                isMicActive = false;
            }
        }
        else
        {
            Debug.LogError("[MicAnalyzer] No microphone found. Using fallback values (rmsValue=0.3, pitchValue=440)");
            isMicActive = false;
        }
    }

    void Update()
    {
        if (!isMicActive)
        {
            rmsValue = Mathf.Lerp(rmsValue, 0.3f, smoothingFactor);
            pitchValue = Mathf.Lerp(pitchValue, 440f, smoothingFactor);
            return;
        }

        timer += Time.deltaTime;
        if (timer < updateInterval)
            return;

        timer = 0f;

        source.GetOutputData(samples, 0);
        float sum = 0;
        float maxSample = 0f;
        for (int i = 0; i < SAMPLE_SIZE; i++)
        {
            sum += samples[i] * samples[i];
            maxSample = Mathf.Max(maxSample, Mathf.Abs(samples[i]));
        }
        float rawRms = Mathf.Sqrt(sum / SAMPLE_SIZE) * sensitivity * 2f; // Increased sensitivity
        rmsValue = Mathf.Lerp(rmsValue, rawRms, smoothingFactor);

        source.GetSpectrumData(spectrum, 0, FFTWindow.BlackmanHarris);
        float maxV = 0f;
        int maxN = 0;
        for (int i = 0; i < SAMPLE_SIZE; i++)
        {
            if (spectrum[i] > maxV && spectrum[i] > 0.0001f) // Lowered threshold
            {
                maxV = spectrum[i];
                maxN = i;
            }
        }

        float freqN = maxN;
        if (maxN > 0 && maxN < SAMPLE_SIZE - 1)
        {
            float dL = spectrum[maxN - 1] / spectrum[maxN];
            float dR = spectrum[maxN + 1] / spectrum[maxN];
            freqN += 0.5f * (dR * dR - dL * dL);
        }

        float rawPitch = freqN * sampleRate / 2 / SAMPLE_SIZE;
        rawPitch = Mathf.Clamp(rawPitch, 20f, 20000f);
        pitchValue = Mathf.Lerp(pitchValue, rawPitch, smoothingFactor);

        Debug.Log($"[MicAnalyzer] RMS={rmsValue:F3}, Pitch={pitchValue:F1}Hz, MaxSample={maxSample:F6}, MaxSpectrum={maxV:F6}");
    }

    void OnDestroy()
    {
        if (isMicActive && source != null && source.clip != null)
        {
            Microphone.End(null);
            source.Stop();
            Destroy(source.clip);
            Debug.Log("[MicAnalyzer] Microphone stopped and cleaned up");
        }
    }
}