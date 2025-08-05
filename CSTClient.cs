using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections;
using System.Linq;
using System.IO;

[Serializable]
public class AtmosphereData
{
    public float cloud_density;
    public float haze_thickness;
    public float[] color; // Array of 3 floats (R, G, B)
}

[Serializable]
public class MeshParamsData
{
    public string type;
    public float height;
    public float complexity;
}

[Serializable]
public class EcosystemData
{
    public float vegetation_density;
    public float water_coverage;
    public float terrain_height; // Added
    public List<LifeformData> lifeforms;
    public float[][] heightmap; // 2D array of floats
    public float[][] biome_map; // 2D array of floats
    public AtmosphereData atmosphere; // Added
}

[Serializable]
public class LifeformData
{
    public string type;
    public float[] position;
    public float[] behavior_vector;
    public float frequency;
    public MeshParamsData mesh_params; // Added
}

[Serializable]
public class CSTEntity
{
    public int id;
    public float mass;
    public float[] position;
    [NonSerialized] public Vector3 positionVector;
    public Vector3 velocity;
    public float psi;
    public float entropy;
    public double frequency;
    public int entity_type;
    public float ecosystem_level;
    public MeshParams mesh_params;
    public ShaderParams shader_params;
    public TextureParams texture_params;
    public float lyapunov_exponent;
    public float path_length;
    public float synaptic_strength;
    public float gravitational_potential;
    public EcosystemData ecosystem_data;

    public void OnDeserialized()
    {
        if (position != null && position.Length == 3)
        {
            Debug.Log($"[CSTEntity {id}] Deserialized position: [{position[0]}, {position[1]}, {position[2]}]");
            positionVector = new Vector3(position[0], position[1], position[2]);
        }
        else
        {
            Debug.LogWarning($"⚠ CSTEntity {id}: Invalid position data, defaulting to Vector3.zero");
            positionVector = Vector3.zero;
        }
    }
}

[Serializable]
public class MeshParams
{
    public string type;
    public float radius;
    public int segments;
    public string biome;
    public float terrain_roughness;
}

[Serializable]
public class ShaderParams
{
    public float[] base_color;
    public float emission_power;
    public float noise_scale;
    public float pulse_speed;
}

[Serializable]
public class TextureParams
{
    public string noise_type;
    public float freq_scale;
    public float entropy_scale;
}

[Serializable]
public class AudioData
{
    public float rms;
    public float pitch;
}

public class CSTClient : MonoBehaviour
{
    public string serverIP = "127.0.0.1";
    public int port = 5555;
    public float updateInterval = 0.05f;
    public int maxBufferSize = 40 * 1024 * 1024;
    public int maxReconnectAttempts = 20;
    public float heartbeatInterval = 2f;
    public float readTimeoutSeconds = 30f;
    public int fallbackPort = 5556;
    public MicAnalyzer micAnalyzer;

    private TcpClient client;
    private NetworkStream stream;
    private Thread listenerThread;
    private volatile bool connected = false;
    private bool isQuitting = false;
    private int reconnectAttempts = 0;
    private volatile bool waitingForResponse = false;
    private float timer;
    private float heartbeatTimer;
    private int currentPort;
    private bool connectionConfirmed = false;
    private float connectionStartTime;

    private List<CSTEntity> entities = new List<CSTEntity>();
    private readonly object lockObj = new object();
    private readonly Queue<List<CSTEntity>> pendingEntities = new Queue<List<CSTEntity>>();
    private bool shouldReconnect = false;
    private volatile bool entitiesReady = false;

    private StringBuilder lineBuilder = new StringBuilder();

    void Start()
    {
        Debug.Log("[CSTClient] Starting client...");
        currentPort = port;
        if (micAnalyzer == null) micAnalyzer = FindFirstObjectByType<MicAnalyzer>();
        if (micAnalyzer == null)
        {
            Debug.LogError("⛔ CSTClient: Missing MicAnalyzer component!");
        }
        StartCoroutine(MainThreadDispatcher());
        ConnectToServer();
    }

    void Update()
    {
        timer += Time.deltaTime;
        heartbeatTimer += Time.deltaTime;

        if (connected && stream != null && connectionConfirmed && !waitingForResponse && timer >= updateInterval)
        {
            timer = 0f;
            RequestUpdate();
            SendAudioData();
        }

        if (connected && stream != null && heartbeatTimer >= heartbeatInterval)
        {
            heartbeatTimer = 0f;
            SendHeartbeat();
        }

        if (connected && !connectionConfirmed && Time.time - connectionStartTime > 10f)
        {
            Debug.LogWarning("[CSTClient] Connection handshake timed out, reconnecting...");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
    }

    private IEnumerator MainThreadDispatcher()
    {
        while (!isQuitting)
        {
            lock (lockObj)
            {
                if (shouldReconnect)
                {
                    StartCoroutine(TryReconnectCoroutine());
                    shouldReconnect = false;
                }

                if (pendingEntities.Count > 0)
                {
                    entities = pendingEntities.Dequeue();
                    foreach (var entity in entities)
                    {
                        if (entity != null)
                        {
                            entity.OnDeserialized();
                        }
                    }
                    entitiesReady = true;
                    Debug.Log($"📥 Processed {entities.Count} entities in main thread, entitiesReady={entitiesReady}");
                }
            }
            yield return null;
        }
    }

    void ConnectToServer()
    {
        if (reconnectAttempts >= maxReconnectAttempts)
        {
            Debug.LogError($"❌ Max reconnection attempts ({maxReconnectAttempts}) reached for port {currentPort}. Stopping reconnection.");
            return;
        }

        Debug.Log($"[CSTClient] Attempting connection to {serverIP}:{currentPort}, Attempt {reconnectAttempts + 1}/{maxReconnectAttempts}");
        try
        {
            client = new TcpClient();
            client.ReceiveTimeout = (int)(readTimeoutSeconds * 1000);
            client.Connect(serverIP, currentPort);
            stream = client.GetStream();
            if (!client.Connected)
            {
                throw new SocketException((int)SocketError.NotConnected);
            }
            connected = true;
            reconnectAttempts = 0;
            connectionConfirmed = false;
            connectionStartTime = Time.time;

            byte[] msg = Encoding.UTF8.GetBytes("ping\n");
            stream.Write(msg, 0, msg.Length);
            stream.Flush();
            Debug.Log("[CSTClient] Sent initial ping for handshake");

            listenerThread = new Thread(ReceiveLoop);
            listenerThread.IsBackground = true;
            listenerThread.Start();

            Debug.Log($"✅ Connected to CST Python backend at {serverIP}:{currentPort}, awaiting handshake");
        }
        catch (SocketException sockEx)
        {
            Debug.LogError($"❌ Connection failed to {serverIP}:{currentPort}: {sockEx.Message} (ErrorCode: {sockEx.SocketErrorCode})");
            if (currentPort == port && sockEx.SocketErrorCode == SocketError.ConnectionRefused)
            {
                currentPort = fallbackPort;
                Debug.LogWarning($"[CSTClient] Switching to fallback port {fallbackPort}");
            }
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Connection failed to {serverIP}:{currentPort}: {e.Message}");
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
    }

    private IEnumerator TryReconnectCoroutine()
    {
        if (isQuitting) yield break;
        reconnectAttempts++;
        float delay = UnityEngine.Random.Range(1f, 3f);
        Debug.Log($"🔄 Attempt {reconnectAttempts}/{maxReconnectAttempts}: Reconnecting in {delay:F1} seconds...");
        waitingForResponse = false;

        CleanupConnection();

        yield return new WaitForSeconds(delay);
        ConnectToServer();
    }

    void RequestUpdate()
    {
        if (!connected || stream == null || !stream.CanWrite || !stream.CanRead) return;
        waitingForResponse = true;
        try
        {
            Debug.Log($"[CSTClient] Sending update request, Socket Connected: {client.Connected}");
            byte[] msg = Encoding.UTF8.GetBytes("update\n");
            stream.Write(msg, 0, msg.Length);
            stream.Flush();
            Debug.Log("[CSTClient] Sent update request");
        }
        catch (IOException ioEx)
        {
            Debug.LogError($"❌ Update send failed (IO): {ioEx.Message}");
            connected = false;
            waitingForResponse = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (SocketException sockEx)
        {
            Debug.LogError($"❌ Update send failed (Socket): {sockEx.Message} (ErrorCode: {sockEx.SocketErrorCode})");
            connected = false;
            waitingForResponse = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Update send failed: {e.Message}");
            connected = false;
            waitingForResponse = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
    }

    void SendHeartbeat()
    {
        if (!connected || stream == null || !stream.CanWrite || !stream.CanRead) return;
        try
        {
            Debug.Log($"[CSTClient] Sending heartbeat, Socket Connected: {client.Connected}");
            byte[] msg = Encoding.UTF8.GetBytes("ping\n");
            stream.Write(msg, 0, msg.Length);
            stream.Flush();
            Debug.Log("[CSTClient] Sent heartbeat");
        }
        catch (IOException ioEx)
        {
            Debug.LogError($"❌ Heartbeat send failed (IO): {ioEx.Message}");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (SocketException sockEx)
        {
            Debug.LogError($"❌ Heartbeat send failed (Socket): {sockEx.Message} (ErrorCode: {sockEx.SocketErrorCode})");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Heartbeat send failed: {e.Message}");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
    }

    void SendAudioData()
    {
        if (!connected || stream == null || !stream.CanWrite || !stream.CanRead || micAnalyzer == null) return;
        try
        {
            Debug.Log($"[CSTClient] Sending audio data, Socket Connected: {client.Connected}");
            AudioData audioData = new AudioData
            {
                rms = micAnalyzer.rmsValue,
                pitch = micAnalyzer.pitchValue
            };
            string json = JsonUtility.ToJson(audioData);
            byte[] msg = Encoding.UTF8.GetBytes($"audio {json}\n");
            stream.Write(msg, 0, msg.Length);
            stream.Flush();
            Debug.Log($"[CSTClient] Sent audio data: RMS={audioData.rms:F3}, Pitch={audioData.pitch:F1}Hz");
        }
        catch (IOException ioEx)
        {
            Debug.LogError($"❌ Audio send failed (IO): {ioEx.Message}");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (SocketException sockEx)
        {
            Debug.LogError($"❌ Audio send failed (Socket): {sockEx.Message} (ErrorCode: {sockEx.SocketErrorCode})");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Audio send failed: {e.Message}");
            connected = false;
            lock (lockObj)
            {
                shouldReconnect = true;
            }
        }
    }

    void ReceiveLoop()
    {
        byte[] buffer = new byte[maxBufferSize];
        StringBuilder lineBuilder = new StringBuilder();

        while (connected && !isQuitting)
        {
            try
            {
                if (!stream.DataAvailable)
                {
                    Thread.Sleep(10);
                    continue;
                }

                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead <= 0)
                {
                    Debug.LogWarning("[CSTClient] Server disconnected");
                    connected = false;
                    waitingForResponse = false;
                    lock (lockObj)
                    {
                        shouldReconnect = true;
                    }
                    continue;
                }

                string chunk = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                lineBuilder.Append(chunk);

                string allData = lineBuilder.ToString();
                int lastNewline = allData.LastIndexOf('\n');
                if (lastNewline == -1)
                {
                    Debug.Log("[CSTClient] No newline found, waiting for more data...");
                    continue;
                }

                string completeData = allData.Substring(0, lastNewline + 1);
                lineBuilder = new StringBuilder(allData.Substring(lastNewline + 1));

                string[] lines = completeData.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (string line in lines)
                {
                    string trimmedLine = line.Trim();
                    if (string.IsNullOrEmpty(trimmedLine)) continue;

                    string logLine = trimmedLine.Length <= 100 ? trimmedLine : trimmedLine.Substring(0, 100) + "...";
                    Debug.Log($"[CSTClient] Processing line: {logLine}");

                    try
                    {
                        if (trimmedLine == "{\"status\":\"connected\"}")
                        {
                            Debug.Log("[CSTClient] Received handshake response, connection confirmed");
                            connectionConfirmed = true;
                            waitingForResponse = false;
                            continue;
                        }
                        else if (trimmedLine == "{\"status\":\"ok\"}" || trimmedLine == "{\"status\":\"error\"}")
                        {
                            Debug.Log("[CSTClient] Received ping or audio response");
                            waitingForResponse = false;
                            continue;
                        }
                        else if (trimmedLine == "[]")
                        {
                            Debug.Log("[CSTClient] Received empty state response");
                            lock (lockObj)
                            {
                                pendingEntities.Enqueue(new List<CSTEntity>());
                            }
                            waitingForResponse = false;
                            continue;
                        }
                        else
                        {
                            // Parse JSON entity array
                            List<CSTEntity> parsed = JsonUtilityHelper.FromJsonArray<CSTEntity>(trimmedLine);
                            lock (lockObj)
                            {
                                var validEntities = parsed.Where(e => e != null && IsValidEntity(e)).ToList();
                                pendingEntities.Enqueue(validEntities);
                                Debug.Log($"📥 Queued {validEntities.Count} entities");
                            }
                            waitingForResponse = false;
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[CSTClient] Failed to process JSON data '{logLine}': {e.Message}");
                        waitingForResponse = false;
                    }
                }
            }
            catch (IOException ioEx)
            {
                Debug.LogWarning($"[CSTClient] IO error: {ioEx.Message}");
                connected = false;
                waitingForResponse = false;
                lock (lockObj)
                {
                    shouldReconnect = true;
                }
            }
            catch (SocketException sockEx)
            {
                Debug.LogWarning($"[CSTClient] Socket error: {sockEx.Message} (ErrorCode: {sockEx.SocketErrorCode})");
                connected = false;
                waitingForResponse = false;
                lock (lockObj)
                {
                    shouldReconnect = true;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[CSTClient] Receive error: {e.Message}");
                connected = false;
                waitingForResponse = false;
                lock (lockObj)
                {
                    shouldReconnect = true;
                }
            }
        }
    }

    private bool IsValidEntity(CSTEntity entity)
    {
        return entity != null &&
               IsValidVector3(entity.positionVector) &&
               entity.mesh_params != null &&
               entity.shader_params != null &&
               entity.texture_params != null &&
               entity.shader_params.base_color != null &&
               entity.shader_params.base_color.Length == 3;
    }

    private bool IsValidVector3(Vector3 v)
    {
        return !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                 float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));
    }

    public List<CSTEntity> GetEntities()
    {
        lock (lockObj)
        {
            Debug.Log($"[CSTClient] GetEntities called, entitiesReady={entitiesReady}, count={entities.Count}");
            if (!entitiesReady)
            {
                return new List<CSTEntity>();
            }
            return new List<CSTEntity>(entities);
        }
    }

    void CleanupConnection()
    {
        try
        {
            connected = false;
            connectionConfirmed = false;
            listenerThread?.Interrupt();
            stream?.Close();
            client?.Close();
            Debug.Log("[CSTClient] Cleaned up connection");
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Cleanup error: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        isQuitting = true;
        CleanupConnection();
        Debug.Log("🛑 CSTClient shutdown");
    }
}

public static class JsonUtilityHelper
{
    [Serializable]
    private class Wrapper<T>
    {
        public List<T> Items;
    }

    public static List<T> FromJsonArray<T>(string json)
    {
        try
        {
            if (string.IsNullOrEmpty(json) || json == "[]")
            {
                Debug.Log("[JsonUtilityHelper] Received empty JSON array");
                return new List<T>();
            }
            string wrapped = "{\"Items\":" + json + "}";
            var wrapper = JsonUtility.FromJson<Wrapper<T>>(wrapped);
            return wrapper?.Items ?? new List<T>();
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ JSON parsing failed in FromJsonArray: {e.Message}");
            return new List<T>();
        }
    }
}