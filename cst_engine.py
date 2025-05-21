import numpy as np
import json
import math
import pyaudio
from scipy.signal.windows import hamming
from scipy.spatial import cKDTree
from typing import List, Dict
import csv
import os
import datetime
import time
import traceback
import colorsys
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from ecosystem_engine import EcosystemEngine  # Import for ecosystem integration
import hashlib

# CST constants
C = 3e8  # Speed of light (m/s)
PHI = (1 + 5 ** 0.5) / 2  # Golden ratio (~1.618)
G = 6.67430e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
A0 = 1.2e-10  # Reference acceleration (m/s²)
PLANCK = 6.62607015e-34  # Planck's constant (J·s)
VOLUME_11D = 1e132  # 11D volume element (m¹¹, (1e12)¹¹)
DT = 0.01  # Time step (s)
RADIUS = 5e11  # Interaction radius for KD-tree (m)
MAX_ENTITIES = 1000000  # Limit for performance
SCENE_EXTENT = 1e6  # Unity scene size (±1e6 units)


class MemoryNodeLog:
    def __init__(self):
        self.csv_file = "memory_node_log.csv"
        self.token_file = "memory_node_tokens.json"
        self.csv_writer = None
        self.previous_token_id = None
        self.initialize_csv()
        self.tokens = self.load_tokens()

    def initialize_csv(self):
        try:
            file_exists = os.path.exists(self.csv_file) and os.stat(self.csv_file).st_size > 0
            self.csv_file_handle = open(self.csv_file, mode='a', newline='')
            self.csv_writer = csv.writer(self.csv_file_handle)
            if not file_exists:
                self.csv_writer.writerow([
                    "Timestamp", "EntityID", "Frequency", "Entropy", "RMS", "Pitch",
                    "R", "G", "B", "Hue", "Saturation", "Value", "TokenID", "PreviousTokenID"
                ])
                self.csv_file_handle.flush()
            print("[MemoryNodeLog] Initialized CSV log")
        except Exception as e:
            print(f"[MemoryNodeLog Init Error] Failed to initialize CSV: {str(e)}")
            self.csv_writer = None

    def load_tokens(self):
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[MemoryNodeLog Load Error] Failed to load tokens: {str(e)}")
            return {}

    def save_tokens(self):
        try:
            with open(self.token_file, 'w') as f:
                json.dump(self.tokens, f, indent=2)
            print("[MemoryNodeLog] Saved tokens to JSON")
        except Exception as e:
            print(f"[MemoryNodeLog Save Error] Failed to save tokens: {str(e)}")

    def generate_token_id(self, data: Dict) -> str:
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def log(self, entity_id, freq, entropy, rms, pitch, r, g, b, hue, saturation, value):
        if self.csv_writer is None:
            print("[MemoryNodeLog] CSV writer not initialized, skipping log")
            return

        timestamp = datetime.datetime.now().isoformat()
        token_data = {
            "timestamp": timestamp,
            "entity_id": entity_id,
            "frequency": freq,
            "entropy": entropy,
            "rms": rms,
            "pitch": pitch,
            "r": r,
            "g": g,
            "b": b,
            "hue": hue,
            "saturation": saturation,
            "value": value
        }
        token_id = self.generate_token_id(token_data)
        token_data["previous_token_id"] = self.previous_token_id
        self.tokens[token_id] = token_data
        self.previous_token_id = token_id

        try:
            self.csv_writer.writerow([
                timestamp, entity_id, freq, entropy, rms, pitch,
                r, g, b, hue, saturation, value, token_id, token_data["previous_token_id"]
            ])
            self.csv_file_handle.flush()
            self.save_tokens()
            print(f"[MemoryNodeLog] Logged token {token_id} for entity {entity_id}, freq={freq:.1f}")
        except Exception as e:
            print(f"[MemoryNodeLog Error] Failed to log: {str(e)}")

    def close(self):
        if self.csv_file_handle is not None:
            try:
                self.csv_file_handle.close()
                print("[MemoryNodeLog] Closed CSV log")
            except Exception as e:
                print(f"[MemoryNodeLog Close Error] Failed to close CSV: {str(e)}")
            self.csv_file_handle = None
            self.csv_writer = None


class AudioProcessor:
    def __init__(self, mock_audio=False):
        self.p = None
        self.stream = None
        self.rms = 0.3
        self.freqs = np.array([440.0])
        self.mags = np.array([1.0])
        self.window = hamming(512)[:, None]
        self.mock_audio = mock_audio
        if mock_audio:
            print("[AudioProcessor] Mock audio mode enabled")
            return
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paInt16, channels=2, rate=44100,
                                      input=True, frames_per_buffer=512)
            print("[AudioProcessor] Initialized successfully")
        except Exception as e:
            print(f"[AudioProcessor Init Error] Failed to initialize: {str(e)}")
            self.p = None
            self.stream = None

    def get_data(self):
        if self.mock_audio:
            return 0.5, np.array([440.0, 880.0]), np.array([0.7, 0.3])
        if self.stream is None:
            print("[AudioProcessor] Stream not initialized, returning default values")
            return 0.3, np.array([440.0]), np.array([1.0])
        try:
            data = np.frombuffer(self.stream.read(512, exception_on_overflow=False), dtype=np.int16)
            data = data.reshape((512, 2)).astype(np.float32) / 32768.0
            data = data * self.window
            raw_rms = np.sqrt(np.mean(data ** 2))
            local_rms = raw_rms * 1.4
            fft_vals = np.fft.rfft(data[:, 0])
            local_freqs = np.fft.rfftfreq(512, 1.0 / 44100)
            local_mags = np.abs(fft_vals)
            valid = (local_freqs >= 20) & (local_freqs <= 20000)
            local_freqs, local_mags = local_freqs[valid], local_mags[valid]
            if local_mags.size > 0:
                local_mags = local_mags / np.max(local_mags + 1e-10)
                freq_weights = 1.0 / (local_freqs + 1e-10)
                local_mags = local_mags * freq_weights
                local_mags = local_mags / np.sum(local_mags)
            else:
                local_freqs, local_mags = np.array([440.0]), np.array([1.0])
            self.rms = (1 - 0.1) * self.rms + 0.1 * min(local_rms, 1.0)
            self.freqs = local_freqs
            self.mags = local_mags
            print(
                f"[AudioProcessor] Raw RMS={raw_rms:.3f}, Adjusted RMS={local_rms:.3f}, Num Frequencies={len(local_freqs)}, Freq Range=[{local_freqs[0]:.1f}, {local_freqs[-1]:.1f}]")
            return self.rms, self.freqs, self.mags
        except Exception as e:
            print(f"[AudioProcessor Error] Failed to process audio data: {str(e)}")
            return 0.3, np.array([440.0]), np.array([1.0])

    def stop(self):
        if self.mock_audio:
            print("[AudioProcessor] Mock audio mode, no cleanup needed")
            return
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"[AudioProcessor Stop Error] {str(e)}")
        if self.p is not None:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"[AudioProcessor Terminate Error] {str(e)}")
        print("[AudioProcessor] Stopped")


def sanitize_vector(vec, scale=1.0):
    return [
        float(min(max(x * scale, -SCENE_EXTENT * 1e3), SCENE_EXTENT * 1e3)) if math.isfinite(x) else 0.0
        for x in vec
    ]


def freq_to_light(local_freq, local_mag=1.0, value_mod=1.0, local_entropy=0.8, local_rms=0.3, local_pitch=440.0):
    """
    Convert frequency to RGB color, incorporating magnitude, value modifier, entropy, and audio inputs.

    Args:
        local_freq (float): Entity frequency (e.g., 22710.5 Hz)
        local_mag (float): Magnitude (e.g., 1.0)
        value_mod (float): Value modifier (e.g., 1.317)
        local_entropy (float): Entity entropy (e.g., 0.634)
        local_rms (float): Audio RMS value (e.g., 0.3)
        local_pitch (float): Audio pitch value (e.g., 440.0 Hz)

    Returns:
        tuple: RGB color as (r, g, b), each in [0, 1]
    """
    try:
        # Clamp inputs
        local_freq = min(max(local_freq, 20.0), 30000.0) if math.isfinite(local_freq) else 440.0
        local_mag = min(max(local_mag, 0.0), 2.0) if math.isfinite(local_mag) else 1.0
        value_mod = min(max(value_mod, 0.0), 2.0) if math.isfinite(value_mod) else 1.0
        local_entropy = min(max(local_entropy, 0.0), 1.0) if math.isfinite(local_entropy) else 0.8
        local_rms = min(max(local_rms, 0.0), 1.0) if math.isfinite(local_rms) else 0.3
        local_pitch = min(max(local_pitch, 20.0), 20000.0) if math.isfinite(local_pitch) else 440.0

        # Normalize frequency to a range for hue mapping
        min_freq = 20.0
        max_freq = 30000.0
        freq_norm = (local_freq - min_freq) / (max_freq - min_freq)
        freq_norm = max(0.0, min(1.0, freq_norm))

        # Map frequency to hue (0 to 1, corresponding to 0 to 360 degrees in HSV)
        hue = freq_norm  # Use full HSV spectrum for maximum variation

        # Adjust saturation based on entropy and RMS
        saturation = 0.7 + (local_entropy * 0.2 + local_rms * 0.1)  # Range: [0.7, 1.0]
        saturation = max(0.0, min(1.0, saturation))

        # Adjust value (brightness) based on mag, value_mod, and pitch
        pitch_norm = (local_pitch - 20.0) / (20000.0 - 20.0)
        value = local_mag * value_mod * (0.5 + local_entropy * 0.3 + pitch_norm * 0.2)
        value = max(0.0, min(1.0, value))

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = rgb

        print(
            f"[FreqToLight] Freq={local_freq:.1f}, Mag={local_mag:.3f}, ValueMod={value_mod:.3f}, Entropy={local_entropy:.3f}, RMS={local_rms:.3f}, Pitch={local_pitch:.1f}, RGB=({r:.3f}, {g:.3f}, {b:.3f})")
        return rgb
    except Exception as e:
        print(f"[FreqToLight Error] Failed to convert frequency to light: {str(e)}")
        return (0.5, 0.5, 0.5)


class CSTEntity:
    def __init__(self, entity_id, mass, position_11d, velocity_11d, entropy=0.0, freq=440.0, entity_type=0):
        self.id = entity_id
        self.mass = float(min(max(mass, 1e25), 1e35)) if math.isfinite(mass) else 1e30
        self.r11d = np.clip(np.array(position_11d, dtype=np.float64), -1e12, 1e12)
        self.v11d = np.clip(np.array(velocity_11d, dtype=np.float64), -1e6, 1e6)
        self.entropy = float(min(max(entropy, 0.0), 1.0)) if math.isfinite(entropy) else 0.5
        self.memory_vector = np.random.rand(12)
        self.E_rest = self.mass * C ** 2
        self.E_chaos = np.random.uniform(0, 0.1 * self.E_rest)
        self.Ec = self.E_rest + self.E_chaos
        self.freq = float(min(max(freq, 20.0), 20000.0)) if math.isfinite(freq) else 440.0
        self.psi = 0.0
        self.entity_type = int(entity_type)  # 0=Star, 1=Planet, 2=BlackHole, 3=Nebula, 4=Galaxy
        self.age = 0.0
        self.ecosystem_level = 0.0 if entity_type != 1 else np.random.uniform(0.1, 1.0)
        self.last_exported_pos = self.r11d.copy()
        self.last_exported_time = 0.0
        # CST-specific fields
        self.lyapunov_exponent = 0.0
        self.path_length = 0.0
        self.synaptic_strength = 0.0
        self.gravitational_potential = 0.0
        print(
            f"[CSTEntity Init {self.id}] Mass={self.mass:.1e}, Pos={self.r11d[:3]}, Vel={self.v11d[:3]}, Entropy={self.entropy:.3f}, Freq={self.freq:.1f}, Type={self.entity_type}, Age={self.age:.1f}, Ecosystem={self.ecosystem_level:.1f}")

    def compute_lyapunov(self, neighbors, delta_t):
        if not neighbors:
            return 0.01
        vel_diffs = [np.log1p(np.abs(np.sum(self.v11d - n.v11d))) for n in neighbors]
        lam = np.mean(vel_diffs) / delta_t if vel_diffs else 0.01
        if not math.isfinite(lam):
            lam = 0.01
        print(f"[Lyapunov {self.id}] Neighbors={len(neighbors)}, Lambda={lam:.3f}")
        self.lyapunov_exponent = min(max(lam, 0.0), 0.1)
        return self.lyapunov_exponent

    def compute_path_length(self, delta_t):
        L = np.sqrt(np.sum(self.v11d ** 2)) * delta_t
        if not math.isfinite(L):
            L = 0.0
        self.path_length = L
        return L

    def update_energy(self, neighbors, delta_t):
        lorenz_state = self.memory_vector[:11]
        # Clamp lorenz_state to prevent numerical overflow
        lorenz_state = np.clip(lorenz_state, -1e10, 1e10)
        dx_dt = np.zeros(11)
        sigma, beta = 10.0, 8 / 3
        for i in range(10):
            dx_dt[i] = sigma * (lorenz_state[i + 1] - lorenz_state[i])
        # Compute the squared sum with clamping to avoid overflow
        squared_sum = np.sum(np.clip(lorenz_state[:10] ** 2, 0, 1e20))
        dx_dt[10] = -beta * lorenz_state[10] + squared_sum
        # Clamp dx_dt to prevent extreme values
        dx_dt = np.clip(dx_dt, -1e10, 1e10)
        self.memory_vector[:11] += delta_t * dx_dt
        # Clamp memory_vector to prevent further overflow
        self.memory_vector = np.clip(self.memory_vector, -1e10, 1e10)
        E_chaos = 0.5 * self.mass * np.sum(self.memory_vector[:11] ** 2)
        self.E_chaos = min(E_chaos, 0.1 * self.E_rest) if math.isfinite(E_chaos) else 0.0
        self.Ec = self.E_rest + self.E_chaos
        raw_freq = self.Ec / PLANCK
        self.freq = 20.0 + (20000.0 - 20.0) * (np.log10(raw_freq + 1) / 75) if math.isfinite(raw_freq) else 440.0
        interaction_term = np.mean([np.linalg.norm(self.v11d - n.v11d) for n in neighbors]) if neighbors else 0.0
        if not math.isfinite(interaction_term):
            interaction_term = 0.0
        self.entropy = min(self.entropy + interaction_term * 0.0001 * delta_t, 0.8)
        self.age += delta_t
        if self.entity_type == 1:
            self.ecosystem_level = min(self.ecosystem_level + self.entropy * 0.001 * delta_t, 1.0)
        print(
            f"[UpdateEnergy {self.id}] E_chaos={self.E_chaos:.1e}, Ec={self.Ec:.1e}, Freq={self.freq:.1f}, Entropy={self.entropy:.3f}, Age={self.age:.1f}, Ecosystem={self.ecosystem_level:.1f}")

    def compute_psi(self, lam, L, Omega, Ugrav, delta_t):
        try:
            if not all(math.isfinite(x) for x in [self.Ec, lam, L, Omega, Ugrav, delta_t]):
                raise ValueError("Invalid input: NaN or Infinity detected")
            term1 = PHI * self.Ec
            term2 = lam * self.Ec * delta_t
            term3 = L * self.mass * C ** 2 / 1e12
            term4 = Omega * self.Ec / A0
            term5 = Ugrav
            terms = [term1, term2, term3, term4, term5]
            if not all(math.isfinite(t) for t in terms):
                raise ValueError("Invalid term: NaN or Infinity in terms")
            psi = (term1 + term2 + term3 + term4 + term5) / VOLUME_11D
            if not math.isfinite(psi):
                raise ValueError("Invalid psi: NaN or Infinity")
            psi = min(max(psi, -1e-10), 1e-10)
            print(
                f"[ComputePsi {self.id}] Ec={self.Ec:.1e}, term1={term1:.1e}, term2={term2:.1e}, term3={term3:.1e}, term4={term4:.1e}, term5={term5:.1e}, Psi={psi:.1e}")
            self.psi = psi
            return psi
        except Exception as e:
            print(f"[ComputePsi Error {self.id}] Failed to compute psi: {str(e)}")
            self.psi = 0.0
            return 0.0

    def needs_export(self, current_time, pos_threshold=1e4, time_threshold=0.05):
        pos_diff = np.linalg.norm(self.r11d - self.last_exported_pos)
        time_diff = current_time - self.last_exported_time
        return pos_diff > pos_threshold or time_diff > time_threshold


class CSTUniverse:
    def __init__(self):
        self.entities: List[CSTEntity] = []
        self.existing_positions = np.empty((0, 3))
        self.next_id = 0
        self.sound_event_id = 0
        self.last_update_time = time.time()
        self.last_tree = None
        self.last_positions = None
        self.audio_data = None  # Store Unity audio data
        self.frame_counter = 0  # For throttling plot generation
        self.freq_to_light_log = None
        self.freq_to_light_writer = None
        self.freq_to_light_data = []  # Store for plotting
        self.memory_node_log = MemoryNodeLog()  # Initialize memory node log
        self.ecosystem_engine = EcosystemEngine()  # Initialize ecosystem engine
        try:
            self.audio = AudioProcessor(mock_audio=os.getenv("MOCK_AUDIO", "0") == "1")
        except Exception as e:
            print(f"[CSTUniverse Init Error] Failed to initialize AudioProcessor: {str(e)}")
            self.audio = None
        self.csv_file = None
        self.csv_writer = None
        self.math_log_file = None
        self.math_log_writer = None
        # Initialize logging for FreqToLight
        try:
            self.freq_to_light_log = open("freq_to_light_log.csv", mode='a', newline='')
            self.freq_to_light_writer = csv.writer(self.freq_to_light_log)
            # Write header if file is empty
            if os.stat("freq_to_light_log.csv").st_size == 0:
                self.freq_to_light_writer.writerow([
                    "Timestamp", "EntityID", "Frequency", "Entropy", "RMS", "Pitch",
                    "R", "G", "B", "Hue", "Saturation", "Value"
                ])
                self.freq_to_light_log.flush()
            print("[CSTUniverse Init] FreqToLight logging initialized")
        except Exception as e:
            print(f"[CSTUniverse Init Error] Failed to initialize FreqToLight log: {str(e)}")
            self.freq_to_light_log = None
            self.freq_to_light_writer = None
        try:
            self.add_entities(10)  # Reduced to 10 entities
            print(f"[CSTUniverse Init] Successfully added {len(self.entities)} initial entities")
        except Exception as e:
            print(f"[CSTUniverse Init Error] Failed to add initial entities: {str(e)}")

    def log_math(self, entity_id, operation, details):
        if self.math_log_writer is not None:
            timestamp = datetime.datetime.now().isoformat()
            try:
                self.math_log_writer.writerow([timestamp, entity_id, operation, details])
                self.math_log_file.flush()
            except Exception as e:
                print(f"[LogMath Error] Failed to write to math_log.csv: {str(e)}")

    def log_freq_to_light(self, entity_id, local_freq, local_entropy, local_rms, local_pitch, r, g, b, hue, saturation,
                          value):
        if self.freq_to_light_writer is not None:
            timestamp = datetime.datetime.now().isoformat()
            try:
                self.freq_to_light_writer.writerow([
                    timestamp, entity_id, local_freq, local_entropy, local_rms, local_pitch,
                    r, g, b, hue, saturation, value
                ])
                self.freq_to_light_log.flush()
                # Store data for plotting
                self.freq_to_light_data.append({
                    "timestamp": timestamp,
                    "entity_id": entity_id,
                    "freq": local_freq,
                    "entropy": local_entropy,
                    "rms": local_rms,
                    "pitch": local_pitch,
                    "r": r,
                    "g": g,
                    "b": b,
                    "hue": hue,
                    "saturation": saturation,
                    "value": value
                })
            except Exception as e:
                print(f"[LogFreqToLight Error] Failed to write to freq_to_light_log.csv: {str(e)}")

        # Log to memory node log
        self.memory_node_log.log(
            entity_id, local_freq, local_entropy, local_rms, local_pitch,
            r, g, b, hue, saturation, value
        )

    def generate_plots(self):
        if not self.freq_to_light_data:
            print("[GeneratePlots] No FreqToLight data to plot")
            return

        try:
            # Create plots directory if it doesn't exist
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plot 1: Frequency vs RGB Components Over Time
            timestamps = [data["timestamp"] for data in self.freq_to_light_data]
            freqs = [data["freq"] for data in self.freq_to_light_data]
            r_values = [data["r"] for data in self.freq_to_light_data]
            g_values = [data["g"] for data in self.freq_to_light_data]
            b_values = [data["b"] for data in self.freq_to_light_data]

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, freqs, label="Frequency (Hz)", color="black")
            plt.xlabel("Timestamp")
            plt.ylabel("Frequency (Hz)")
            plt.title("Frequency Over Time")
            plt.xticks(rotation=45)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(timestamps, r_values, label="R", color="red")
            plt.plot(timestamps, g_values, label="G", color="green")
            plt.plot(timestamps, b_values, label="B", color="blue")
            plt.xlabel("Timestamp")
            plt.ylabel("RGB Component")
            plt.title("RGB Components Over Time")
            plt.xticks(rotation=45)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"plots/freq_rgb_over_time_{timestamp}.png")
            plt.close()
            print(f"[GeneratePlots] Saved frequency vs RGB plot to plots/freq_rgb_over_time_{timestamp}.png")

            # Plot 2: RGB Scatter Plot
            plt.figure(figsize=(8, 8))
            colors = [(data["r"], data["g"], data["b"]) for data in self.freq_to_light_data]
            plt.scatter(
                [data["r"] for data in self.freq_to_light_data],
                [data["g"] for data in self.freq_to_light_data],
                c=colors,
                s=50,
                alpha=0.6
            )
            plt.xlabel("R")
            plt.ylabel("G")
            plt.title("RGB Color Distribution (R vs G)")
            plt.savefig(f"plots/rgb_scatter_{timestamp}.png")
            plt.close()
            print(f"[GeneratePlots] Saved RGB scatter plot to plots/rgb_scatter_{timestamp}.png")

            # Plot 3: Entropy vs Hue
            plt.figure(figsize=(8, 6))
            entropies = [data["entropy"] for data in self.freq_to_light_data]
            hues = [data["hue"] for data in self.freq_to_light_data]
            plt.scatter(entropies, hues, c=colors, s=50, alpha=0.6)
            plt.xlabel("Entropy")
            plt.ylabel("Hue")
            plt.title("Entropy vs Hue")
            plt.savefig(f"plots/entropy_vs_hue_{timestamp}.png")
            plt.close()
            print(f"[GeneratePlots] Saved entropy vs hue plot to plots/entropy_vs_hue_{timestamp}.png")

            # Clear data to prevent memory buildup
            self.freq_to_light_data = []
        except Exception as e:
            print(f"[GeneratePlots Error] Failed to generate plots: {str(e)}")
            print(traceback.format_exc())

    def generate_spaced_position(self, min_distance=1e10, max_attempts=20):
        for attempt in range(max_attempts):
            pos = np.random.uniform(-SCENE_EXTENT * 1e3, SCENE_EXTENT * 1e3, 11)  # Adjusted to match Unity scale
            if self.existing_positions.size == 0:
                print(f"[GeneratePosition] Entity {self.next_id}: New position {pos[:3]} (first entity)")
                return pos
            distances = np.linalg.norm(self.existing_positions - pos[:3], axis=1)
            min_dist = np.min(distances) if distances.size > 0 else float('inf')
            if np.all(distances >= min_distance):
                print(
                    f"[GeneratePosition] Entity {self.next_id}: New position {pos[:3]}, min distance {min_dist:.1e}, attempt {attempt + 1}")
                return pos
            print(
                f"[GeneratePosition] Entity {self.next_id}: Attempt {attempt + 1} failed, min distance {min_dist:.1e}")
        pos = np.random.uniform(-SCENE_EXTENT * 2e3, SCENE_EXTENT * 2e3, 11)
        print(f"[GeneratePosition] Entity {self.next_id}: Fallback position {pos[:3]} after {max_attempts} attempts")
        return pos

    def add_entities(self, count, freq=440.0, mag=1.0):
        initial_count = len(self.entities)
        for _ in range(count):
            if len(self.entities) >= MAX_ENTITIES:
                print("[AddEntities] Max entity limit reached")
                break
            entity_type = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.4, 0.1, 0.1, 0.1])
            mass = (5e37 if entity_type == 2 else
                    np.random.uniform(1e25, 1e30) if entity_type == 0 else
                    np.random.uniform(5e21, 5e24) if entity_type == 1 else
                    np.random.uniform(1e25, 1e27) if entity_type == 3 else
                    np.random.uniform(1e30, 1e33)) * (1 + mag)
            try:
                pos = self.generate_spaced_position(min_distance=1e10)
                self.existing_positions = np.append(self.existing_positions, [pos[:3]], axis=0)
            except Exception as e:
                print(f"[AddEntities Error] Failed to generate position for entity {self.next_id}: {str(e)}")
                continue
            vel = np.random.uniform(-1e5, 1e5, 11) * (1 + mag)
            entropy = np.random.uniform(0.3, 0.8) * (1 + mag)
            entropy = min(max(entropy, 0.3), 0.8)
            try:
                entity = CSTEntity(self.next_id, mass, pos, vel, entropy, freq, entity_type)
                self.entities.append(entity)
                # Add ecosystem for planets (entity_type == 1)
                if entity.entity_type == 1:
                    self.ecosystem_engine.add_ecosystem(entity)
                self.next_id += 1
                print(f"[AddEntity] Added entity {entity.id}, type={entity_type}, freq={freq:.1f}")
            except Exception as e:
                print(f"[AddEntities Error] Failed to create entity {self.next_id}: {str(e)}")
                continue
        added_count = len(self.entities) - initial_count
        print(f"[AddEntities] Added {added_count} entities, total {len(self.entities)}")

    def spawn_from_audio(self, local_rms, local_freqs, local_mags):
        if len(self.entities) >= MAX_ENTITIES:
            print("[SpawnFromAudio] Entity limit reached, skipping spawn")
            return
        max_spawns = min(1, int(local_rms * 10 / (1 + len(self.entities) / 10000)))
        valid_idx = np.where(local_mags > 0.0001)[0]
        if len(valid_idx) > max_spawns:
            weights = local_mags[valid_idx] / np.sum(local_mags[valid_idx])
            selected_idx = np.random.choice(valid_idx, size=max_spawns, replace=False, p=weights)
        else:
            selected_idx = valid_idx
        print(
            f"[SpawnFromAudio] RMS={local_rms:.3f}, MaxSpawns={max_spawns}, ValidFreqs={len(valid_idx)}, SelectedFreqs={len(selected_idx)}")
        initial_count = len(self.entities)
        added_count = 0
        local_pitch = local_freqs[0] if len(local_freqs) > 0 else 440.0  # Use first frequency as pitch
        for idx in selected_idx:
            sound_event_id = self.sound_event_id
            self.sound_event_id += 1
            freq = local_freqs[idx]
            entity_type = (2 if freq < 500 else
                           0 if freq < 2000 else
                           1 if freq < 8000 else
                           3 if freq < 12000 else
                           4)
            mass = (5e37 if entity_type == 2 else
                    np.random.uniform(1e25, 1e30) if entity_type == 0 else
                    np.random.uniform(5e21, 5e24) if entity_type == 1 else
                    np.random.uniform(1e25, 1e27) if entity_type == 3 else
                    np.random.uniform(1e30, 1e33)) * (1 + local_rms)
            try:
                pos = self.generate_spaced_position(min_distance=1e10)
                self.existing_positions = np.append(self.existing_positions, [pos[:3]], axis=0)
            except Exception as e:
                print(f"[SpawnFromAudio Error] Failed to generate position for entity {self.next_id}: {str(e)}")
                continue
            vel = np.random.normal(0, 1e5 * local_mags[idx], 11)
            entropy = np.random.uniform(0.3, 0.8) * (1 + local_mags[idx])
            entropy = min(max(entropy, 0.3), 0.8)
            if entity_type == 1 and entropy <= 0.5:
                print(f"[SpawnFromAudio] Skipping low-entropy planet (ID={self.next_id}, Entropy={entropy:.3f})")
                continue
            r, g, b = freq_to_light(freq, local_mag=local_mags[idx], value_mod=1.0 + entropy * 0.5,
                                    local_entropy=entropy,
                                    local_rms=local_rms, local_pitch=local_pitch)
            # Log FreqToLight data
            hsv = colorsys.rgb_to_hsv(r, g, b)
            self.log_freq_to_light(self.next_id, freq, entropy, local_rms, local_pitch, r, g, b, hsv[0], hsv[1], hsv[2])
            print(
                f"[SoundEvent {sound_event_id}] RMS={local_rms:.3f}, Freq={freq:.1f}Hz, Type={entity_type}, Color=(R={r:.3f}, G={g:.3f}, B={b:.3f}), Entropy={entropy:.3f}, EntityID={self.next_id}")
            try:
                entity = CSTEntity(self.next_id, mass, pos, vel, entropy, freq, entity_type)
                self.entities.append(entity)
                # Add ecosystem for planets (entity_type == 1)
                if entity.entity_type == 1:
                    self.ecosystem_engine.add_ecosystem(entity)
                self.next_id += 1
                added_count += 1
                print(f"[AddEntity] Added audio-spawned entity {entity.id}, type={entity_type}, freq={freq:.1f}")
            except Exception as e:
                print(f"[SpawnFromAudio Error] Failed to create entity {self.next_id}: {str(e)}")
                continue
        print(f"[SpawnFromAudio] Added {added_count} entities, total {len(self.entities)}")

    def process_audio(self, audio_data):
        try:
            local_rms = float(audio_data.get('rms', 0.3))
            local_pitch = float(audio_data.get('pitch', 440.0))
            if not math.isfinite(local_rms) or not math.isfinite(local_pitch):
                print("[ProcessAudio] Invalid audio data, using defaults")
                local_rms, local_pitch = 0.3, 440.0
            local_rms = min(max(local_rms, 0.0), 1.0)
            local_pitch = min(max(local_pitch, 20.0), 20000.0)
            self.audio_data = {'rms': local_rms, 'freqs': np.array([local_pitch]), 'mags': np.array([1.0])}
            print(f"[ProcessAudio] Received Unity audio: RMS={local_rms:.3f}, Pitch={local_pitch:.1f}Hz")
        except Exception as e:
            print(f"[ProcessAudio Error] Failed to process audio data: {str(e)}")
            self.audio_data = None

    def step(self, dt):
        current_time = time.time()
        if current_time - self.last_update_time < 0.05:
            return
        self.last_update_time = current_time
        self.frame_counter += 1
        try:
            if self.audio_data is not None:
                try:
                    local_rms, local_freqs, local_mags = self.audio_data['rms'], self.audio_data['freqs'], \
                    self.audio_data['mags']
                    if len(self.entities) < MAX_ENTITIES:
                        self.spawn_from_audio(local_rms, local_freqs, local_mags)
                except Exception as e:
                    print(f"[Step Error] Failed to process Unity audio data: {str(e)}")
            elif self.audio is not None:
                try:
                    local_rms, local_freqs, local_mags = self.audio.get_data()
                    if len(self.entities) < MAX_ENTITIES:
                        self.spawn_from_audio(local_rms, local_freqs, local_mags)
                except Exception as e:
                    print(f"[Step Error] Failed to process local audio data: {str(e)}")
            else:
                print("[Step] No audio data available, skipping audio spawn")
            # Update ecosystems
            self.ecosystem_engine.update(self.entities, self.audio_data, dt)
            try:
                positions = np.array([e.r11d for e in self.entities])
                # Update last_positions to match the current number of entities
                if self.last_positions is None or self.last_positions.shape[0] != len(self.entities):
                    self.last_tree = cKDTree(positions)
                    self.last_positions = positions.copy()
                elif np.any(np.abs(positions - self.last_positions) > 1e5):
                    self.last_tree = cKDTree(positions)
                    self.last_positions = positions.copy()
                tree = self.last_tree
                for e in self.entities:
                    indices = tree.query_ball_point(e.r11d, RADIUS)
                    neighbors = [self.entities[i] for i in indices if i != e.id]
                    force = self.compute_net_force(e, neighbors)
                    acc = force / (e.mass + 1e-20)
                    e.v11d += acc * dt
                    e.r11d += e.v11d * dt
                    e.update_energy(neighbors, dt)
                    self.existing_positions[e.id] = e.r11d[:3]
            except Exception as e:
                print(f"[Step Error] Failed to update entities: {str(e)}")
                print(traceback.format_exc())
        except Exception as e:
            print(f"[Step Error] General failure in step: {str(e)}")
            print(traceback.format_exc())
        print(f"[Step] Total entities: {len(self.entities)}")

    def compute_net_force(self, target: CSTEntity, neighbors: List[CSTEntity]) -> np.ndarray:
        F_total = np.zeros(11)
        for other in neighbors:
            delta = other.r11d - target.r11d
            r2 = np.sum(delta ** 2) + 1e-10
            r = np.sqrt(r2)
            F_grav = G * target.mass * other.mass / r2
            direction = delta / r
            F_total += F_grav * direction
            rho_DM = 1e-22 / (1 + (r / 1e11) ** 2)
            F_total += -1e-10 * rho_DM * delta
        F_total = np.clip(F_total, -1e10, 1e10)
        print(f"[ComputeNetForce {target.id}] Total Force={F_total[:3]}")
        return F_total

    def export_state(self) -> List[Dict]:
        result = []
        if not self.entities:
            print("[ExportState] No entities to export")
            return result
        current_time = time.time()
        try:
            positions = np.array([e.r11d for e in self.entities])
            tree = cKDTree(positions)
            max_mass = max([e.mass for e in self.entities]) if self.entities else 1e30
            # Get audio data for color calculation
            export_rms = self.audio_data['rms'] if self.audio_data else 0.3
            export_pitch = self.audio_data['freqs'][0] if self.audio_data and len(
                self.audio_data['freqs']) > 0 else 440.0
            for e in self.entities:
                if not e.needs_export(current_time):
                    continue
                indices = tree.query_ball_point(e.r11d, RADIUS)
                neighbors = [self.entities[i] for i in indices if i != e.id]
                lam = e.compute_lyapunov(neighbors, DT)
                L = e.compute_path_length(DT)
                Omega = sum([
                    (G * e.mass * other.mass / ((np.linalg.norm(e.r11d - other.r11d) + 1e-6) ** 2 * A0)) +
                    (1e-22 / (1 + (np.linalg.norm(e.r11d - other.r11d) / 1e11) ** 2) * 1e-10 / A0)
                    for other in neighbors
                ])
                Omega = np.clip(Omega, -1e5 * max_mass ** 2, 1e5 * max_mass ** 2)
                e.synaptic_strength = Omega
                Ugrav = -sum([
                    G * e.mass * other.mass / (np.linalg.norm(e.r11d - other.r11d) + 1e-6)
                    for other in neighbors
                ])
                Ugrav = np.clip(Ugrav, -1e10, 1e10)
                e.gravitational_potential = Ugrav
                psi = e.compute_psi(lam, L, Omega, Ugrav, DT)
                projected_pos = sanitize_vector(e.r11d[:3], scale=1.0)
                projected_vel = sanitize_vector(e.v11d[:3], scale=1.0)
                r, g, b = freq_to_light(e.freq, local_mag=1.0, value_mod=1.0 + e.entropy * 0.5, local_entropy=e.entropy,
                                        local_rms=export_rms, local_pitch=export_pitch)
                # Log FreqToLight data
                hsv = colorsys.rgb_to_hsv(r, g, b)
                self.log_freq_to_light(e.id, e.freq, e.entropy, export_rms, export_pitch, r, g, b, hsv[0], hsv[1],
                                       hsv[2])
                # Validate all fields for finiteness
                if not (math.isfinite(e.mass) and math.isfinite(e.entropy) and math.isfinite(e.freq) and
                        math.isfinite(psi) and math.isfinite(e.ecosystem_level) and math.isfinite(
                            e.lyapunov_exponent) and
                        math.isfinite(e.path_length) and math.isfinite(e.synaptic_strength) and math.isfinite(
                            e.gravitational_potential)):
                    print(f"[ExportState Warning] Skipping entity {e.id} due to non-finite values")
                    continue
                # Get biome based on frequency
                biome = ("desert" if e.freq < 500 else
                         "forest" if e.freq < 2000 else
                         "ocean" if e.freq < 8000 else
                         "tundra")
                # Get ecosystem data for planets
                ecosystem_data = self.ecosystem_engine.export(e.id) if e.entity_type == 1 else None
                print(
                    f"[Export] Entity {e.id}: Freq={e.freq:.1f}, Entropy={e.entropy:.3f}, Psi={psi:.3f}, Pos={projected_pos}, Mass={e.mass:.1e}, Type={e.entity_type}, Ecosystem={e.ecosystem_level:.1f}")
                result.append({
                    "id": int(e.id),
                    "mass": float(e.mass),
                    "position": projected_pos,
                    "velocity": projected_vel,
                    "psi": float(psi),
                    "entropy": float(e.entropy),
                    "frequency": float(e.freq),
                    "entity_type": int(e.entity_type),
                    "ecosystem_level": float(e.ecosystem_level),
                    "mesh_params": {
                        "type": "sphere" if e.entity_type in [0, 1, 2] else "cube" if e.entity_type == 3 else "cluster",
                        "radius": 1.0 + e.entropy * 2.0 + e.ecosystem_level * 2.0 + e.mass / 1e30,
                        "segments": 16 if e.entity_type in [0, 1] else 8,
                        "biome": biome,
                        "terrain_roughness": e.entropy * 2.0
                    },
                    "shader_params": {
                        "base_color": [r, g, b],
                        "emission_power": 1.5 + e.entropy * 3.0 + e.ecosystem_level,
                        "noise_scale": 5.0 + e.entropy * 5.0 + e.ecosystem_level * 2.0,
                        "pulse_speed": 1.0 + (psi / 1e-10) * 2.0 * PHI
                    },
                    "texture_params": {
                        "noise_type": "perlin" if e.entity_type != 1 else "planetary",
                        "freq_scale": e.freq / 20000.0 * 10.0,
                        "entropy_scale": e.entropy * 5.0 + e.ecosystem_level * 3.0
                    },
                    "lyapunov_exponent": float(e.lyapunov_exponent),
                    "path_length": float(e.path_length),
                    "synaptic_strength": float(e.synaptic_strength),
                    "gravitational_potential": float(e.gravitational_potential),
                    "ecosystem_data": ecosystem_data
                })
                e.last_exported_pos = e.r11d.copy()
                e.last_exported_time = current_time
            # Generate plots every 100 frames
            if self.frame_counter % 100 == 0:
                self.generate_plots()
            # Debug JSON output
            print(f"[ExportState] Exporting {len(result)} entities: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"[ExportState Error] Failed to export state: {str(e)}")
            print(traceback.format_exc())
        print(f"[ExportState] Exported {len(result)} entities")
        return result

    def cleanup(self):
        try:
            if self.audio is not None:
                self.audio.stop()
            if self.freq_to_light_log is not None:
                self.freq_to_light_log.close()
            if self.csv_file is not None:
                self.csv_file.close()
            if self.math_log_file is not None:
                self.math_log_file.close()
            self.memory_node_log.close()
            print("[CSTUniverse Cleanup] Successfully cleaned up resources")
        except Exception as e:
            print(f"[CSTUniverse Cleanup Error] Failed to cleanup: {str(e)}")


class CSTEngine:
    def __init__(self):
        try:
            self.universe = CSTUniverse()
            print("[CSTEngine] Initialized successfully")
        except Exception as e:
            print(f"[CSTEngine Init Error] Failed to initialize: {str(e)}")
            raise

    def update(self, dt):
        try:
            self.universe.step(dt)
            state = self.universe.export_state()
            return json.dumps(state)
        except Exception as e:
            print(f"[CSTEngine Update Error] Failed to update: {str(e)}")
            return "[]"

    def ping(self):
        try:
            return json.dumps({"status": "ok", "entities": len(self.universe.entities)})
        except Exception as e:
            print(f"[CSTEngine Ping Error] Failed to ping: {str(e)}")
            return json.dumps({"status": "error"})

    def process_audio(self, audio_data):
        try:
            self.universe.process_audio(audio_data)
        except Exception as e:
            print(f"[CSTEngine ProcessAudio Error] Failed to process audio: {str(e)}")
            raise

    def cleanup(self):
        try:
            self.universe.cleanup()
            print("[CSTEngine] Cleaned up successfully")
        except Exception as e:
            print(f"[CSTEngine Cleanup Error] Failed to cleanup: {str(e)}")