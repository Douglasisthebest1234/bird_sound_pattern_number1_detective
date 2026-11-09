import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

filename = input("Audio file: ")

# Load audio
y, sr = librosa.load(filename, sr=None)

print("\n===== RECORDING SUMMARY =====")

# Duration
duration = librosa.get_duration(y=y, sr=sr)
print(f"Duration: {duration:.2f} s")

# -------------------
# PITCH ANALYSIS
# -------------------
f0, _, _ = librosa.pyin(
    y,
    fmin=100,
    fmax=10000
)

pitch = f0[~np.isnan(f0)]

if len(pitch):
    avg_pitch = np.mean(pitch)
    min_pitch = np.min(pitch)
    max_pitch = np.max(pitch)

    print(f"\nAverage pitch: {avg_pitch:.1f} Hz")
    print(f"Pitch range: {min_pitch:.1f} - {max_pitch:.1f} Hz")

    pitch_change = np.mean(np.diff(pitch))

    if pitch_change > 5:
        print("Pitch pattern: generally rising")
    elif pitch_change < -5:
        print("Pitch pattern: generally falling")
    else:
        print("Pitch pattern: mostly stable")

# -------------------
# LOUDNESS / ENERGY
# -------------------
energy = librosa.feature.rms(y=y)[0]

print(f"\nAverage loudness: {np.mean(energy):.4f}")
print(f"Maximum loudness: {np.max(energy):.4f}")

# -------------------
# CHIRP DETECTION
# -------------------
peaks, _ = find_peaks(
    energy,
    height=np.mean(energy) + np.std(energy),
    distance=10
)

times = librosa.frames_to_time(peaks, sr=sr)

print(f"\nDetected calls/chirps: {len(peaks)}")

# -------------------
# RHYTHM ANALYSIS
# -------------------
if len(times) > 1:

    intervals = np.diff(times)

    avg_gap = np.mean(intervals)
    rhythm_var = np.std(intervals)

    print(f"Average gap between calls: {avg_gap:.2f} s")
    print(f"Rhythm variation: {rhythm_var:.2f} s")

    chirps_per_second = len(peaks) / duration

    print(f"Chirp rate: {chirps_per_second:.2f} chirps/s")

    if chirps_per_second > 8:
        print("Tempo: very fast")
    elif chirps_per_second > 4:
        print("Tempo: fast")
    elif chirps_per_second > 1:
        print("Tempo: moderate")
    else:
        print("Tempo: slow")

    if rhythm_var < 0.05:
        print("Sequence type: highly regular trill")
    elif rhythm_var < 0.15:
        print("Sequence type: moderately regular")
    else:
        print("Sequence type: irregular/variable")

# -------------------
# REPETITION LEVEL
# -------------------
if len(peaks) < 5:
    print("Repetition: low")
elif len(peaks) < 20:
    print("Repetition: moderate")
else:
    print("Repetition: high")

# -------------------
# DOMINANT FREQUENCY
# -------------------
S = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)

avg_spectrum = np.mean(S, axis=1)
dominant_freq = freqs[np.argmax(avg_spectrum)]

print(f"\nDominant frequency: {dominant_freq:.1f} Hz")

# -------------------
# BRIGHTNESS
# -------------------
centroid = librosa.feature.spectral_centroid(
    y=y,
    sr=sr
)[0]

print(f"Average brightness: {np.mean(centroid):.1f} Hz")

# -------------------
# NOTE TYPES
# -------------------
if len(pitch):

    low_cut = np.percentile(pitch, 33)
    high_cut = np.percentile(pitch, 66)

    low_notes = np.sum(pitch < low_cut)
    mid_notes = np.sum((pitch >= low_cut) & (pitch <= high_cut))
    high_notes = np.sum(pitch > high_cut)

    print("\nNote distribution:")
    print(f"Low notes: {low_notes}")
    print(f"Mid notes: {mid_notes}")
    print(f"High notes: {high_notes}")

# -------------------
# SIMPLE COMMUNICATION GUESS
# -------------------
print("\nPossible function:")

if len(peaks) > 20 and dominant_freq > 3000:
    print("May be territorial or mating song")
elif len(peaks) < 5 and np.max(energy) > np.mean(energy) * 2:
    print("May be alarm/warning calls")
else:
    print("May be social communication")

# -------------------
# SPECTROGRAM
# -------------------
D = librosa.amplitude_to_db(
    np.abs(librosa.stft(y)),
    ref=np.max
)

plt.figure(figsize=(12, 6))

librosa.display.specshow(
    D,
    sr=sr,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.tight_layout()

plt.show()
