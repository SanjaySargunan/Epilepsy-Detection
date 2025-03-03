import librosa
import pandas as pd
import numpy as np
from scipy.signal import hilbert
import os

# Function to extract features from a WAV file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Load audio with the original sample rate
    noise_ratio = np.mean(librosa.effects.preemphasis(y))
    spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=10)
    mfcc_mean = np.mean(mfcc)
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    open_phases_duration = len(np.where(instantaneous_phase > 0)[0])  # Count positive phases
    closed_phases_duration = len(np.where(instantaneous_phase <= 0)[0])  # Count non-positive phases

    open_quotient = open_phases_duration / (open_phases_duration + closed_phases_duration)
    closed_quotient = 1 - open_quotient
    
    zero_crossings = sum(librosa.zero_crossings(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    
    # Additional features
    chroma_stft = np.mean(librosa.feature.chroma_stft(y, sr=sr))
    chroma_cqt = np.mean(librosa.feature.chroma_cqt(y, sr=sr))
    tonnetz = np.mean(librosa.feature.tonnetz(y))

    return {
        "noise_ratio": noise_ratio,
        "spectral_contrast_mean": spectral_contrast_mean,
        "mfcc_mean": mfcc_mean,
        "open_quotient": open_quotient,
        "closed_quotient": closed_quotient,
        "zero_crossings": zero_crossings,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
        "rolloff": rolloff,
        "tempo": tempo,
        "chroma_stft": chroma_stft,
        "chroma_cqt": chroma_cqt,
        "tonnetz": tonnetz,
    }

# Specify the directory where your audio files are located
audio_directory = "dataset"

data = []

for label, subfolder in enumerate(os.listdir(audio_directory)):
    subfolder_path = os.path.join(audio_directory, subfolder)
    
    for audio_file in os.listdir(subfolder_path):
        if audio_file.endswith(".wav"):
            audio_file_path = os.path.join(subfolder_path, audio_file)
            
            features = extract_features(audio_file_path)
            
            # Add the label to the features
            features["label"] = label
            
            # Append the data to the list
            data.append(features)

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("audio_features.csv", index=False)
