import librosa
import numpy as np
import librosa
import pandas as pd
import joblib
from scipy.signal import hilbert
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tkinter as tk
from tkinter import filedialog
# Load the SVM model
model = joblib.load('Egg_ensemble.sav')

# Load the scaler
scaler = joblib.load('eggscaler.sav')

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

def predict_class(audio_file):
    features = extract_features(audio_file)

    # Create a DataFrame from the extracted features
    features_df = pd.DataFrame([features])

    # Standardize the features using the loaded scaler
    features_scaled = scaler.transform(features_df)

    # Make predictions
    predicted_class = model.predict(features_scaled)[0]
    return predicted_class

def choose_audio_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    if file_path:
        return file_path
    else:
        return None


def main():
    audio_file_path = choose_audio_file()
    DATADIR = "dataset"
    CATEGORIES = os.listdir(DATADIR)

    if audio_file_path:
        predicted_class = predict_class(audio_file_path)
        class_label = CATEGORIES[int(predicted_class)]
        print(f'Predicted class: {class_label}')
        

if __name__ == '__main__':
    main()
