from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import pandas as pd
import joblib
from scipy.signal import hilbert
import os
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
app = Flask(__name__)

# Load the SVM model & Scaler
model = joblib.load('Egg_ensemble.sav')
scaler = joblib.load('eggscaler.sav')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

DATADIR = "dataset"
CATEGORIES = os.listdir(DATADIR)
# Feature extraction function
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  
    noise_ratio = np.mean(librosa.effects.preemphasis(y))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y, sr=sr))
    mfcc_mean = np.mean(librosa.feature.mfcc(y, sr=sr, n_mfcc=10))
    
    analytic_signal = hilbert(y)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    open_phases_duration = len(np.where(instantaneous_phase > 0)[0])
    closed_phases_duration = len(np.where(instantaneous_phase <= 0)[0])
    
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
        "spectral_contrast": spectral_contrast,
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
        "tonnetz": tonnetz

    }


def generate_gpt2_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = gpt2_model.generate(input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        file_path = "temp.wav"
        file.save(file_path)
        
        features = extract_features(file_path)
        features_df = pd.DataFrame([features])
        features_scaled = scaler.transform(features_df)
        predicted_class = model.predict(features_scaled)[0]
        class_label = CATEGORIES[int(predicted_class)]
        explanation = generate_gpt2_response(f"ELECTROENCEPHALOGRAM(EEG) sound data is classified as {class_label} disease,give me remedy")
        print(explanation)
        gy=class_label +" REMEDY:"+ explanation
        print(gy)
        return jsonify({"prediction": gy})

    return jsonify({"error": "No file uploaded"}), 400

if __name__ == '__main__':
    app.run(debug=True)