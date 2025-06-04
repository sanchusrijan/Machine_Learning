import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('urban_app.h5')

# Function to extract features from uploaded audio
def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=42)
    mfccs_mean = mfccs.mean(axis=1)
    # Add any other features if your model expects
    return mfccs_mean.reshape(1, -1)

# Streamlit UI
st.title("Urban Sound Classification")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio.wav")

    # Extract features and predict
    features = extract_features("temp_audio.wav")
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map classID to class names (use your actual mapping)
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer",
        "siren", "street_music"
    ]

    st.write(f"Predicted Sound Class: **{class_names[predicted_class]}**")