#!/usr/bin/env python
# coding: utf-8

# ## GUI CREATION FOR MUSIC GENRE CLASSIFICATION

# ### IMPORTING MODULES

# In[5]:


import tkinter as tk
from tkinter import filedialog, Label
import numpy as np
import librosa
import joblib


# ### Load the trained model

# In[8]:


model_path = r"C:\Users\ravis\Downloads\MUSIC_GENRE_CLASSIFICATION_XGBOOST\MODELS\music_genre_xgb_model.pkl"
model = joblib.load(model_path)


# ### Genre labels mapping

# In[11]:


genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_mapping = {i: genre for i, genre in enumerate(genres)}


# ### FEATURES EXTRACTION

# In[14]:


# Extracting features from a single audio segment
def extract_features_from_segment(y, sr):
    features = {
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_std': np.std(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rmse_mean': np.mean(librosa.feature.rms(y=y)),
        'rmse_std': np.std(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_std': np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_bandwidth_std': np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_std': np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
        'zero_crossing_rate_std': np.std(librosa.feature.zero_crossing_rate(y)),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
    }
    # Extracting MFCCs and adding means and stds
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_std'] = np.std(mfccs[i-1])

    return np.array([list(features.values())])  # Return as array for model input


# ### PREDICTION

# In[17]:


# Predict genre based on each 30-second segment and aggregate results
def predict_genre(file_path):
    y, sr = librosa.load(file_path)  # Load entire audio file
    segment_length = 30  # seconds
    num_segments = len(y) // (segment_length * sr)
    
    genre_votes = []

    # Process each 30-second segment
    for i in range(num_segments):
        start_sample = i * segment_length * sr
        end_sample = start_sample + segment_length * sr
        segment_y = y[start_sample:end_sample]

        # Extract features and predict genre for each segment
        features = extract_features_from_segment(segment_y, sr)
        genre_index = model.predict(features)[0]
        genre_votes.append(genre_index)
    
    # Predict remaining part if any (less than 30 seconds)
    if len(y) % (segment_length * sr) != 0:
        remaining_y = y[num_segments * segment_length * sr:]
        features = extract_features_from_segment(remaining_y, sr)
        genre_index = model.predict(features)[0]
        genre_votes.append(genre_index)
    
    # Aggregate predictions - majority voting
    final_genre_index = max(set(genre_votes), key=genre_votes.count)
    final_genre = genre_mapping.get(final_genre_index, "Unknown Genre")
    return final_genre


# ### File selection and prediction

# In[20]:


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        genre = predict_genre(file_path)
        result_label.config(text=f"Predicted Genre: {genre}")


# ### GUI setup

# In[23]:


root = tk.Tk()
root.title("Music Genre Classification")
root.geometry("400x300")

# GUI Elements
Label(root, text="Music Genre Classification", font=("Arial", 18)).pack(pady=20)
Label(root, text="Choose an audio file for genre prediction:").pack(pady=10)
tk.Button(root, text="Select Audio File", command=open_file).pack(pady=10)
result_label = Label(root, text="Predicted Genre:", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()


# In[ ]:




