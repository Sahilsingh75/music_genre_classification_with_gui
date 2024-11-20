#!/usr/bin/env python
# coding: utf-8

# ### MUSIC GENRE CLASSIFICATION

# In[1]:


import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[2]:


# Path to the audio files
audio_path = r"C:\Users\ravis\Downloads\gtzan\Data\genres_original"


# ### GENRE LABELLING

# In[3]:


genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


# ### FEATURES EXTRACTION

# In[7]:


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=30)
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
        'tempo': librosa.beat.tempo(y=y, sr=sr)  # Remove [0]
    }
    
    # Extracting MFCCs and taking mean and standard deviation
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_std'] = np.std(mfccs[i-1])
        
    return features


# In[9]:


# Extract features and labels
features_list = []
labels = []

for genre in genres:
    genre_path = os.path.join(audio_path, genre)
    for file_name in os.listdir(genre_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(genre_path, file_name)
            features = extract_features(file_path)
            features_list.append(features)
            labels.append(genre)


# ### Convert features and labels to DataFrame and encode labels

# In[11]:


features_df = pd.DataFrame(features_list)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)


# ### Split data

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(features_df, labels_encoded, test_size=0.2, random_state=42)


# ### Convert 'tempo' and any other object columns to numeric or category as needed

# In[21]:


for column in X_train.select_dtypes(include=['object']).columns:
    try:
        # Attempt to convert to numeric if the data is numeric
        X_train[column] = pd.to_numeric(X_train[column], errors='coerce')
        X_test[column] = pd.to_numeric(X_test[column], errors='coerce')
    except ValueError:
        # If conversion fails, convert to 'category'
        X_train[column] = X_train[column].astype('category')
        X_test[column] = X_test[column].astype('category')


# ### Train the XGBoost classifier

# In[23]:


xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)


# ### Evaluate the model

# In[25]:


y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")


# ### Save the model

# In[27]:


import joblib
joblib.dump(xgb_model, r'C:\Users\ravis\Downloads\gtzan\music_genre_xgb_model.pkl')

