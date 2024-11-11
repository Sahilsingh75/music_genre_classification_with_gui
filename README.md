# music_genre_classification with gui

## Project Overview
This project is a music genre classification tool that leverages an **XGBoost** model trained on the **GTZAN dataset**. The model uses extracted audio features (e.g., chroma, spectral features, MFCCs) to classify audio into one of 10 genres: *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock*.

### Genres Supported
The model can classify audio into the following genres:
- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Features
- **Genre Prediction**: Classifies uploaded audio files into one of 10 genres.
- **GUI**: A user-friendly interface built with Tkinter that allows users to upload and classify audio files.
- **Pre-trained XGBoost Model**: Uses XGBoost for high-accuracy genre classification based on extracted features.

## Project Structure
music_genre_classification_with_gui/ 
├── DATA/ # Placeholder folder for the dataset (do NOT include large files here)
├── LICENSE # Project license file 
├── README.md # Project documentation (this file) 
├── requirements.txt # Required Python packages and versions 
└── SCRIPTS/ # Source code for training and GUI 
├── NOTEBOOKS/ # Jupyter notebooks for model training and gui creation
├── MODELS/ #saved model

## Getting Started

### Prerequisites
To run this project, you need:
- Python 3.7 or higher
- [TensorFlow](https://www.tensorflow.org/) for model training and prediction
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI (usually comes with Python)
- Other dependencies listed in `requirements.txt`

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sahilsingh75/music_genre_classification_with_gui.git
   cd music_genre_classification_with_gui
2. **Install Dependencies Install the required packages using pip:**
   ```bash
   pip install -r requirements.txt
### Dataset
Due to its large size, the dataset is not included in this repository. You can download the GTZAN dataset from Google Drive (https://drive.google.com/drive/folders/1Vgu541Yy2DbLgjC5T1bwKBC1sZRmfapl?usp=sharing) or Kaggle (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place it in the DATA/ folder.

For detailed instructions, refer to DATASET_DOWNLOAD.txt.

### Model Training
The model was trained using an XGBoost classifier on extracted audio features:

Features: Chroma, spectral centroid, bandwidth, rolloff, zero-crossing rate, tempo, and MFCCs.
Labels: Encoded genre labels based on the GTZAN dataset.
Accuracy: The model achieved a validation accuracy of approximately X% (adjust based on actual results).
For training details, refer to the notebooks.

### License
This project is licensed under the MIT License.
