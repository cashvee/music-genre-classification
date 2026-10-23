# MLP-Based Music Genre Classification

## Overview
This project implements a Multi-Layer Perceptron (MLP) neural network to classify music into 10 genres using the GTZAN Music Genre Dataset.

The model extracts numerical audio features and learns patterns that help distinguish different music genres.

## Dataset
- **Dataset:** GTZAN Music Genre Dataset
- **Genres:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock
- **Successfully processed audio files:** 999

## Feature Extraction
A total of 56 features are extracted from each audio file:

- **MFCC (26 features):** 13 mean values and 13 standard deviations to represent audio timbre.
- **Chroma (24 features):** 12 mean values and 12 standard deviations to represent pitch-class information.
- **Spectral features (6 features):** Mean and standard deviation of spectral centroid, spectral bandwidth, and spectral rolloff.

The extracted features are standardized using `StandardScaler`.

## Model Architecture
The MLP model consists of:

1. Input layer with 56 features
2. Dense layer with 128 neurons and ReLU activation
3. Dropout layer with a rate of 0.3
4. Dense layer with 64 neurons and ReLU activation
5. Dropout layer with a rate of 0.3
6. Output layer with 10 neurons and Softmax activation

**Total trainable parameters:** 16,202

## Training
- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy
- **Batch size:** 32
- **Maximum epochs:** 100
- **Early stopping:** Used to monitor validation loss and restore the best model weights
- **Data split:** 70% training, 15% validation, and 15% testing (approximately)

The data was split before scaling, and the scaler was fitted only on the training data to prevent data leakage.

## Results

| Metric | Result |
|---|---:|
| Test accuracy | 68.67% |
| Test loss | 1.0403 |
| Macro-average precision | 0.70 |
| Macro-average recall | 0.69 |
| Macro-average F1-score | 0.69 |

The model performed particularly well on Classical and Jazz genres. Performance varied across genres, indicating opportunities for further improvement.

## Evaluation
The model was evaluated using:
- Test accuracy and loss
- Precision, recall, and F1-score for each genre
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

## Technologies Used
- Python
- TensorFlow / Keras
- Librosa
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn
- Google Colab

## Files
- `mlp_music_genre.ipynb` — notebook containing feature extraction, model building, training, and evaluation.
- `mlp_music_genre_model.keras` — saved trained model.
- `mlp_feature_scaler.pkl` — saved feature scaler.
- `mlp_label_encoder.pkl` — saved genre label encoder.

## Future Improvements
- Experiment with additional audio features and hyperparameters.
- Improve performance on genres that are harder to distinguish.
- Compare the MLP results with CNN, LSTM, and Transformer architectures.

## Conclusion
The MLP model provides a baseline for music genre classification using summary-based audio features. Its performance can be compared with other deep learning architectures as part of the overall project.

