# Transformer Music Genre Classification

This repository contains the Transformer implementation for a four-model comparative deep learning project evaluating MLP, 1D CNN, LSTM, and Transformer architectures on music genre classification. The Transformer model processes sequential Mel-Frequency Cepstral Coefficients (MFCCs) using self-attention mechanisms to learn relationships between audio frames.

## Dataset

The project uses the GTZAN Music Genre Dataset containing 1,000 approximately 30-second audio tracks across 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Audio files that cannot be processed adequately are skipped during feature extraction.

## Preprocessing

- Audio is divided into 3-second non-overlapping segments sampled at 22,050 Hz.
- 40 MFCC features are extracted with `n_fft = 2048` and `hop_length = 512`, producing an input shape of `(130, 40)` per segment.
- A song-level stratified split (70% train, 15% validation, 15% test) prevents segments from the same song appearing across different subsets.
- Features are standardized using `StandardScaler` fitted exclusively on the training set.

## Model

The model is a Transformer encoder network:

- **Input:** `(130, 40)`
- **Dense Projection Layer:** Maps the 40 MFCC features to a 128-dimensional representation.
- **Positional Embedding:** Adds learnable positional information to preserve the order of audio frames.
- **Transformer Encoder Block 1:** Four-head self-attention, feed-forward network, residual connections, layer normalization, and dropout.
- **Transformer Encoder Block 2:** Further learns temporal relationships and refines the sequence representation.
- **Global Average Pooling:** Aggregates information across the temporal dimension.
- **Dense Layer:** 128 units with ReLU activation.
- **Dropout:** Rate of 0.2 for regularization.
- **Output Layer:** 10 units with Softmax activation for genre classification.

**Model Parameters:**

- Transformer encoder blocks: 2
- Attention heads per block: 4
- Model dimension: 128
- Feed-forward dimension: 256
- Dropout rate: 0.2
- Total parameters: As reported by `transformer_model.summary()`.

## Training

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Maximum Epochs:** 30
- **Early Stopping:** Monitors `val_loss` with `patience = 5` and `restore_best_weights = True`.
- **Learning Rate Scheduling:** `ReduceLROnPlateau` reduces the learning rate when validation loss stops improving.

## Results

| **Metric** | **Score** |
|---|---:|
| Test Loss | 1.2943 |
| Accuracy | 61.73% |
| Precision (Weighted) | 63.61% |
| Recall (Weighted) | 61.73% |
| F1-Score (Weighted) | 62.01% |


## Files

- `transformer_genre_classification.ipynb`: Complete notebook containing dataset preparation, MFCC extraction, model definition, training, evaluation, and visualizations.
- `TRANSFORMER_METHODOLOGY.md`: Detailed methodology and literature notes for the Transformer model.
- `results/transformer_training_curves.png`: Training and validation accuracy and loss plots.
- `results/transformer_confusion_matrix.png`: Confusion matrix on the held-out test set.
- `results/transformer_metrics.csv`: Exported test evaluation metrics.
- `results/transformer_gtzan.keras`: Saved trained Transformer model.

## References

1. Alvaro Harryanto, Andreas Aditya, Kevin Gunawan, Rio Nagano, and Rhio Sutoyo (2022), *"Music Classification Model Development Based on Audio Recognition using Transformer Model,"* 2022 3rd International Conference on Artificial Intelligence and Data Sciences (AiDAS). (https://doi.org/10.1109/AiDAS56890.2022.9918787).

2. Greer, Timothy, Xuan Shi, Benjamin Ma, and Shrikanth Narayanan (2023), *"Creating Musical Features Using Multi-Faceted, Multi-Task Encoders Based on Transformers,"* *Scientific Reports*, Volume 13, Article 10713. (https://www.nature.com/articles/s41598-023-36714-z).