# Transformer Methodology

### 1. Dataset and Preprocessing
We use the GTZAN dataset consisting of 1,000 audio tracks (approximately 30 seconds each) across 10 musical genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Audio files are processed to extract three-second, non-overlapping segments at a sampling rate of 22,050 Hz. Corrupted or incomplete segments are skipped during preprocessing to ensure that only usable audio data is included in the experiment.

### 2. MFCC Representation
For each three-second audio segment, we extract 40 Mel-Frequency Cepstral Coefficients (MFCCs) using an FFT window size of 2,048 samples and a hop length of 512 samples. Each segment is represented as a two-dimensional matrix of shape (40, 130), which is transposed to (130, 40) so that the Transformer can process the audio as a sequence of 130 time frames, each containing 40 acoustic features. The complete dataset is represented as a three-dimensional tensor of shape (samples, 130, 40).

### 3. Data Splitting
To prevent data leakage, the dataset is split at the original-song level, ensuring that segments from the same song appear in only one subset. The data is divided into 70% training, 15% validation, and 15% testing subsets using stratified sampling and a fixed random seed (random_state=42). The validation set is used to monitor model training, while the test set is reserved for final performance evaluation. The actual number of usable songs and segments depends on the audio files successfully processed.

### 4. Feature Scaling
Feature scaling is performed using StandardScaler to standardize the 40 MFCC coefficients. The three-dimensional training tensor is reshaped into a two-dimensional array of shape (samples × 130, 40) before fitting the scaler. The fitted scaler is then applied to the validation and test sets, after which all datasets are reshaped back to (samples, 130, 40). This maintains the temporal sequence structure while preventing information from the validation and test sets from influencing the scaling parameters.

### 5. Transformer Architecture
The model is implemented using TensorFlow/Keras as a Transformer encoder network designed to learn temporal relationships between MFCC frames.
- Input Layer: Accepts sequences of shape (130, 40).
- Dense Projection Layer: Maps the 40 MFCC features into a 128-dimensional representation.
- Positional Embedding Layer: Adds learnable positional information to preserve the order of audio frames.
- Transformer Encoder Block 1: Uses four-head self-attention to capture relationships between different time frames, followed by a feed-forward network, residual connections, layer normalization, and dropout.
- Transformer Encoder Block 2: Further refines the learned temporal representations through another self-attention and feed-forward block.
- Global Average Pooling Layer: Aggregates information across the time dimension into a fixed-length feature vector.
- Dense Layer: Contains 128 units with ReLU activation to learn higher-level genre-discriminative features.
- Dropout Layers: Use a dropout rate of 0.2 to reduce overfitting.
- Output Layer: Contains 10 units with Softmax activation to predict the probability of each music genre.
Model Parameters:
- Number of encoder blocks: 2
- Number of attention heads per block: 4
- Model dimension: 128
- Feed-forward dimension: 256
- Dropout rate: 0.2
- Total and trainable parameters: Obtain these from transformer_model.summary() rather than estimating them.

### 6. Training Configuration
The Transformer is trained using the Adam optimizer with a learning rate of 0.001 and sparse categorical cross-entropy as the loss function. The batch size is set to 32, with a maximum of 30 epochs. Early stopping monitors validation loss with a patience of five epochs and restores the best-performing weights. A ReduceLROnPlateau callback reduces the learning rate when validation loss stops improving, helping the model refine its parameters during training.

### 7. Evaluation Metrics
The model is evaluated on the held-out test set of 1,500 segments using accuracy, weighted precision, weighted recall, and weighted F1-score.
- Accuracy: 61.73% (0.6173)
- Precision (weighted): 63.61% (0.6361)
- Recall (weighted): 61.73% (0.6173)
- F1-score (weighted): 62.01% (0.6201)
- Test Loss: 1.2943
These results indicate moderate classification performance, with differences in predictive reliability across the ten music genres.

### 8. Brief Error Analysis
Based on the test confusion matrix and classification report:
- Best Performing Genres: Classical achieves the highest recall (148/150 segments, 98.7%) and an F1-score of 0.90. Metal achieves the highest precision (0.89), while Jazz has a precision of 0.86 and an F1-score of 0.73.
- Main Confusions:
  - Rock: Has the lowest F1-score (0.40), with 66/150 segments correctly classified (44.0% recall). It is frequently confused with Country (25 segments), Reggae (18), and Hiphop (11).
  - Disco: Achieves 45.3% recall and is often misclassified as Pop (25 segments), Hiphop (14), and Classical (11).
  - Blues: Achieves 47.3% recall and is frequently confused with Country (30 segments) and Hiphop (18).
  - Country: Achieves 61.3% recall and is most frequently confused with Rock (31 segments).
  - Hiphop and Reggae: Achieve recalls of 64.0% and 62.0%, respectively, with confusion between these genres and other rhythmically similar styles.
Overall, the Transformer performs well on Classical music but struggles to distinguish several genres with overlapping instrumentation and rhythmic characteristics.

### Literature Notes

#### Paper 1
- Citation: Andreas Aditya Alvaro Harryanto, Kevin Gunawan, Rio Nagano, and Rhio Sutoyo (2022), “Music Classification Model Development Based on Audio Recognition using Transformer Model,” 2022 3rd International Conference on Artificial Intelligence and Data Sciences (AiDAS).
- Problem: Classifying music genres from audio signals by learning relevant acoustic features and relationships between audio frames.
- Dataset: GTZAN dataset containing 1,000 audio tracks across ten music genres.
- Transformer Method: Extracts MFCC features from three-second audio segments and applies a Transformer encoder with multi-head self-attention, positional encoding, and feed-forward layers. The study also investigates data augmentation and hyperparameter tuning.
- Main Result: The best Transformer configuration achieved 75.1% accuracy, outperforming the other configurations and the CNN and ANN baselines evaluated in the study. Time-shift augmentation also improved performance compared with the original dataset in the reported experiment. 
- Relevance: This paper directly supports our approach of applying Transformer encoders to MFCC-based GTZAN audio segments and demonstrates the importance of model configuration and preprocessing in music genre classification.

#### Paper 2
- Citation: Timothy Greer, Xuan Shi, Benjamin Ma, and Shrikanth Narayanan (2023), “Creating Musical Features Using Multi-Faceted, Multi-Task Encoders Based on Transformers,” Scientific Reports, Volume 13, Article 10713.
- Problem: Developing general-purpose music representations that can support multiple music understanding tasks, including genre classification, without relying exclusively on task-specific supervised learning.
- Dataset: The study uses multiple music datasets, including GTZAN for genre classification, alongside datasets for music tagging, instrument classification, and emotion recognition.
- Transformer Method: Proposes M3BERT, a music representation model based on bidirectional Transformers and self-attention. It uses self-supervised masked reconstruction during pretraining and subsequently fine-tunes the learned representations on multiple music-related tasks.
- Main Result: The learned representations outperform several comparison audio and music embeddings on multiple downstream tasks. However, the authors report that the benefits for GTZAN genre classification are more limited, partly because the dataset is relatively small and multi-task training can favor larger datasets. 
- Relevance: This study supports the use of Transformer-based attention mechanisms for learning musical representations while highlighting the challenges of limited training data. It provides context for interpreting our model's performance and suggests that larger datasets, suitable audio features, and careful training strategies may improve genre classification.