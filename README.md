# IMDB Dataset for Sentiment Classification

## Overview
The IMDB dataset is a widely used dataset for binary sentiment classification. It consists of movie reviews from the Internet Movie Database (IMDB) and is often used to build and evaluate natural language processing (NLP) models. Each review is labeled as either **positive** or **negative**, making it ideal for sentiment analysis tasks.

## Dataset Details

- **Source**: [IMDB Movie Reviews][(https://ai.stanford.edu/~amaas/data/sentiment/](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Task**: Binary sentiment classification (positive or negative sentiment)
- **Dataset Size**:
  - Training Set: 25,000 labeled reviews (50% positive, 50% negative)
  - Test Set: 25,000 labeled reviews (50% positive, 50% negative)
  - Unlabeled Data: 50,000 reviews (for unsupervised learning tasks)

## File Structure


### 3. `unsup/`
Contains 50,000 unlabeled reviews, which can be used for pretraining or unsupervised learning tasks.

## Format

Each review is stored as a plain text file (.txt). The content of each file contains a single review. No preprocessing (e.g., tokenization, removal of stop words) has been applied to the text.

## Getting Started

### 1. Downloading the Dataset
You can download the dataset from the [official website][(https://ai.stanford.edu/~amaas/data/sentiment/](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


### 2. Dependencies
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow` or `pytorch`

Install the required libraries:
```bash
pip install numpy pandas scikit-learn tensorflow
```

### 3. Preprocessing Steps
1. Tokenize the text.
2. Convert text into numerical representations, e.g., **TF-IDF**, **word embeddings**, or **bag-of-words**.
3. Pad or truncate sequences to ensure uniform input length for models like RNNs.

Example:
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Tokenization and padding example
max_words = 10000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)

# Fit on training data
x_train = tokenizer.texts_to_sequences(train_texts)
x_train = pad_sequences(x_train, maxlen=max_len)
```

## Usage

### Training a Model
You can train a variety of models, such as:
- Logistic Regression
- Support Vector Machines
- Neural Networks (e.g., RNNs, LSTMs, Transformers)

Example using TensorFlow:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Build a simple LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=500),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)
```

### Evaluation
Evaluate your model on the test dataset to calculate metrics like accuracy, precision, recall, and F1-score.

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

## Applications
The IMDB dataset can be used for:
- Sentiment classification
- Transfer learning with embeddings (e.g., GloVe, Word2Vec)
- Experimenting with NLP techniques like **TF-IDF**, **skip-gram**, or **CBOW**


---

For any questions or feedback, feel free to contact [Saurav Sahay](mailto:sahatsrv@gmail.com).
