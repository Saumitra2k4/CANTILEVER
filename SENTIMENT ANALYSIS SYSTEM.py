import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the IMDB dataset
num_words = 5000
max_len = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Convert integer sequences back to text
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([index_word.get(i, '?') for i in encoded_review])

train_reviews = [decode_review(review) for review in X_train]
test_reviews = [decode_review(review) for review in X_test]

train_df = pd.DataFrame({'review': train_reviews, 'sentiment': y_train})
test_df = pd.DataFrame({'review': test_reviews, 'sentiment': y_test})

# Combine train and test data
df = pd.concat([train_df, test_df], ignore_index=True)

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(preprocess_text)
corpus = df['cleaned_review'].tolist()
y = df['sentiment'].values

# TF-IDF Feature Extraction for SVM
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()

# Tokenization and Padding for RNN
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(corpus)
X_tokenized = tokenizer.texts_to_sequences(corpus)
X_padded = pad_sequences(X_tokenized, maxlen=max_len)

# Splitting the data
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_train_pad, X_test_pad, _, _ = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# SVM Model Training
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# RNN Model Training
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_len))
rnn_model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
rnn_model.add(Dense(1, activation='sigmoid'))

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate SVM Model
y_pred_tfidf = svm_model.predict(X_test_tfidf)
print("SVM Model with TF-IDF Features")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf))

# Evaluate RNN Model
loss, accuracy = rnn_model.evaluate(X_test_pad, y_test)
print("RNN Model with Tokenized and Padded Features")
print(f"Accuracy: {accuracy}")

# Predict Sentiment of New User Input Reviews
def predict_sentiment(review, model_type='svm'):
    cleaned_review = preprocess_text(review)
    
    if model_type == 'svm':
        features = tfidf_vectorizer.transform([cleaned_review]).toarray()
        prediction = svm_model.predict(features)
    else:
        tokenized_review = tokenizer.texts_to_sequences([cleaned_review])
        padded_review = pad_sequences(tokenized_review, maxlen=max_len)
        prediction = rnn_model.predict(padded_review)
        prediction = (prediction > 0.5).astype("int32")
    
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment

print("Enter movie reviews to predict their sentiment. Type 'done' when finished.")
while True:
    user_review = input("Enter a movie review: ")
    if user_review.lower() == 'done':
        break
    sentiment_svm = predict_sentiment(user_review, model_type='svm')
    sentiment_rnn = predict_sentiment(user_review, model_type='rnn')
    print(f"SVM Prediction: {sentiment_svm}")
    print(f"RNN Prediction: {sentiment_rnn}")
