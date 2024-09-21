import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from Bi_LSTM_v5 import DataGenerator

# Load the dataset
def load_data(filename):
    df = pd.read_csv(filename, sep='|', header=0)
    texts = df['comment'].tolist()
    labels = df['label'].tolist()
    return texts, np.array(labels)

# Load the word vectors (word2vec for English, fasttext for French)
def load_word_vectors_en(filename):
    return KeyedVectors.load_word2vec_format(filename, binary=True)

def load_word_vectors_fr(filename):
    return load_and_reduce_word_vectors(filename)

# Evaluation function
def evaluate_model(model_path, dataset_filename, word_vectors_filename, form, max_len=300):
    # Load the data
    texts, labels = load_data(dataset_filename)

    # Load word vectors (ensure binary=True for FastText .bin format)
    if form == "en":
        word_vectors = load_word_vectors_en(word_vectors_filename)
    else:
        word_vectors = load_word_vectors_fr(word_vectors_filename)
    # Split the data into training and test sets
    texts = [text if isinstance(text, str) else "" for text in texts]
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Prepare the test generator
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=max_len, shuffle=False)

    # Load the saved model
    model = load_model(model_path)

    # Predict on the test set
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Extract actual test labels
    y_test_actual = []
    for _, y in test_generator:
        y_test_actual.extend(y)

    y_test_actual = np.array(y_test_actual)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_actual, y_pred)
    precision = precision_score(y_test_actual, y_pred)
    recall = recall_score(y_test_actual, y_pred)
    f1 = f1_score(y_test_actual, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


def main():
    # English word2vec evaluation
    model_path = "bi_LSTM_model_v5_pretrained.keras"  
    dataset_filename = "reduced_1000.csv"  
    word_vectors_filename = "google_word2vec.bin"  

    # Evaluate the English model
    evaluate_model(model_path, dataset_filename, word_vectors_filename, "en")

    # French fasttext evaluation
    model_path_fr = "transfer_learning_model_v0.keras"  
    dataset_filename_fr = "translated_r2500.csv"  
    word_vectors_filename_fr = "fasttext_fr.bin"  

    # Evaluate the French model
    evaluate_model(model_path_fr, dataset_filename_fr, word_vectors_filename_fr, "fr")

if __name__ == "__main__":
    main()
