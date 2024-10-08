import numpy as np
import pandas as pd
import sys
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from Bi_LSTM import DataGenerator
from Transfer import load_and_reduce_word_vectors

def load_data(filename, form):
    if form == "en":
        df = pd.read_csv(filename, sep='|', header=0)
        texts = df['comment'].tolist()
    if form == "fr":
        try:
            df = pd.read_csv(filename, sep='|', header=0, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(filename, sep='|', header=0, encoding='utf-16')
        texts = df['translated'].tolist()
    labels = df['label'].tolist()
    return texts, np.array(labels)

# Load the word vectors (word2vec for English, fasttext for French)
def load_word_vectors_en(filename):
    return KeyedVectors.load_word2vec_format(filename, binary=True)

def load_word_vectors_fr(filename):
    return load_and_reduce_word_vectors(filename)


def evaluate_model(model_path, dataset_filename, word_vectors_filename, form, max_len=300):
    # Load the data
    texts, labels = load_data(dataset_filename, form)

    # Load word vectors
    if form == "en":
        word_vectors = load_word_vectors_en(word_vectors_filename)
    if form == "fr":
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
    batches = len(y_pred)

    # Extract actual test labels
    y_test_actual = []
    for i, (_, y) in enumerate(test_generator):
        y_test_actual.extend(y)
        if i > batches:
            break
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

    with open("test_logs.txt", 'a') as file:
        file.write(f"Model: {model_path}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall} \n")
        file.write(f"F1 Score: {f1}\n \n")
    

def main():
    
    if len(sys.argv) != 5:
        sys.exit("Usage: evaluation.py en OR fr dataset.csv model_name.keras word_vectors.bin")
    model_path = sys.argv[3]
    dataset_filename = sys.argv[2]
    word_vectors_filename = sys.argv[4]
    language = sys.argv[1]
    if language != "en" and language != "fr":
        sys.exit("Unknown language option entered")

    evaluate_model(model_path, dataset_filename, word_vectors_filename, language)

if __name__ == "__main__":
    main()
