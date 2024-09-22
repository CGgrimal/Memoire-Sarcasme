import numpy as np
import pandas as pd
import sys
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from Bi_LSTM_v5 import DataGenerator

def load_data(filename):
    column_name = "comment"
    label_column = "label"
    df = pd.read_csv(filename, sep='|', header=0)
    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()
    return texts, np.array(labels)

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 evaluation.py dataset.csv word_vectors.kv model.keras")
    filename = sys.argv[1]
    vectors_name = sys.argv[2]
    model_path = sys.argv[3]

    # Load the data
    text, labels = load_data(filename)
    word_vectors = KeyedVectors.load(filename)

    # Split the data into training and test sets
    texts = [text if isinstance(text, str) else "" for text in texts]
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Wrapping into generator form 
    #train_generator = DataGenerator(X_train, y_train, word_vectors, max_len=max_len)
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=200, shuffle=False)

    # Load the saved model
    model = load_model(model_path)

    # Predict on the test set
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int)
    test_actual = []

    for _, y in test_generator():
        test_actual.append(y)

    # Calculate precision, recall, and F1 score
    precision = precision_score(test_actual, y_pred)
    recall = recall_score(test_actual, y_pred)
    f1 = f1_score(test_actual, y_pred)
    #loss, accuracy = model.evaluate(test_generator)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == "__main__":
    main()
