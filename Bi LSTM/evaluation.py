import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_data(filename, vectors_name):
    # Load the text vectors and labels from the CSV file
    filename = filename + ".csv"
    column_name = "comment"  
    label_column = "label"

    df = pd.read_csv(filename, sep='|', header=0)
    
    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()

    # Load the preprocessed text vectors
    text_vectors = np.load(vectors_name)

    return text_vectors, np.array(labels)

def main():
    filename = str(input("Dataset filename: "))
    vectors_name = str(input("Text vectors (npy) file: "))
    model_path = str(input("Model file: "))

    # Load the data
    text_vectors, labels = load_data(filename, vectors_name)

    # Split the data into training and test sets
    _, X_test, _, y_test = train_test_split(text_vectors, labels, test_size=0.2, random_state=42)

    # Reshape data to match LSTM input requirements
    X_test = np.expand_dims(X_test, axis=2)

    # Load the saved model
    model = load_model(model_path)

    # Predict on the test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == "__main__":
    main()
