import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.optimizers import RMSprop

def load_data(filename, vectors_name):
    # Load the text vectors and labels from the CSV file
    filename = filename + ".csv"
    column_name = "comment"  
    label_column = "label"

    df = pd.read_csv(filename, sep = '|', header = 0)
    
    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()

    # Load the preprocessed text vectors
    text_vectors = np.load(vectors_name)

    return text_vectors, np.array(labels)

def build_model(input_shape, embedding_dim = 200, output_size = 20):
    model = Sequential()
    model.add(Bidirectional(LSTM(output_size, input_shape = input_shape)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = "ReLU"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    filename = str(input("Dataset filename: "))
    vectors_name = str(input("Text vectors (npy) file: "))
    # Load the data
    text_vectors, labels = load_data(filename, vectors_name)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(text_vectors, labels, test_size=0.2, random_state=42)

    # Get the input shape for the model
    input_shape = (X_train.shape[1], 1)

    # Reshape data to match LSTM input requirements
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build the model
    model = build_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')

    # Save the model
    model.save("bidirectional_lstm_model.h5")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
