import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(filename):
    # Load the text vectors and labels from the CSV file
    filename = filename + ".csv"
    column_name = "comment"  
    label_column = "label"

    df = pd.read_csv(filename, sep = '|', header = 0)
    
    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()

    return text, np.array(labels)

def load_word_vectors(filename):
    return KeyedVectors.load(filename)

def texts_to_sequences(texts, word_vectors, max_len):
    sequences = []
    for text in texts:
        words = text.split()  # Assuming texts are preprocessed and tokenized
        seq = [word_vectors[word] for word in words if word in word_vectors]
        sequences.append(seq)
    # Pad sequences to ensure consistent input shape
    sequences_padded = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0)
    return np.array(sequences_padded)

def build_model(input_shape, embedding_dim = 200, output_size = 20):
    model = Sequential()
    model.add(Bidirectional(LSTM(output_size, return_sequences=True, input_shape=input_shape)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Bidirectional(LSTM(output_size)))
    model.add(Dense(40, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    dataset_filename = str(input("Dataset filename: "))
    word_vectors_filename = str(input("word vectors (kv) file: "))
    # Load the data
    texts, labels = load_data(dataset_filename)

    # Load the word vectors
    word_vectors = load_word_vectors(word_vectors_filename)

    # Convert texts to sequences of word vectors
    max_len = max(len(text.split()) for text in texts)  # Find the max length of texts
    text_sequences = texts_to_sequences(texts, word_vectors, max_len)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(text_sequences, labels, test_size=0.2, random_state=42)

    # Get the input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.2)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')

    # Save the model
    model.save("bidirectional_lstm_model.keras")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
