import pandas as pd
import numpy as np
import sys
import csv
import fasttext
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Flatten, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, texts, labels, word_vectors, batch_size=128, max_len=100, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.texts = texts
        self.labels = labels
        self.word_vectors = word_vectors
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        texts_temp = [self.texts[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        X = self.__data_generation(texts_temp)
        y = np.array(labels_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.texts))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, texts_temp):
        sequences = []
        for text in texts_temp:
            words = text.split()
            seq = [self.word_vectors.get_word_vector(word) for word in words if word in self.word_vectors]
            sequences.append(seq)
        sequences_padded = pad_sequences(sequences, maxlen=self.max_len, dtype='float32', padding='post', truncating='post', value=0.0)
        return np.array(sequences_padded)

def load_data(filename):
    column_name = "translated"
    label_column = "label"
    
    try:
        df = pd.read_csv(filename, sep='|', header=0, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filename, sep='|', header=0, encoding='utf-16')

    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()
    return texts, np.array(labels)

def load_word_vectors(filename):
    word_vectors = fasttext.load_model(filename)
    return word_vectors

def replace_embedding_layer(model, word_vectors, max_len):
    vocab_size = len(word_vectors.words)  # FastText vocab size
    embedding_dim = word_vectors.get_dimension()  # FastText embedding dimension
    
    # Create a new embedding layer using FastText vectors
    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dim, 
                                input_length=max_len, 
                                trainable=False)
    
    # Replace the old embedding layer
    model.layers[0] = embedding_layer
    return model

def modify_output_layer(model, num_classes):
    # Remove the original output layer
    model.layers.pop()
    
    # Add a new output layer with a unique name
    output_layer = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid', name='new_dense_output')
    model.add(output_layer)
    
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy', 
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 transfer_learning.py pretrained_model.keras translated_dataset.csv word_vectors.bin")

    model_filename = sys.argv[1]
    dataset_filename = sys.argv[2]
    word_vectors_filename = sys.argv[3]
    texts, labels = load_data(dataset_filename)
    word_vectors = load_word_vectors(word_vectors_filename)

    texts = [text if isinstance(text, str) else "" for text in texts]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    max_len = 200  # Same max_len as before

    train_generator = DataGenerator(X_train, y_train, word_vectors, max_len=max_len)
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=max_len, shuffle=False)

    # Load your existing model
    model = load_model(model_filename)

    # Replace the embedding layer with FastText embeddings
    model = replace_embedding_layer(model, word_vectors, max_len)

    # Modify the output layer to match the number of classes in the new dataset
    num_classes = len(set(labels))
    model = modify_output_layer(model, num_classes)

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)

    # Train the model with the new data
    model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {accuracy}')

    # Save the updated model
    model.save("transfer_learning_model.keras")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
