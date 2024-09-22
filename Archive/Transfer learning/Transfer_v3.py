import pandas as pd
import numpy as np
import sys
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Flatten
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
            words = text.split()  # Assuming texts are preprocessed and tokenized
            seq = [self.word_vectors[word] for word in words if word in self.word_vectors]
            sequences.append(seq)
        sequences_padded = pad_sequences(sequences, maxlen=self.max_len, dtype='float32', padding='post', truncating='post', value=0.0)
        return np.array(sequences_padded)

def load_data(filename):
    column_name = "comment"
    label_column = "label"
    df = pd.read_csv(filename, sep='|', header=0)
    if column_name not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{column_name}' or '{label_column}' not found in CSV.")
    texts = df[column_name].tolist()
    labels = df[label_column].tolist()
    return texts, np.array(labels)

def load_word_vectors(filename):
    return KeyedVectors.load(filename)

def modify_output_layer(model, num_classes):
    # Modify the last layer to match the number of target classes
    model.layers.pop()  # Remove the original output layer
    output_layer = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    model.add(output_layer)
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy', 
                  optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 transfer_learning.py translated_dataset.csv word_vectors.kv")

    dataset_filename = sys.argv[1]
    word_vectors_filename = sys.argv[2]
    texts, labels = load_data(dataset_filename)
    word_vectors = load_word_vectors(word_vectors_filename)

    texts = [text if isinstance(text, str) else "" for text in texts]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    max_len = 200  # Use the same max_len as before

    train_generator = DataGenerator(X_train, y_train, word_vectors, max_len=max_len)
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=max_len, shuffle=False)

    # Load your existing model
    model = load_model("bidirectional_lstm_model.keras")

    # Modify the output layer to match the number of classes in the new dataset
    num_classes = len(set(labels))  # Adjust for binary or multi-class classification
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
