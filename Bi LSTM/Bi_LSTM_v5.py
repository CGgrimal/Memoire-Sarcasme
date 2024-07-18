import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
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
    filename = filename + ".csv"
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

def build_model(input_shape, output_size=20):
    model = Sequential()
    model.add(Bidirectional(LSTM(output_size, return_sequences=True, input_shape=input_shape)))
    model.add(Flatten()) # Flattening l-dimensional array into 2D (dubious)
    model.add(BatchNormalization())
    model.add(Dense(40, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    dataset_filename = str(input("Dataset filename: "))
    word_vectors_filename = str(input("word vectors (kv) file: "))
    texts, labels = load_data(dataset_filename)
    word_vectors = load_word_vectors(word_vectors_filename)

    texts = [text if isinstance(text, str) else "" for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    max_len = 200  # Cap the maximum length to a manageable value

    train_generator = DataGenerator(X_train, y_train, word_vectors, max_len=max_len)
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=max_len, shuffle=False)

    input_shape = (max_len, word_vectors.vector_size)

    model = build_model(input_shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)

    model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {accuracy}')

    model.save("bidirectional_lstm_model.keras")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
