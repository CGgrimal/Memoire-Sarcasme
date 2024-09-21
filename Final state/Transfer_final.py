import pandas as pd
import numpy as np
import sys
import csv
import fasttext
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
            seq = [self.word_vectors[word] for word in words if word in self.word_vectors]
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

# Load FastText word vectors and reduce dimensions using PCA
def load_and_reduce_word_vectors(filename, target_dim=200):
    # Load FastText model
    word_vectors = fasttext.load_model(filename)
    
    # Get all word vectors (shape: vocab_size x 300)
    words = word_vectors.get_words()
    word_vecs = np.array([word_vectors.get_word_vector(word) for word in words])

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=target_dim)
    reduced_word_vecs = pca.fit_transform(word_vecs)

    # Create a dictionary for reduced vectors
    reduced_vectors = {word: reduced_word_vecs[i] for i, word in enumerate(words)}
    
    # Return the reduced vectors in a dictionary-like format for easy access
    return reduced_vectors
def modify_output_layer(model, num_classes):
    # Modify the last layer to match the number of target classes
    model.pop()  # Remove the original output layer
    output_layer = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid', name='new_output')
    model.add(output_layer)
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy', 
                  optimizer=optimizer, metrics=['accuracy'])
    return model

def fine_tune_model(model):
    """
    Unlock the top layers of the model for fine-tuning.
    This step will allow these layers to be retrained on the new data.
    """
    # Set the first layers (LSTM, etc.) to non-trainable
    for layer in model.layers[:-1]:  # Only retrain top layers (last few layers)
        layer.trainable = False

    # Compile again after setting layers to non-trainable
    optimizer = RMSprop(learning_rate=0.0001)  # Use a lower learning rate for fine-tuning
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def retrain_top_layers(model):
    """
    Retrain the top layers by unlocking them for training.
    """
    # Unlock the top layers
    for layer in model.layers:
        layer.trainable = True

    # Re-compile the model for the final training stage with a smaller learning rate
    optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate to avoid large weight updates
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 Transfer.py pretrained_model.keras translated_dataset.csv word_vectors.bin")

    model_filename = sys.argv[1]
    dataset_filename = sys.argv[2]
    word_vectors_filename = sys.argv[3]
    texts, labels = load_data(dataset_filename)
    word_vectors = load_and_reduce_word_vectors(word_vectors_filename, target_dim=200)

    texts = [text if isinstance(text, str) else "" for text in texts]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    max_len = 200  # Use the same max_len as in the English model

    train_generator = DataGenerator(X_train, y_train, word_vectors, max_len=max_len)
    test_generator = DataGenerator(X_test, y_test, word_vectors, max_len=max_len, shuffle=False)

    # Load your existing model
    model = load_model(model_filename)

    # Fine-tune the top layers
    model = fine_tune_model(model)

    # Modify the output layer to match the number of classes in the new dataset
    # This option is completely superflous, but it was added early to accomodate the originally intended dataset and I cant be bothered to remove it at this point
    num_classes = 1 #this is set to 1 as the current dataset has only 2 classes, but if things ever evolve you know where to find me
    model = modify_output_layer(model, num_classes)

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    # Train the model with the new data (fine-tuning the unlocked layers)
    model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

    # Retrain top layers for fine-tuning
    model = retrain_top_layers(model)

    # Continue training the entire model
    model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {accuracy}')

    # Save the updated model
    model.save("transfer_learning_model_fine_tuned.keras")
    print("Model saved successfully")

if __name__ == "__main__":
    main()
