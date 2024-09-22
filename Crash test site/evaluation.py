import numpy as np
import pandas as pd
import sys
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class DataGenerator(Sequence):
    def __init__(self, texts, labels, word_vectors, batch_size=128, max_len=300, shuffle=True, **kwargs):
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
            seq = [self.word_vectors[word] if word in self.word_vectors else np.zeros(self.word_vectors.vector_size) for word in words]
            sequences.append(seq)
        sequences_padded = pad_sequences(sequences, maxlen=self.max_len, dtype='float32', padding='post', truncating='post', value=0.0)
        return np.array(sequences_padded)

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

def load_data(filename, form):
    df = pd.read_csv(filename, sep='|', header=0)
    if form == "en":
        texts = df['comment'].tolist()
    if form == "fr":
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

    # Load word vectors (ensure binary=True for FastText .bin format)
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

    # Extract actual test labels
    y_test_actual = np.concatenate([y for _, y in test_generator])

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
