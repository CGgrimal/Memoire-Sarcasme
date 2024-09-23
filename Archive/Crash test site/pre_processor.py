import pandas as pd
import re
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        print(f"Non-string value encountered: {text}")
        return []
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stop words and perform stemming
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    
    return words

def vectorize_text(text, model):
    words = preprocess_text(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def save_word_vectors(model, filename):
    word_vectors = model.wv
    word_vectors.save(filename)

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 pre_processor.py dataset.csv output_prefix bool(save all)")
    
    filename_ext = sys.argv[1]
    output_name = sys.argv[2]
    save_all = sys.argv[3]
    
    # Load the CSV file into a DataFrame with error handling for irregular rows
    try:
        df = pd.read_csv(filename_ext, sep = '|', header = 0)
    except pd.errors.ParserError as e:
        print(f"Error reading {filename_ext}: {e}")
        return

    column_name = "comment"
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in CSV.")
        return

    texts = df[column_name].tolist()

    # Initialize stop words and stemmer
    global stop_words
    stop_words = set(stopwords.words('english'))
    global stemmer
    stemmer = PorterStemmer()

    # Apply preprocessing to all texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    print("Sample preprocessed texts:")
    for text in processed_texts[:5]:
        print(text)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=processed_texts, vector_size=200, window=5, min_count=1, workers=4)

    if save_all == 1:
        # Save the entire Word2Vec model
        word2vec_model.save(f"{output_name}_vectorized.model")
        print("Model saved successfully")

        # save only the word vectors
        save_word_vectors(word2vec_model, f"{output_name}_word_vectors.kv")
        print("Word vectors saved successfully")

        # Save the text vectors to a file
        text_vectors = np.array([np.mean([word2vec_model.wv[word] for word in text if word in word2vec_model.wv] or [np.zeros(200)], axis=0) for text in processed_texts])
        np.save(f"{output_name}_text_vectors.npy", text_vectors)
        print("Text vectors saved successfully.")

        # Save tokenized texts
        tokenized_df = df[["comment", "label"]].copy()
        tokenized_df["comment"] = processed_texts
        tokenized_df.to_csv(f"{output_name}_tokens", sep='|', mode='w', index=False, header=True)
        print("Tokenized database saved successfully.")

    # AND save the word vectors as a .Bin file
    word2vec_model.wv.save_word2vec_format(f'{output_name}_word_vectors.bin', binary=True)

    
if __name__ == "__main__":
    main()
