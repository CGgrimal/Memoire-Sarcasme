import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
filename = str(input("filename: "))
output_name = str(input("output file: "))

def preprocess_text(text):
    if not isinstance(text, str):
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

def main():
    filename_ext = filename + ".csv"
    
    # Load the CSV file into a DataFrame with error handling for irregular rows
    try:
        df = pd.read_csv(filename_ext, sep = '|', header = 0)
    except pd.errors.ParserError as e:
        print(f"Error reading {filename}: {e}")
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
    model = Word2Vec(sentences=processed_texts, vector_size=200, window=5, min_count=1, workers=4)

    # Save the entire Word2Vec model
    model.save(f"{output_file}_vectorized.model")

    # Alternatively, save only the word vectors
    model.wv.save_word2vec_format(f"{output_file}_word_vectors.bin", binary=True)

    # Vectorize each text entry
    text_vectors = [vectorize_text(text, model) for text in texts]

    # Save the text vectors to a file for later use
    np.save(f"{output_file}_text_vectors.npy", text_vectors)
    print("Text vectors saved successfully.")

def check_model(filename):
    try:
        # Load the entire Word2Vec model
        model = Word2Vec.load(f"{filename}_vectorized.model")
        print("Word2Vec model loaded successfully.")
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return
    
    try:
        # Load only the word vectors
        word_vectors = KeyedVectors.load_word2vec_format(f"{filename}_word_vectors.bin", binary=True)
        print("Word vectors loaded successfully.")
    except Exception as e:
        print(f"Error loading word vectors: {e}")
        return
    
    # Check if specific words are in the vocabulary
    words_to_check = ['travel', 'problem']
    for word in words_to_check:
        if word in word_vectors:
            print(f"'{word}' is in the model.")
        else:
            print(f"'{word}' is NOT in the model.")
    
    # Get the vector for a known word
    word = 'right'
    if word in word_vectors:
        vector = word_vectors[word]
        print(f"Vector for '{word}':\n{vector}")
    else:
        print(f"'{word}' not found in the model.")
    
    # Check the similarity between two words
    word1 = 'example'
    word2 = 'sample'
    if word1 in word_vectors and word2 in word_vectors:
        similarity = word_vectors.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity}")
    else:
        print(f"One or both words ('{word1}', '{word2}') are not in the model.")
    
    # Find the most similar words to a given word
    word = 'home'
    if word in word_vectors:
        most_similar = word_vectors.most_similar(word, topn=5)
        print(f"Most similar words to '{word}':")
        for similar_word, similarity in most_similar:
            print(f"  {similar_word}: {similarity}")
    else:
        print(f"'{word}' not found in the model.")
    

if __name__ == "__main__":
    main()
    #check_model(filename)
