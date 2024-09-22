import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy import triu
import nltk
from gensim.models import Word2Vec

def preprocess_text(text):
    """
    performing operations on sample 
    """
    if not isinstance(text, str):
        return []
    
    #removing punctuation
    text = re.sub(r'[^\w\s]', '', text)

    #tokenizing
    words = word_tokenize(text)

    #removing stop words and stem
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return words


def main():
    filename = str(input("Filename: "))
    df = pd.read_csv(filename, sep = '|', on_bad_lines = 'skip')
    texts = df["comment"].tolist()
    nltk.download('punkt')
    nltk.download('stopwords')

    global stop_words
    stop_words = set(stopwords.words('english'))
    global stemmer
    stemmer = PorterStemmer()
    
    processed_texts = [preprocess_text(text) for text in texts]

    #training word2vec
    model = Word2Vec(sentences =  processed_texts, window = 5, min_count = 1, workers = 4)
    
    #getting word vectors and saving
    word_vectors = model.wv
    word_vectors.save_word2vec_format("sarc_w2v.bin", binary=True)
    model.save("sarc_w2v.model")
    
if __name__ == "__main__":
    main()

