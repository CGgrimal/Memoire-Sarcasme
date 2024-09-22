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
