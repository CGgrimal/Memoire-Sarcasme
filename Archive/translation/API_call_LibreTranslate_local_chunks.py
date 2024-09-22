import pandas as pd
import sys
import os
from libretranslatepy import LibreTranslateAPI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Initialize LibreTranslate connection to your local instance
lt = LibreTranslateAPI("http://localhost:5000")

# Function to translate text
def translate_text(text, source_lang="en", target_lang="fr"):
    try:
        return lt.translate(text, source=source_lang, target=target_lang)
    except Exception as e:
        print(f"Error translating text: {e}")
        return text  # Return the original text if there's an error

# Batch translation using ThreadPoolExecutor for parallel translation with progress tracking
def batch_translate_texts(texts, source_lang="en", target_lang="fr", max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        translations = list(tqdm(executor.map(lambda text: translate_text(text, source_lang, target_lang), texts), total=len(texts), desc="Translating", unit="text"))
    return translations

# Translate the dataset in chunks and save progressively
def translate_dataset_in_chunks(csv_file, column_name="comment", output_file="translated_dataset.csv", max_workers=4, chunk_size=1000):
    # Load the CSV file in chunks
    chunk_iter = pd.read_csv(csv_file, sep='|', header=0, usecols=[column_name], chunksize=chunk_size)
    
    # Create an empty file or append if it exists
    mode = 'a' if os.path.exists(output_file) else 'w'
    
    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i + 1}...")

        # Perform translation with progress tracking
        chunk['translated'] = batch_translate_texts(chunk[column_name].tolist(), max_workers=max_workers)

        # Save each chunk of the translated dataset
        chunk.to_csv(output_file, sep = '|', mode=mode, header=(mode == 'w'), index=False)
        
        # Update mode to append after the first chunk
        mode = 'a'
        
        print(f"Chunk {i + 1} saved to {output_file}")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 API_call_LibreTranslate_local.py database.csv")
    csv_file = sys.argv[1] 

    # Set the output file name and other parameters
    output_file = "translated_dataset.csv"
    
    # Translate dataset in chunks (chunk size of 1000 rows as an example)
    translate_dataset_in_chunks(csv_file, column_name="comment", output_file=output_file, max_workers=15, chunk_size=10000)

if __name__ == "__main__":
    main()
