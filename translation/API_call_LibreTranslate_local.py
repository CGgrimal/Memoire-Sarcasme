import pandas as pd
import sys
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

# Translate the dataset and save to a new CSV
def translate_dataset(csv_file, column_name="comment", output_file="translated_dataset.csv", max_workers=4):
    
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='|', header=0, usecols=[column_name])

    # Perform translation with progress
    print(f"Translating {len(df)} rows of text from English to French...")
    df['translated'] = batch_translate_texts(df[column_name].tolist(), max_workers=max_workers)

    # Save the translated dataset to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Translated dataset saved to {output_file}")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 API_call_LibreTranslate_local.py database.csv")
    csv_file = sys.argv[1] 
    translate_dataset(csv_file, column_name="comment", output_file="translated_dataset.csv", max_workers=15)

if __name__ == "__main__":
    main()
