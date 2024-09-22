import pandas as pd
import deepl
import os

# Initialize DeepL Translator
def init_deepl_client(api_key):
    return deepl.Translator(api_key)

# Function to translate text using DeepL API
def translate_text(text, translator, target_language='FR'):
    if isinstance(text, str):
        # Translation request to DeepL API
        result = translator.translate_text(text, target_lang=target_language)
        return result.text
    else:
        return ""

# Load dataset
def load_data(filename):
    df = pd.read_csv(filename + ".csv", sep='|', header=0)
    return df

# Translate a portion of the dataset
def translate_dataset(df, column_name, n_samples, translator):
    translated_texts = []
    for index, text in df[column_name].head(n_samples).items():
        translated_text = translate_text(text, translator)
        translated_texts.append(translated_text)
    return translated_texts

# Save translated data to a new CSV file
def save_translated_data(df, translated_texts, output_filename):
    df_translated = df.head(len(translated_texts)).copy()
    df_translated['translated_comment'] = translated_texts
    df_translated.to_csv(output_filename + "_translated.csv", sep='|', index=False)
    print(f"Translated dataset saved to {output_filename}_translated.csv")

def main():
    # Fetch DeepL API Key
    api_key = os.getenv('DEEPL_API_KEY')  # Set your API key as an environment variable
    if not api_key:
        raise ValueError("Please set the DEEPL_API_KEY environment variable with your DeepL API key.")
    
    # Initialize DeepL client
    translator = init_deepl_client(api_key)

    # Get user inputs
    dataset_filename = str(input("Dataset filename (without extension): "))
    column_name = str(input("Column to translate: "))
    n_samples = int(input("Number of samples to translate: "))
    output_filename = str(input("Output filename prefix (without extension): "))
    
    # Load the dataset
    df = load_data(dataset_filename)
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset.")
    
    # Translate the dataset
    translated_texts = translate_dataset(df, column_name, n_samples, translator)
    
    # Save the translated dataset to a new CSV file
    save_translated_data(df, translated_texts, output_filename)

if __name__ == "__main__":
    main()
