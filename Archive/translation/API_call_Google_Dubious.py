import pandas as pd
from googletrans import Translator
import time

# Initialize the Translator
translator = Translator()

# Load the CSV file and the specific column with text
filename = 'your_input_file.csv'
df = pd.read_csv(filename, usecols=['comment'])

# Add a new column for the translated text
df['translated_comment'] = None

# Function to translate a single text
def translate_text(text):
    try:
        translated = translator.translate(text, src='en', dest='fr')
        return translated.text
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

# Translate each comment and store it in the new column
for index, row in df.iterrows():
    text = row['comment']
    if pd.isna(text):
        continue
    
    translated_text = translate_text(text)
    df.at[index, 'translated_comment'] = translated_text
    
    # Print progress
    if index % 100 == 0:
        print(f"Translated {index} rows")
    
    # Sleep between requests to avoid getting blocked
    time.sleep(1)  # Adjust this if needed

# Save the translated comments to a new CSV file
df.to_csv('translated_comments.csv', index=False)
print("Translation completed and saved.")
