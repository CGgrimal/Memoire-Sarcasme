import pandas as pd
import sys

def merge_label_column(original_csv, translated_csv, output_csv, label_column="label", comment_column="comment"):
    # Load the original CSV
    df_original = pd.read_csv(original_csv, sep='|', header=0)

    # Load the translated CSV
    df_translated = pd.read_csv(translated_csv, sep='|')

    # Add the label column from the original to the translated
    df_translated[label_column] = df_original[label_column].astype(int)

    # Save the result to a new CSV
    df_translated.to_csv(output_csv, sep='|', index=False)

    print(f"Labels merged successfully into {output_csv}")

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python merge_labels.py original.csv translated.csv output.csv")
    
    original_csv = sys.argv[1]
    translated_csv = sys.argv[2]
    output_csv = sys.argv[3]

    merge_label_column(original_csv, translated_csv, output_csv)

if __name__ == "__main__":
    main()
