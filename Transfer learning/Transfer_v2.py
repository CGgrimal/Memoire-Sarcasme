import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained model
def load_pretrained_model(model_path):
    return load_model(model_path)

# Freeze the layers except the last few layers
def freeze_layers(model, num_layers_to_train):
    for layer in model.layers[:-num_layers_to_train]:
        layer.trainable = False
    return model

def modify_output_layer(model, num_classes, max_len):
    # Ensure the model is built by calling it with dummy data
    model.build(input_shape=(None, max_len))

    # Modify the output layer
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(model.output)

    # Create the new model
    return tf.keras.Model(inputs=model.input, outputs=output_layer)


# Prepare the dataset for training (tokenization, padding, etc.)
def prepare_french_dataset(csv_file, tokenizer, max_len, label_column='label', text_column='translated'):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Tokenize and pad the French texts
    sequences = tokenizer.texts_to_sequences(df[text_column].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    # Prepare labels
    labels = df[label_column].values

    return padded_sequences, labels

# Train the fine-tuned model on French dataset
def train_fine_tuned_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr])

    return model

def main():

    if len(sys.argv) != 3:
        sys.exit("Usage: python3 Transfer_v1.py model.keras translated_dataset.csv")
        
    # Paths to the pre-trained model and the French dataset
    pretrained_model_path = sys.argv[1]
    french_dataset_path = sys.argv[2]

    # Load the pre-trained model
    model = load_pretrained_model(pretrained_model_path)
    max_len = 200  # Example max_len, should match with English dataset model

    # Freeze layers except the last few (set the number you want to fine-tune)
    model = freeze_layers(model, num_layers_to_train=2)  # Example: Fine-tune the last 2 layers

    # Modify the output layer for the new French classification task
    num_classes = 2  # Example: binary classification (modify accordingly)
    model = modify_output_layer(model, num_classes, max_len)

    # Prepare tokenizer (use the same tokenizer you used for English dataset, adjust vocab if needed)
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")  # Adjust tokenizer params if necessary

    # Prepare the French dataset
    X, y = prepare_french_dataset(french_dataset_path, tokenizer, max_len)

    # Split the French dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the fine-tuned model
    model = train_fine_tuned_model(model, X_train, y_train, X_val, y_val)

    # Save the fine-tuned model
    model.save("fine_tuned_french_model.h5")

if __name__ == "__main__":
    main()
