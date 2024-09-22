#!/bin/bash

exec > log.txt 2>&1

if [ ! -f requirements.txt ]; then
    echo "requirements.txt file not found!"
    exit 1
fi

echo "Installing Python libraries from requirements.txt..."
pip install -r requirements.txt

if [ ! -f pre_processor.py ]; then
    echo "pre-processor file not found!"
    exit 1
fi

echo "Running pre-processor..."
python3 pre_processor.py reduced_1000.csv r1000_custom_vectors 0

if [ ! -f Bi_LSTM.py ]; then
    echo "Bi-LSTM file not found!"
    exit 1
fi

echo "Running Bi-LSTM model..."
python3 Bi_LSTM.py reduced_1000.csv r1000_custom_vectors_word_vectors.bin Bi_LSTM_r1000

if [ ! -f evaluation.py ]; then
    echo "evaluation file not found!"
    exit 1
fi

echo "Running evaluation on Bi-LSTM..."
python3 evaluation.py en reduced_1000.csv Bi_LSTM_r1000.keras r1000_custom_vectors_word_vectors.bin

if [ ! -f Transfer.py ]; then
    echo "transfer file not found!"
    exit 1
fi

echo "Running Transfer learning..."
python3 Transfer.py Bi_LSTM_r1000.keras translated_r2500.csv fasttext_fr.bin

echo "Running evaluation on French model..."
python3 evaluation.py fr translated_r2500.csv transfer_learning_model.keras fasttext_fr.bin

echo "All tasks completed successfully!"

