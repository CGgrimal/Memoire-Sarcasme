#!/bin/bash

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

echo "All tasks completed successfully!"

