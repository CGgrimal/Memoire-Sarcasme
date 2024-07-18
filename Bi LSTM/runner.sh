#!/bin/bash

if [ ! -f requirements.txt ]; then
    echo "requirements.txt file not found!"
    exit 1
fi

echo "Installing Python libraries from requirements.txt..."
pip install -r requirements.txt

if [ ! -f pre_processor_non_verbal.py ]; then
    echo "pre-processor file not found!"
    exit 1
fi

echo "Running pre-processor..."
python3 script1.py reduced_1000.csv r1000

if [ ! -f Bi_LSTM_v5_nonverbal.py ]; then
    echo "Bi-LSTM file not found!"
    exit 1
fi

echo "Running Bi-LSTM model..."
python3 script2.py reduced_1000 r1000_word_vectors.kv

echo "All tasks completed successfully!"

