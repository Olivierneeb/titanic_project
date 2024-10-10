# Makefile

all: prepare_data train evaluate

prepare_data:
    python3 src/preprocess.py

train:
    python3 src/train.py

evaluate:
    python3 src/evaluate.py
