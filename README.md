# Language Modeling and Speech-to-Text with Transformers

This repository contains the implementation of a project focusing on implementing and understanding transformer models for language modeling and end-to-end speech recognition. The project is divided into two parts: language modeling and speech-to-text transcription.

---

## Table of Contents
1. [Part 1: Language Modeling with Causal Transformers](#part-1-language-modeling-with-causal-transformers)
2. [Part 2: Attention-Based Speech-to-Text Transformer](#part-2-attention-based-speech-to-text-transformer)

---

## Part 1: Language Modeling with Causal Transformers

### Overview
In this part, a decoder-only transformer model is implemented for character-based language modeling. The tasks include:
- Implementing self-attention mechanisms (forward and backward passes).
- Training a causal transformer for next-token prediction and sequence generation.
- Incorporating positional encoding and multi-head attention.

### Key Objectives
- Implement a causal decoder-only transformer for next-token prediction.
- Develop an understanding of self-attention, multi-head attention, and positional encoding.
- Train the model using maximum likelihood estimation (MLE) for accurate language modeling.

### Results
- **Test Perplexity**: Achieved **1.18** on the test dataset.
- **Negative Log Likelihood (NLL)**: Final test NLL of **2.9**.
- **Sequence Generation Quality**: Produced coherent and meaningful text completions.

---

## Part 2: Attention-Based Speech-to-Text Transformer

### Overview
This part focuses on developing a transformer-based sequence-to-sequence model for automatic speech recognition (ASR). The model transcribes speech audio into English text using an encoder-decoder architecture.

### Key Objectives
- Implementing self-attention, multi-head attention, and cross-attention in encoder and decoder layers.
- Training the transformer model on the **train-clean-100** dataset from LibriSpeech.
- Utilizing teacher forcing during training and applying inference techniques like greedy and beam search for decoding.

### Results
- **Character Error Rate (CER)**: Achieved **10.21%** CER on the validation set.
- **Word Error Rate (WER)**: Achieved **11.43%** WER on the validation set.
- **Inference Quality**: Successfully transcribed test samples into accurate English text.

### Notable Features
- The encoder was trained on FBank or MFCC features, which were passed through CNN-pBLSTM embeddings for enhanced sequence representation.
- The implementation leverages positional encoding and masking techniques to handle variable-length input sequences.
- Fine-tuned hyperparameters such as attention heads, model dimensions, and learning rate schedules for optimal performance.


---

## Repository Structure
- `attention.py`: Implementation of self-attention mechanisms for Part 1.
- `language_modeling.ipynb`: Notebook for training and evaluating the causal transformer (Part 1).
- `speech_transformer.ipynb`: Notebook for training and evaluating the speech-to-text model (Part 2).
- `README.md`: This documentation file.

---
