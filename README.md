# Text Emotion Classification

## Overview

This repository contains code for a text emotion classification model using deep learning techniques. The model is trained to classify text data into different emotion categories.

## Dataset

The dataset used for training the model is from the "TextEmotion NLP" dataset, consisting of text samples labeled with different emotions. It is split into training and testing sets for model evaluation.

## Dependencies

- Python 3.x
- pandas
- numpy
- keras
- tensorflow
- scikit-learn

## Usage

1. **Reading Data**: Data is read from the provided CSV file (`train.txt`) containing text samples and corresponding emotion labels.

2. **Preprocessing**:
   - Tokenization using the Tokenizer class from Keras.
   - Padding sequences to ensure uniform length.
   - Label encoding using the LabelEncoder class from scikit-learn.
   - One-hot encoding of labels.

3. **Model Training**: 
   - Definition of a deep learning model using Keras Sequential API.
   - Model consists of an Embedding layer, Flatten layer, Dense layers, and softmax activation function.
   - Compilation with Adam optimizer and categorical cross-entropy loss.
   - Training using the training data.

4. **Model Evaluation**: 
   - Performance evaluation using test data with accuracy metrics.

5. **Prediction**: 
   - Prediction of emotions for input text samples.
   - Preprocessing of input sentences and prediction using the trained model.

