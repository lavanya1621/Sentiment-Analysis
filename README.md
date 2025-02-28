Hybrid Sentiment Analysis with PyTorch, SpaCy & Google's API
Overview
This project implements a hybrid sentiment analysis system using PyTorch, SpaCy, and Google's Natural Language API. It classifies text as positive, negative, or neutral, combining a custom LSTM model with Google's API for comparison. Right now, it's trained on only 5 examples, so performance is limited—but the focus is on demonstrating skills in these AI libraries.

How It Works
Preprocessing with SpaCy – Tokenization, lemmatization, and stopword removal.
Custom Dataset Handling – A ReviewDataset class converts text into numerical data.
Sentiment Classification – A PyTorch LSTM model predicts sentiment.
Google API Integration – Reviews are also analyzed using Google’s Natural Language API to compare results.
Entity Extraction – Both SpaCy and Google’s API extract key terms related to sentiment.
Current Performance
Since the model is trained on just 5 reviews, it isn’t highly accurate yet. The PyTorch model misclassifies some reviews, while Google’s API is more reliable. However, this project highlights skills in PyTorch, NLP, and API integration. With more training data and better embeddings, the system could become production-ready. 🚀
