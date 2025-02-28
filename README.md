This project is a hybrid sentiment analysis system using PyTorch, SpaCy, and Google’s Natural Language API to classify reviews as positive, negative, or neutral. Right now, it’s only trained on 5 examples, so it’s not performing at its best. The goal isn’t just accuracy but also showcasing skills in these AI libraries. The custom PyTorch LSTM model sometimes misclassifies reviews, while Google’s API is more accurate. Both systems also extract important words (entities) from reviews, with Google identifying more relevant terms than SpaCy’s Named Entity Recognition (NER).



How It Works
Text Preprocessing: Uses SpaCy to clean text (tokenization, lemmatization, stopword removal).
Data Conversion: A ReviewDataset class converts text into numbers.
Custom Model: A PyTorch LSTM neural network is trained to predict sentiment.
Google API Comparison: The same reviews are analyzed using Google’s Natural Language API to compare results.
Entity Extraction: Both Google API and SpaCy pull out important words linked to sentiment.
Performance: The model isn’t perfect because it’s only trained on 5 reviews, but it still proves strong skills in AI libraries like PyTorch, SpaCy, and Google’s API.
