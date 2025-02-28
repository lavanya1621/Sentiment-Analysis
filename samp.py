import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from google.cloud import language_v1
import os
from dotenv import load_dotenv

nlp = spacy.load("en_core_web_sm")

load_dotenv()


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")



class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        
        # Process with SpaCy for text preprocessing
        doc = self.tokenizer(review)
        
        # Extract cleaned tokens (removing stopwords and punctuation)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        tokens = tokens[:self.max_len]  # Truncate to max length
        
        # Create word IDs (simplified - in production you'd use a proper vocabulary)
        word_ids = [hash(token) % 10000 for token in tokens]  # Simple hashing for demo
        
        # Pad sequence
        if len(word_ids) < self.max_len:
            word_ids = word_ids + [0] * (self.max_len - len(word_ids))
            
        # Convert to tensor
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {"input_ids": word_ids_tensor, "label": label, "original_text": review}

# Simple sentiment analysis model using PyTorch
class SentimentModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=100, hidden_dim=128, output_dim=3):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Google NL API sentiment analyzer
def analyze_with_google_api(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )
    
    # Analyze sentiment
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    
    # Analyze entities
    entities = client.analyze_entities(request={'document': document}).entities
    
    # Map score to categories (negative: -1.0 to -0.25, neutral: -0.25 to 0.25, positive: 0.25 to 1.0)
    if sentiment.score < -0.25:
        category = 0  # Negative
    elif sentiment.score > 0.25:
        category = 1  # Positive
    else:
        category = 2  # Neutral
        
    # Extract important entities
    key_entities = [entity.name for entity in entities if entity.salience > 0.1]
    
    return {
        "category": category,
        "score": sentiment.score,
        "magnitude": sentiment.magnitude,
        "key_entities": key_entities
    }

# Function to perform named entity recognition with SpaCy
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
        
    return entities

# Main sentiment analysis pipeline
def analyze_reviews(reviews_data):
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        reviews_data["text"], reviews_data["label"], test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = ReviewDataset(train_texts.tolist(), train_labels.tolist(), nlp)
    val_dataset = ReviewDataset(val_texts.tolist(), val_labels.tolist(), nlp)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = SentimentModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model (simplified)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"]
            labels = batch["label"]
            
            # Forward pass
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_dataloader):.4f}")
    
    # Evaluate with our model and Google API
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"]
            labels = batch["label"]
            original_texts = batch["original_text"]
            
            # Get PyTorch model predictions
            outputs = model(input_ids)
            _, predictions = torch.max(outputs, 1)
            
            # Process each review
            for i, text in enumerate(original_texts):
                # Get Google API analysis
                try:
                    google_result = analyze_with_google_api(text)
                except Exception as e:
                    print(f"Google API error: {e}")
                    google_result = {"category": -1, "score": 0, "magnitude": 0, "key_entities": []}
                
                # Get named entities with SpaCy
                entities = extract_entities(text)
                
                # Store results
                results.append({
                    "text": text,
                    "true_label": labels[i].item(),
                    "pytorch_prediction": predictions[i].item(),
                    "google_prediction": google_result["category"],
                    "google_score": google_result["score"],
                    "google_magnitude": google_result["magnitude"],
                    "key_entities": google_result["key_entities"],
                    "spacy_entities": entities
                })
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Sample data - in real use, you'd load this from a CSV file
    sample_reviews = pd.DataFrame({
        "text": [
            "This product is amazing, I love it!",
            "Terrible quality, broke after first use.",
            "It's okay, nothing special but works as described.",
            "Absolutely fantastic product, exceeded expectations!",
            "Waste of money, don't buy this."
        ],
        "label": [1, 0, 2, 1, 0]  # 0: negative, 1: positive, 2: neutral
    })
    
    # Analyze reviews
    results = analyze_reviews(sample_reviews)
    
    # Print results
    print("\nSentiment Analysis Results:")
    for _, row in results.iterrows():
        print(f"Text: {row['text']}")
        print(f"True label: {row['true_label']}, PyTorch: {row['pytorch_prediction']}, Google: {row['google_prediction']}")
        print(f"Key entities: {row['key_entities']}")
        print(f"SpaCy entities: {row['spacy_entities']}")
        print("-" * 50)