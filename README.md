# **Hybrid Sentiment Analysis System**  

## **Summary**  
This project implements a **hybrid sentiment analysis system** combining **PyTorch, SpaCy, and Google's Natural Language API**. It features a **custom LSTM neural network** that classifies reviews as **positive, negative, or neutral**, while also using **Google's API for comparison**.  

Right now, the model is trained on only 5 reviews, so it's not performing at its best. The key focus of this project is showcasing my **skills in PyTorch, SpaCy, and integrating APIs**.  

## **How It Works**  
- **Text Processing:** Uses **SpaCy** for **tokenization, lemmatization, and stopword removal**.  
- **Feature Conversion:** A **custom ReviewDataset class** converts processed text into numerical features.  
- **Neural Network:** A **PyTorch-based LSTM model** with **embedding layers** and **LSTM units** to understand text sequences.  
- **Google API Comparison:** Google's Natural Language API is used to compare the **model's predictions** and extract **named entities**.  
- **Current Performance:** Since it's trained on only **5 reviews**, the PyTorch model misclassifies some cases, whereas **Google's API performs better**. This highlights the importance of a **hybrid approach** and the potential for improvement with more training data.  

This project demonstrates **integration of multiple frameworks** and a **modular design** that could be improved with **larger datasets and better embeddings**.  

---

Let me know if you want any more changes! 🚀
