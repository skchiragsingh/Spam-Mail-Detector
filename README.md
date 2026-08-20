# Spam Mail Detector

A Machine Learning based Spam Mail Detection system that classifies text messages as **Spam** or **Ham (Not Spam)**.

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to classify SMS messages into two categories:

- **Spam**
- **Ham**

The model processes text messages, converts them into numerical features using **TF-IDF Vectorization**, and uses machine learning algorithms for classification.

## 🚀 Features

- SMS text preprocessing
- Lowercasing and text cleaning
- Stopword removal
- Tokenization
- TF-IDF feature extraction
- Naive Bayes classification
- Logistic Regression comparison
- Model evaluation using Accuracy, Precision, Recall and F1 Score
- Spam/Ham prediction for new messages

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset** containing labelled SMS messages classified as spam or ham.

## ⚙️ Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Text Preprocessing
   ↓
Train-Test Split
   ↓
TF-IDF Vectorization
   ↓
Model Training
   ├── Naive Bayes
   └── Logistic Regression
   ↓
Model Evaluation
   ↓
Spam/Ham Prediction
