# Spam Mail Detector

A Machine Learning based Spam Mail Detection system that classifies text messages as **Spam** or **Ham (Not Spam)**.

📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to classify SMS messages into two categories:

- **Spam**
- **Ham**

The model processes text messages, converts them into numerical features using **TF-IDF Vectorization**, and uses machine learning algorithms for classification.

🚀 Features

- SMS text preprocessing
- Lowercasing and text cleaning
- Stopword removal
- Tokenization
- TF-IDF feature extraction
- Naive Bayes classification
- Logistic Regression comparison
- Model evaluation using Accuracy, Precision, Recall and F1 Score
- Spam/Ham prediction for new messages

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit

📊 Dataset

The project uses the **SMS Spam Collection Dataset** containing labelled SMS messages classified as spam or ham.

⚙️ Machine Learning Workflow

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
```

📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Improved Naive Bayes Model Results

| Metric | Score |
|--------|-------|
| Accuracy | 96.62% |
| Precision | 100% |
| Recall | 73.28% |
| F1 Score | 84.58% |

📂 Project Structure

```text
Spam-Mail-Detector/
│
├── spam_detector.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/skchiragsingh/Spam-Mail-Detector.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook spam_detector.ipynb
```

🔮 Future Improvements

- Deploy the application using Streamlit
- Experiment with additional machine learning models
- Improve text preprocessing
- Add real-time email classification

👨‍💻 Author

**Chirag Singh**

B.Tech CSE (Artificial Intelligence)
