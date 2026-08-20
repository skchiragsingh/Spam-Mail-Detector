import streamlit as st
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@st.cache_resource
def load_model():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"

    df = pd.read_csv(
        url,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        tokens = word_tokenize(text)
        tokens = [
            word for word in tokens
            if word not in stop_words
        ]

        return " ".join(tokens)

    df["cleaned_message"] = df["message"].apply(preprocess_text)

    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(df["cleaned_message"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return model, tfidf, stop_words


model, tfidf, stop_words = load_model()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Mail Detector")

st.write(
    "Enter a message below and the machine learning model "
    "will classify it as Spam or Ham."
)

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)


if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        cleaned_message = preprocess_text(message)

        message_tfidf = tfidf.transform(
            [cleaned_message]
        )

        prediction = model.predict(
            message_tfidf
        )[0]

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM — Legitimate Message")
