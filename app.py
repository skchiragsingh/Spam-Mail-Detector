import streamlit as st
import pandas as pd
import re
import nltk
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="MailScan | Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_resources():

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

    def clean_text(text):

        text = text.lower()

        text = re.sub(
            r"http\S+|www\S+|https\S+",
            "",
            text
        )

        text = re.sub(
            r"[^a-zA-Z\s]",
            "",
            text
        )

        tokens = word_tokenize(text)

        tokens = [
            word
            for word in tokens
            if word not in stop_words
        ]

        return " ".join(tokens)

    df["cleaned_message"] = df["message"].apply(clean_text)

    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(
        df["cleaned_message"]
    )

    model = MultinomialNB()

    model.fit(X, df["label"])

    return model, tfidf, stop_words


model, tfidf, stop_words = load_resources()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(99, 102, 241, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at bottom right,
            rgba(16, 185, 129, 0.12),
            transparent 35%
        ),
        #0f172a;
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    color: #f8fafc;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 3rem;
}

div[data-testid="stTextArea"] label {
    color: #f8fafc !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}

.stTextArea textarea {
    background-color: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    padding: 16px !important;
    min-height: 170px !important;
    box-shadow:
        0 10px 30px
        rgba(0, 0, 0, 0.18);
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
    opacity: 1;
}

.stTextArea textarea:focus {
    border: 1px solid #818cf8 !important;
    box-shadow:
        0 0 0 3px
        rgba(99, 102, 241, 0.18)
        !important;
}

.stButton > button {
    width: 100%;
    height: 54px;
    border: none;
    border-radius: 14px;
    background:
        linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );
    color: white;
    font-size: 17px;
    font-weight: 600;
    margin-top: 10px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #6366f1
        );
    box-shadow:
        0 12px 30px
        rgba(99, 102, 241, 0.35);
}

.result-spam {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    color: #fca5a5;
    font-size: 20px;
    font-weight: 700;
    margin-top: 25px;
}

.result-ham {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    color: #6ee7b7;
    font-size: 20px;
    font-weight: 700;
    margin-top: 25px;
}

.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 3rem;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">MailScan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered message analysis to detect potential spam instantly.</div>',
    unsafe_allow_html=True
)


message = st.text_area(
    "Enter your message",
    height=170,
    placeholder="Paste or type a message here..."
)


check = st.button(
    "🔍 Analyze Message"
)


if check:

    if message.strip() == "":

        st.warning(
            "Please enter a message first."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing your message..."
        ):

            time.sleep(1.5)

            cleaned_message = preprocess_text(message)

            message_tfidf = tfidf.transform(
                [cleaned_message]
            )

            prediction = model.predict(
                message_tfidf
            )[0]

        if prediction == "spam":

            st.markdown("""
            <div class="result-spam">
                🚨 SPAM DETECTED
                <br><br>
                <span style="font-size:14px; font-weight:400;">
                    This message appears to contain spam-like content.
                </span>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-ham">
                ✅ MESSAGE LOOKS SAFE
                <br><br>
                <span style="font-size:14px; font-weight:400;">
                    This message is classified as legitimate.
                </span>
            </div>
            """, unsafe_allow_html=True)


st.markdown(
    '<div class="footer-text">Powered by Machine Learning • TF-IDF • Naive Bayes</div>',
    unsafe_allow_html=True
)
