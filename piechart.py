import pandas as pd
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Load and clean dataset
df = pd.read_csv("fake_or_real_news.csv")
stop_words = stopwords.words('english')

def clean_text(text):
    if type(text) != str:
        return ""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label'].map({'REAL': 0, 'FAKE': 1})

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: navy;'>📰 Fake News Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a news article or headline below to detect whether it's <b>FAKE</b> or <b>REAL</b>.</p>", unsafe_allow_html=True)

if "fake_count" not in st.session_state:
    st.session_state.fake_count = 0
if "real_count" not in st.session_state:
    st.session_state.real_count = 0

news_input = st.text_area(" Enter News Here", height=200)

if st.button(" Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        cleaned = clean_text(news_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]

        if pred == 1:
            st.session_state.fake_count += 1
            st.markdown("<div style='padding:20px; background-color:#ffe6e6; border-radius:10px; text-align:center;'><h2 style='color:red;'>❌ FAKE NEWS</h2></div>", unsafe_allow_html=True)
        else:
            st.session_state.real_count += 1
            st.markdown("<div style='padding:20px; background-color:#e6ffe6; border-radius:10px; text-align:center;'><h2 style='color:green;'>✅ REAL NEWS</h2></div>", unsafe_allow_html=True)

if st.session_state.fake_count + st.session_state.real_count > 0:
    st.subheader("🧮 Prediction Summary This Session:")
    labels = ['REAL', 'FAKE']
    sizes = [st.session_state.real_count, st.session_state.fake_count]
    colors = ['#90ee90', '#ff9999']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  
    st.pyplot(fig)
