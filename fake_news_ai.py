import os
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(BASE_DIR, "news.zip")

# Read CSV directly from zip , Load dataset
data = pd.read_csv(zip_path, compression="zip")

# Features and labels
x = np.array(data["title"])
y = np.array(data["label"])

# Text → Vector
cv = TfidfVectorizer(stop_words="english", max_df=0.7)
x = cv.fit_transform(x)

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=21)

# Navies bayes algo Model
model = MultinomialNB()
model.fit(xtrain,ytrain)

# Adding LLMs to reason only not classify
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv() #load the env from .env file

# ---------------- URL HANDLING ---------------- #

def is_url(text):
    text = text.strip().lower()
    return text.startswith(("http://", "https://", "www."))
def extract_text_from_url(url):
    try:
        if url.startswith("www."):
            url = "https://" + url

        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())

        return text[:500]   # limit size for ML + LLM

    except Exception:
        return None
# ---------------- LLM EXPLANATION ---------------- #

def get_llm_explanation(text, predict, confidence):
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        token=os.environ["HF_TOKEN"]
    )
    
    if predict == "REAL":
        prompt = f"""
You are an AI assistant for a fake news detection system.

The ML model classified the news as REAL with confidence {confidence:.2f}%.

TASK:
1. Give a short factual summary in 2 lines only.
2. State tone in ONE WORD (Neutral / Emotional / Biased).

FORMAT:
Summary:
- <line 1>
- <line 2>

Tone: <one word>

News:
{text}
"""
    else:
        prompt = f"""
You are an AI assistant for a fake news detection system.

The ML model classified the news as FAKE with confidence {confidence:.2f}%.

TASK:
Explain WHY it looks fake using language patterns.

FORMAT:
Tone: <one word>

Why it looks fake:
- <point 1>
- <point 2>
- <point 3>

News:
{text}
"""
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120
    )

    return completion.choices[0].message.content


# Create GUI for FAKE NEWS DETECTION SYSTEM USING STREAMLIT

import streamlit as st

st.set_page_config(page_title="FAKE NEWS DETECTOR",page_icon="📑",layout="wide")

st.title("📑 **FAKE NEWS DETECTION SYSTEM** " ,text_alignment="center")
st.caption("Naive Bayes Classification with LLM-based Explanation")


# Fake News detection----

def fakenews():
    user_input = st.text_area("Enter News(TEXT / URL to Check (REAL | FAKE) :) ")
    Check = st.button("Check")
    if Check :
        if not user_input.strip():    
            st.warning("Please enter some news text")
            return
        # URL handling
        if is_url(user_input):
            st.info("🔗 URL detected. Extracting article content...")
            extracted = extract_text_from_url(user_input)

            if extracted is None or len(extracted.strip()) == 0:
                st.error("❌ Could not extract text from this URL.")
                return
            else:
                sample = extracted
                st.success("✅ Article text extracted successfully.")
        else:
            sample = user_input

        # Prediction    
        sample = user_input
        data = cv.transform([sample])  # Convert text -> vector
        predict = model.predict(data)[0] # Prediction - using naive bayes 
        prob = model.predict_proba(data)[0] 
          
        if predict == "REAL":
            st.success(f"📰 News is **{predict}**")
        else:
            st.error(f"📰 News is **{predict}**")

        st.subheader("Prediction Probability")
        st.write(f"Fake Probability: {prob[0]*100:0.2f}%") #fake_prob = prob[0]
        st.write(f"Real Probability: {prob[1]*100:0.2f}%")  # real_prob = prob[1]
        confidence = max(prob) * 100
        st.write("Confidence of model :",confidence)
           # ⚠️ Confidence warning
        if confidence > 90 and is_url(user_input):
            st.info("ℹ️ High confidence prediction, but URL-based headlines may be misclassified due to dataset limitations.")
        elif confidence < 60:
            st.warning("⚠️ Model is uncertain about this prediction.")

        # -------- LLM Explanation --------
        st.subheader("🧠 AI Explanation")

        with st.spinner("AI is analyzing the news..."):
            explanation = get_llm_explanation(sample, predict,confidence)

        st.write(explanation)

fakenews()
