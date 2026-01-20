# 📰 Fake News Detection System with ML & LLM Explanation

A Streamlit-based web application that classifies news as REAL or FAKE using a Naive Bayes machine learning model and provides human-readable AI explanations using Hugging Face Large Language Models (LLMs) for transparency and trust.

This project combines traditional NLP classification with LLM-based reasoning to not only predict fake news but also explain why a piece of news may look real or fake.

---

## 🔍 Problem Statement

Fake news spreads rapidly on digital platforms and can mislead users, influence opinions, and cause social harm.

This system aims to:

- Automatically detect fake news using Machine Learning  
- Display probability and confidence scores for predictions  
- Provide simple AI-generated explanations for better interpretability  
- Support both text input and news URLs  

---

## 🧠 Approach & Architecture

### 🔁 Processing Pipeline

User Input (Text / URL)  
→ Text Extraction (if URL)  
→ TF-IDF Vectorization  
→ Naive Bayes Classification  
→ Prediction Probability & Confidence  
→ LLM Explanation (Reasoning only)  
→ Result Display (Streamlit UI)  

---

## ⚙️ Technologies Used

- Programming Language: Python  
- ML Model: Multinomial Naive Bayes (scikit-learn)  
- Text Vectorization: TF-IDF Vectorizer  
- Web Framework: Streamlit  
- Web Scraping: BeautifulSoup + Requests  
- LLM API: Hugging Face Inference API  
- Environment Manager: uv  

---

## 🤖 Why Naive Bayes?

- Works extremely well for text classification problems  
- Fast and lightweight  
- Performs reliably on bag-of-words and TF-IDF features  
- Easy to interpret and deploy  

---

## 📊 Confidence Score

- Confidence is computed using `predict_proba()` from scikit-learn  
- Represents how sure the model is about its prediction  
- The system shows:
  - Fake probability  
  - Real probability  
  - Final confidence score  

⚠️ A warning is shown when confidence is low or when URL-based content may be unreliable.

---

## 🧠 Role of the LLM (Explainability Layer)

The LLM:

- Does NOT classify the news  
- Only explains the ML model’s prediction  
- Analyzes:
  - Writing style  
  - Language patterns  
  - Tone and bias  

### Output examples:

For REAL news:
- Short 2-line factual summary  
- Tone detection (Neutral / Emotional / Biased)  

For FAKE news:
- Tone detection  
- Bullet-point explanation of suspicious patterns  

This improves model transparency and user trust.

---

## 🖥️ Features

- Paste news text or URL  
- Automatic article text extraction from URLs  
- Real / Fake classification  
- Prediction probabilities  
- Confidence score with warnings  
- AI-generated explanation  
- Clean and interactive Streamlit UI  

---

## 🚀 How to Run Locally (Using `uv`)

### 1️⃣ Clone the Repository

git clone https://github.com/piyushsb007/FAKE-NEWS-DETECTION.git

cd FAKE-NEWS-DETECTION

---

### 2️⃣ Initialize Environment with `uv`

uv init .

---

### 3️⃣ Install Dependencies

uv add streamlit pandas numpy scikit-learn huggingface-hub python-dotenv beautifulsoup4 requests  

---

### 4️⃣ Create `.env` File

Create a file named `.env` in the project root:

HF_TOKEN=your_huggingface_token_here  

---

### 5️⃣ Run the Application

uv run streamlit run fake_news_ai.py  

---

## 📂 Dataset

- Dataset file: news.csv / news.zip  
- Columns used:
  - title → News text  
  - label → REAL / FAKE  

The model is trained using supervised learning on labeled news headlines.

---

## 📈 Model Details

- Algorithm: Multinomial Naive Bayes  
- Vectorization: TF-IDF  
- Train-Test Split: 80% / 20%  
- Probability scores used for confidence estimation  

---

## 🧩 Project Structure

fake-news-detection-ml-llm/

├── fake_news_ai.py        # Main Streamlit application  
├── news.zip / news.csv   # Dataset  
├── .env                  # Hugging Face API token (not committed)  
├── pyproject.toml        # uv dependency file  
└── README.md             # Project documentation  

---

## ⚠️ Disclaimer

This system analyzes language patterns only and does NOT verify real-world facts.

Predictions should be used for educational and research purposes only, not as a final authority on news authenticity.

---

## 🔮 Future Improvements

- Add larger and more diverse training datasets  
- Improve article extraction for complex websites  
- Add source credibility scoring  
- Integrate stronger LLMs for better explanations  
- Add charts and history tracking in UI  


---
