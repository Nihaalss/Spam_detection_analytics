import streamlit as st
import joblib
import re
import string

# --- Clean text function ---
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load model and vectorizer ---
model = joblib.load("model/svm_model.pkl")  # adjust filename if needed
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# --- Page setup ---
st.set_page_config(page_title="Spam Detection App", page_icon="📧", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #2d3748;
        }
        .stTextArea > div > div > textarea {
            border-radius: 10px;
            border: 2px solid #e1e5e9;
            background-color: #f7fafc !important;
            color: #2d3748 !important;
        }
        .stButton > button {
            border-radius: 8px;
            border: none;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #e2e8f0; margin-bottom: 10px;">📩 Spam Message Classifier</h1>
        <p style="color: #cbd5e0; font-size: 18px;">Type your message below and check if it's spam or not.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Input Area ---
user_input = st.text_area("✏️ Enter a message to classify:", height=150)

# --- Predict Button ---
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before clicking Predict.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        vectorized_input = vectorized_input.toarray()
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("🚫 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **NOT SPAM (HAM)**.")

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #4a5568; color: #a0aec0;">
        <em>Built using Streamlit ✨</em>
    </div>
    """,
    unsafe_allow_html=True
)
