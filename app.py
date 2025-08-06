import streamlit as st
import joblib
import re

# Load the model
model = joblib.load('model/spam_model.pkl')

# Preprocessing function (modify this to match your training phase)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit app layout
st.title("📧 Spam Message Detector")

st.write("Enter a message to check whether it's spam or not:")

user_input = st.text_area("Message", height=150)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_text = preprocess(user_input)
        prediction = model.predict([clean_text])[0]
        if prediction == 1 or prediction == "spam":
            st.error("❌ This is a SPAM message.")
        else:
            st.success("✅ This is a NOT SPAM (ham) message.")
