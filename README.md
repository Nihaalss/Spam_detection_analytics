# Spam_detection_analytics
# 📩 SMS Spam Detection using SVM

A machine learning project to detect spam SMS messages using text processing techniques and Support Vector Machines (SVM). This project uses the classic SMS Spam Collection dataset and performs preprocessing, exploratory data analysis (EDA), model building, and evaluation.

---

## 🔍 Project Overview

This project classifies SMS messages as **spam** or **ham (not spam)** using a linear SVM classifier. It includes:
- Cleaning and preprocessing the text data
- Exploratory data analysis (EDA) for insights
- TF-IDF vectorization
- Model training and evaluation
- Message prediction/validation for custom input

---

## 📁 Dataset

The dataset used is the [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5,572 SMS messages labeled as 'ham' or 'spam'.

---

## 🧼 Data Cleaning

Steps:
- Removed duplicates
- Dropped unnecessary columns
- Cleaned text by removing non-alphabetic characters and converting to lowercase
- Labeled: `ham` = 0, `spam` = 1

---

## 📊 Exploratory Data Analysis (EDA)

Performed visual analysis using Seaborn and Matplotlib:
- Count of spam vs ham
- Message length distribution
- Correlation heatmap between message length and label
- Boxplots to explore variation in length

---

## 🤖 Model Building

- **Vectorization**: Used `TfidfVectorizer` with English stopwords and 5000 max features
- **Model**: Linear Support Vector Machine (`SVC` with `linear` kernel)
- **Training**: Split into 70% training and 30% test sets

---

## ✅ Evaluation

- **Accuracy**: `98.13%`
- **Precision, Recall, F1-Score**:
    ```
    Precision: 0.98 (Ham), 0.98 (Spam)
    Recall:    1.00 (Ham), 0.88 (Spam)
    F1-Score:  0.99 (Ham), 0.93 (Spam)
    ```
- **Confusion Matrix**: Visualized using Seaborn heatmap

---

## 🧪 Custom Validation

# Example input
new_message = "You've won a free ticket to Goa. Text WON to claim."

# Preprocessing function
def preprocess_message(message, vectorizer):
    import re
    message_cleaned = re.sub('[^a-zA-Z]', ' ', message).lower()
    message_cleaned = ' '.join(message_cleaned.split())
    return vectorizer.transform([message_cleaned]).toarray()

# Make sure you already trained the vectorizer and model like this:
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X = tfidf_vectorizer.fit_transform(df2['cleaned_text']).toarray()
# svm = SVC(kernel='linear', random_state=42)
# svm.fit(X_train, y_train)

# Vectorize and predict
new_message_vectorized = preprocess_message(new_message, tfidf_vectorizer)
prediction = svm.predict(new_message_vectorized)
result = 'Spam' if prediction[0] == 1 else 'Ham'

print(f"The message is classified as: {result}")

