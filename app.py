import streamlit as st
from model import train_model, predict

st.title("📧 Spam Email Classifier (No ML Library)")

spam_words, ham_words = train_model()

user_input = st.text_area("Enter a message")

if st.button("Check"):
    if user_input:
        result = predict(user_input, spam_words, ham_words)

        if result == 1:
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")