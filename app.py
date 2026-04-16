import streamlit as st
from model import train_model

st.title("📧 Spam Email Classifier")

model, vectorizer = train_model()

user_input = st.text_area("Enter a message")

if st.button("Check"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)

        if prediction[0] == 1:
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")