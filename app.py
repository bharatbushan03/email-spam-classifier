import streamlit as st
import pickle
import preprocessing
import nltk
st.title("Email/SMS Spam classifier")

text = st.text_area("Message", placeholder = "Paste the email/sms text here")

model = pickle.load(open(r'model.pkl', 'rb'))
vectorize = pickle.load(open(r'vectorize.pkl', 'rb'))

if st.button("Predict"):
    tokenized = preprocessing.preprocessing_text(text)
    vectors = vectorize.transform([tokenized]).toarray()
    prediction = model.predict(vectors)

    if prediction == 1:
        st.write("Email/SMS is spam")
    else:
        st.write("Email/SMS is not spam")
