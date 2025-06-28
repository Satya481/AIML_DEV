import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
ps= PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Keep only alphanumeric
    alnum_tokens = [i for i in tokens if i.isalnum()]
    
    # Remove stopwords and punctuation
    filtered_tokens = [i for i in alnum_tokens if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Stemming
    stemmed_tokens = [ps.stem(i) for i in filtered_tokens]
    
    return " ".join(stemmed_tokens)
        


        
        
model = pickle.load(open('model.pkl', 'rb'))
st.title("Spam Email/SMS Classifier")
st.text_input = st.text_area("Enter the email/SMS content here:")
if st.button('Predict'):
    #preprocess the input
    transformed_sms = transform_text(st.text_input)

    #vectorize the input
    vector_input=tfidf.transform([transformed_sms])

    #predict 
    result = model.predict(vector_input)[0]
    #display the result
    st.header("Prediction:")
    st.subheader("Spam" if result == 1 else "Not Spam")

