import streamlit as st
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data (if necessary)
nltk.download('stopwords')
nltk.download('punkt')

# Function to create a word cloud from text
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Streamlit app
st.title("Textual Data Analysis with Streamlit")

# Upload file (or load predefined dataset)
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Show the first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(df.head())
    
    # Assume there is a 'text' column in the dataset containing textual data
    if 'text' in df.columns:
        # Text preprocessing: Remove NaN and combine text into one string
        text_data = df['text'].dropna().str.cat(sep=' ')
        
        # Display word cloud
        st.subheader("Word Cloud of Text Data")
        generate_wordcloud(text_data)
        
        # Show most frequent words using CountVectorizer
        vectorizer = CountVectorizer(stop_words='english', max_features=10)
        word_count = vectorizer.fit_transform(df['text'].dropna())
        word_freq = pd.DataFrame(word_count.toarray(), columns=vectorizer.get_feature_names_out()).sum(axis=0).sort_values(ascending=False)
        
        st.subheader("Top 10 Frequent Words")
        st.bar_chart(word_freq.head(10))
        
    else:
        st.error("No 'text' column found in the dataset.")

           