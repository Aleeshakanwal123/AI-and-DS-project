import streamlit as st
from transformers import pipeline

# Load the GPT-2 model for text generation
model_name = "gpt2"
generator = pipeline("text-generation", model=model_name)

# Title of the app
st.title("Text Generation with GPT-2")

# Input text area
input_text = st.text_area("Enter a prompt for text generation:")

# Slider for controlling the number of words generated
max_length = st.slider("Select maximum length of generated text:", min_value=10, max_value=100, value=50)

# Button for generating text
if st.button("Generate"):
    if input_text:
        try:
            # Generate text based on the input
            generated_text = generator(input_text, max_length=max_length, num_return_sequences=1)[0]['generated_text']
            # Display the result
            st.write("Generated Text:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"Error during text generation: {str(e)}")
    else:
        st.warning("Please enter a prompt for text generation.")

# Clear button
if st.button("Clear"):
    st.session_state.input_text = ""
