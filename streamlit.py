#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image

# ----------------- Load Required Resources -----------------

# Load tokenizer


# Load trained captioning model
model = load_model('caption_model.keras')
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

# Load the CNN encoder model (InceptionV3 up to the last pooling layer)
def build_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(base_model.input, base_model.layers[-2].output)
    return model

cnn_model = build_feature_extractor()

# Set max length (based on training)
max_length = 38  # Replace with your own value

# ----------------- Feature Extraction -----------------

def extract_features(image):
    image = image.resize((299, 299))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array, verbose=0)
    return features

# ----------------- Caption Generation -----------------

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)[0]  # Get the actual sequence array
        sequence = np.reshape(sequence, (1, max_length))  # Reshape to match model's expected input
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += f' {word}'
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title(" Image Captioning App")
st.markdown("Upload an image and get an AI-generated description using InceptionV3 + LSTM")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Analyzing image and generating caption..."):
            photo = extract_features(image)
            caption = generate_caption(model, tokenizer, photo, max_length)
        st.success("Caption generated:")
        st.markdown(f"** {caption}**")

