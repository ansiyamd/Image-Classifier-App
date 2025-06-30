
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Image Classifier", layout="centered")

# Load a sample pre-trained model (assumed to be saved as 'model.h5')
@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()

# Class labels (adjust based on your model)
class_names = ['Cat', 'Dog']

# Title
st.markdown("<h1 style='text-align: center; color: #2196F3;'>📷 Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("### Upload an image and see what it is!")

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.write("This app uses a CNN model to classify images into categories.")
    st.markdown("**Supported Classes:**")
    for label in class_names:
        st.markdown(f"- {label}")

# Upload image
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index] * 100

    # Output
    st.markdown(
        f"<div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;'>"
        f"<h2>Prediction: {class_names[class_index]} 🧠</h2>"
        f"<p>Confidence: {confidence:.2f}%</p>"
        f"</div>",
        unsafe_allow_html=True
    )
