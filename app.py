import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("mnist_model.h5")

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = image.astype("float32") / 255.0  # Normalize (0-1)
    image = image.reshape(1, 784)  # Flatten to match model input
    return image

# Streamlit UI
st.title("MNIST Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image",  width=150)  # Updated to use_container_width
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)
    
    # Display the predicted digit
    st.markdown(f"<h2 style='text-align: center;'>Predicted Digit: {digit}</h2>", unsafe_allow_html=True)
