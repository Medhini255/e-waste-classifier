import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to classify image
def classify_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    prediction = np.argmax(output_data)
    labels = ["Keyboard", "Laptop_mobile", "Laptop_mouse", "Mobile", "Monitor", "Mouse"]
    return labels[prediction]

st.title("E-Waste Image Classifier")

# File upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = classify_image(image)
    st.write(f"Prediction: **{prediction}**")

# API Endpoint
if "file" in st.query_params:
    file = st.query_params["file"]
    image = Image.open(io.BytesIO(file))
    prediction = classify_image(image)
    st.json({"prediction": prediction})
