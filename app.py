# 
import streamlit as st
import numpy as np
import tensorflow.lite as tflite
from PIL import Image

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ["Keyboard", "Laptop_mobile", "Laptop_mouse" ,"Mobile", "Monitor","Mouse"]

# Streamlit UI
st.title("E-Waste Scanner")
st.write("Upload an image to classify if it's an e-waste item.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Adjust based on model input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class
    predicted_class = np.argmax(predictions)

    # Display result
    st.write(f"### Prediction: **{class_labels[predicted_class]}**")
