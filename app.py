# 
import streamlit as st
import numpy as np
import tensorflow.lite as tflite
from PIL import Image

# Load TFLite model
model_path = r"C:\Users\medhi\e_waste\flask-project\model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Streamlit UI
st.title("E-Waste Scanner")
st.write("Upload an image to classify if it's an e-waste item.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    st.write("Prediction:", prediction)
