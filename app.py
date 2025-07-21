import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog and let the model predict!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="cat_dog_mobilenetv2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    st.subheader(f"Prediction: {label} ({prediction:.2f})")