import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("cat_dog_model.h5")

# Preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 256, 256, 3))
    return img_array

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model_1.predict(img_array)[0][0]

    if prediction < 0.5:
        st.write("### 🐱 It's a **Cat**!")
    else:
        st.write("### 🐶 It's a **Dog**!")