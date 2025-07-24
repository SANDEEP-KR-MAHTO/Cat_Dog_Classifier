# 🐶🐱 Cat vs Dog Classifier - Deep Learning Project

This project is an **end-to-end deep learning image classification system** that distinguishes between cats and dogs using **transfer learning** with **MobileNetV2**. It includes everything from data preprocessing and model training to deployment as a live **Streamlit app**.

## 🚀 Project Highlights

- ✅ Achieved **95%+ validation accuracy**
- 📦 Used **MobileNetV2** for efficient and fast feature extraction
- 🧠 Fine-tuned model with a small learning rate for improved performance
- 🖼️ Accepts image uploads for real-time prediction
- 🌐 Deployed using **Streamlit**

## 📂 Project Structure

- Cat_Dog_Classifier/
- ├── app.py # Streamlit web app
- ├── cat_dog_model.pkl # Saved model (Pickle format)
- ├── model.h5 # Saved model (Keras format)
- ├── requirements.txt # Python dependencies
- ├── README.md # Project documentation
- └── sample_images/ # Example test images



## 📸 Demo

Try the live app:  
👉 [https://catdogclassifier.streamlit.app](https://catdogclassifier.streamlit.app)

## 🛠️ Installation

1. **Clone the repository:**

git clone https://github.com/SANDEEP-KR-MAHTO/Cat_Dog_Classifier.git cd Cat_Dog_Classifier

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pre-trained on ImageNet)

- Layers Added:

    - GlobalAveragePooling2D

    - Dense(64, activation='relu')

    - Dropout

    - Dense(1, activation='sigmoid')

## 📊 Results
Metric	Value
Accuracy	95%+
Loss	Low
Overfitting	Controlled via dropout & fine-tuning

## 📌 Requirements
- tensorflow
- keras
- numpy
- pillow
- matplotlib
- streamlit


## 🤝 Acknowledgements
- TensorFlow & Keras

- Streamlit for deployment

- Kaggle for dataset inspiration

