import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image
from utils.preprocess import preprocess_image_forMNIST

st.set_page_config(
    page_title="MNIST CNN", 
    page_icon="🐸"
    )

st.title("MNIST Classifier (CNN)")
st.write("Draw a digit (0–9) and click **Predict**")

model = torch.jit.load("models/mnist_scripted_model.pt", map_location="cpu")
model.eval()  #추론모드로 전환


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=17,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear"):
        st.rerun()

with col2:
    predict_btn = st.button("Predict")

# 예측
if predict_btn:
    img = canvas_result.image_data

    # RGBA → grayscale
    img = img[:, :, 0]
        
    processed = preprocess_image_forMNIST(img)
        
    with torch.no_grad():
        input = processed.float()
        prediction = model(input) 
        result = torch.argmax(prediction, 1).item()
        st.success(f"Prediction: {result}")
        
              
    processed_img = processed.squeeze().numpy()
    st.subheader("Processed 28×28 Input")
    st.image(processed_img, width=150, clamp=True)
    
        



