
import streamlit as st
import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Configure Streamlit page UI
st.set_page_config(page_title="Pneumonia Detection App", layout="centered")
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a DICOM (.dcm) or standard image (PNG/JPG) to classify the image into one of three classes:")
st.write("**Normal**, **No Lung Opacity / Not Normal**, or **Lung Opacity (Pneumonia)**.")

# Map output indices to human readable names
class_mapping = {0: 'Normal', 1: 'No Lung Opacity / Not Normal', 2: 'Lung Opacity'}

@st.cache_resource
def load_champion_model():
    model_path = 'best_pneumonia_vgg16_model.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found in the root directory. Please upload it!")
        return None

model = load_champion_model()

def preprocess_image(img_data):
    # Resize to match model input resolution
    resized_img = cv2.resize(img_data, (128, 128))
    # Normalize intensity to [0, 1]
    normalized_img = resized_img.astype(np.float32) / 255.0
    # Replicate channels to match VGG16 3-channel layout
    stacked_img = np.stack([normalized_img, normalized_img, normalized_img], axis=-1)
    return np.expand_dims(stacked_img, axis=0)

uploaded_file = st.file_uploader("Choose a medical scan file...", type=["dcm", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Process file based on extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'dcm':
        try:
            # Read DICOM file metadata and pixel array
            dicom = pydicom.dcmread(uploaded_file)
            raw_pixels = dicom.pixel_array
            # Map scale to 0-255 range
            if raw_pixels.max() > 255:
                raw_pixels = (raw_pixels / raw_pixels.max() * 255).astype(np.uint8)
            img_display = raw_pixels
        except Exception as e:
            st.error(f"Error parsing DICOM: {e}")
            img_display = None
    else:
        try:
            # Read standard image
            image = Image.open(uploaded_file).convert('L')
            img_display = np.array(image)
        except Exception as e:
            st.error(f"Error parsing image: {e}")
            img_display = None

    if img_display is not None:
        # Display the image
        st.image(img_display, caption="Uploaded Chest X-Ray Scan (Grayscale)", use_container_width=True)
        
        # Prepare and predict
        if model is not None:
            with st.spinner("Running Deep Learning Classifier Inference..."):
                processed_tensor = preprocess_image(img_display)
                predictions = model.predict(processed_tensor)
                predicted_class_idx = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_class_idx]
                
            # Print results
            st.subheader("Classification Results")
            predicted_label = class_mapping[predicted_class_idx]
            
            if predicted_class_idx == 2:
                st.error(f"🚨 **Detection Outcome:** {predicted_label} (Pneumonia Detected)")
            elif predicted_class_idx == 1:
                st.warning(f"⚠️ **Detection Outcome:** {predicted_label}")
            else:
                st.success(f"✅ **Detection Outcome:** {predicted_label} (Healthy)")
                
            st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
            
            # Render confidence bar chart
            st.write("**Class Probability Distribution:**")
            for idx, label in class_mapping.items():
                prob = predictions[0][idx]
                st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")
