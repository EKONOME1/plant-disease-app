import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")

# --- 2. CUSTOM PROFESSIONAL HEADER ---
st.markdown("""
    <style>
    .header-bar {
        background-color: #1B4332; 
        padding: 15px 30px;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 18px;
        font-weight: 500;
        letter-spacing: 1.5px;
        margin: 0;
    }
    .header-owner {
        color: #95D5B2; 
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 14px;
        margin: 0;
    }
    </style>
    
    <div class="header-bar">
        <p class="header-title">🌿 PLANT DISEASE DETECTION SYSTEM</p >
        <p class="header-owner">&copy; MUGISHA OLIVIER</p >
    </div>
    """, unsafe_allow_html=True)

# --- 3. LOAD CNN MODEL (Works Locally and on Cloud) ---
@st.cache_resource
def load_model():
    try:
        # Smart path detection: Checks if you are on your laptop, otherwise uses cloud path
        if os.path.exists(r"E:\PLANT-DISEASE\MobileNetV2_Perfect.keras"):
            model_path = r"E:\PLANT-DISEASE\MobileNetV2_Perfect.keras"
        else:
            model_path = "MobileNetV2_Perfect.keras"
            
        model = tf.keras.models.load_model(model_path)
        
        classes = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]
        return model, classes
    except Exception as e:
        st.error(f"🔴 SYSTEM ERROR: {str(e)}")
        return None, None

model, classes = load_model()

# --- 4. TREATMENT DESCRIPTIONS ---
TREATMENTS = {
    "Tomato___Late_blight": {
        "steps": ["Remove and destroy infected leaves immediately.", "Apply copper-based fungicide.", "Avoid overhead watering to keep leaves dry."],
        "explanation": "A highly destructive disease causing large, greasy, brown-black patches on leaves and fruit. It can destroy a crop in days during cool, wet weather."
    },
    "Tomato___Early_blight": {
        "steps": ["Rotate crops to prevent soil reinfection.", "Apply chlorothalonil or copper fungicide.", "Remove plant debris."],
        "explanation": "A fungal disease causing concentric dark rings on lower leaves. It thrives in warm, humid conditions and can spread to the fruit."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "steps": ["Rotate crops with non-host plants.", "Apply foliar fungicide.", "Plant resistant hybrids."],
        "explanation": "This fungal disease causes long, cigar-shaped gray-green lesions on leaves, destroying the plant's ability to photosynthesize."
    },
    "default": {
        "steps": ["Remove and destroy infected plant parts.", "Improve air circulation around plants.", "Consult a local agricultural extension."],
        "explanation": "A disease affecting the plant's health. Proper sanitation and airflow are key to prevention."
    }
}

# --- 5. MAIN UI ---
st.write("Upload a leaf image to detect the disease and receive immediate treatment advice.")

uploaded_file = st.file_uploader("Upload a leaf image (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    with col2:
        if model is None:
            st.warning("⚠️ Model not loaded. Please check the error message above.")
        else:
            with st.spinner("Analyzing leaf..."):
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)[0]
                best_index = np.argmax(predictions)
                predicted_class = classes[best_index]
                confidence = predictions[best_index] * 100
                
                display_name = predicted_class.replace("___", " - ").replace("_", " ")
                
                st.success(f"**CNN Prediction:** {display_name}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                st.divider()
                
                info = TREATMENTS.get(predicted_class, TREATMENTS["default"])
                
                st.subheader("💊 Recommended Treatment")
                for step in info["steps"]:
                    st.markdown(f"- {step}")
                
                st.subheader("📖 Explanation & Prevention")
                st.write(info["explanation"])