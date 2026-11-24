import streamlit as st
import numpy as np
from base64 import b64decode
from pathlib import Path
from hashlib import sha256
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, UnidentifiedImageError

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Plant Disease Detection System", page_icon="🌿", layout="wide")

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    /* Hide default menu, footer, and header padding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove default padding so title touches the top */
    .block-container { padding-top: 0rem !important; padding-bottom: 1rem !important; }
    
    /* Black Background */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* Top Branding (Solid Green, Top Right) */
    .top-brand { 
        text-align: right; color: #4caf50 !important; font-weight: bold; 
        font-size: 14px; margin-top: 10px; margin-right: 20px; letter-spacing: 1px;
        margin-bottom: -20px;
    }
    
    /* Main Title with Leaf */
    .header-container { 
        text-align: center; margin-bottom: 20px; margin-top: 0px; 
        display: flex; justify-content: center; align-items: center; gap: 10px;
    }
    .title-text { color: #4caf50 !important; font-size: 42px; font-weight: 900; margin: 0; letter-spacing: 2px;}
    .leaf-icon { font-size: 42px; }

    /* Style the Navigation Buttons (Pills) */
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        background-color: #121212;
        color: #ffffff;
        border: 1px solid #333;
        font-weight: bold;
        padding: 10px;
    }
    div.stButton > button:hover {
        border-color: #4caf50;
        color: #4caf50;
    }
    
    /* Model Dropdown */
    [data-testid="stSelectbox"] label { color: #ff9800 !important; font-weight: bold; font-size: 18px; }
    
    /* Custom Panels */
    .panel { background:#121212; border-radius:12px; padding:20px; margin-bottom:15px; color: #f0f0f0; box-shadow: 0 4px 6px rgba(255,255,255,0.05);}
    
    .prediction-label { color: #2196f3 !important; font-size: 16px; font-weight: bold; margin-bottom: 5px; } /* Blue */
    .disease-name { color: #FFD700 !important; font-size: 26px; font-weight: bold; margin: 0;} /* Yellow */
    .definition { color: #cccccc !important; font-size: 15px; font-style: italic; margin-top: 5px; margin-bottom: 10px;}
    .confidence { color: #4caf50 !important; font-size: 20px; font-weight: bold; margin-top: 10px;} /* Solid Green */
    .model-weight { color: #FFD700 !important; font-weight: bold; font-size: 16px; margin-top: 5px;} 
    
    /* Treatment & Cause Panels */
    .treat { background:#081526; border-left:5px solid #2196f3; margin-bottom: 15px;}
    .treat h3 { color:#64b5f6 !important; }
    .cause { background:#1a1408; border-left:5px solid #ff9800; }
    .cause h3 { color:#2196f3 !important; } /* Solid Blue text */
    
    /* Upload Button */
    [data-testid="stFileUploader"] button {
        background-color: #4caf50 !important; color: black !important;
        font-weight: bold !important; border: 1px solid #388e3c !important;
    }
    
    /* Solid Yellow Treatment List */
    .treat-list-item { color: #FFD700 !important; font-size: 16px; line-height: 1.6; margin-bottom: 8px;}
    
    /* Ensure all default text is white */
    p, h1, h2, h3, h4, h5, h6, span, li { color: #ffffff !important; }
    .leaf-rejection {
        color: #ff4d4f;
        background: #251313;
        border-left: 4px solid #ff4d4f;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & BRANDING ---
st.markdown('<div class="top-brand">Developed by Mugisha Olivier</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="header-container">
        <span class="leaf-icon">🌿</span>
        <p class="title-text">PLANT DISEASE DETECTION SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. TOP NAVIGATION & STATE MANAGEMENT ---
# This is the magic that stops the data from disappearing when you switch tabs!
if "page" not in st.session_state:
    st.session_state.page = "Diagnostics"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("1. Diagnostics", use_container_width=True):
        st.session_state.page = "Diagnostics"
with col2:
    if st.button("2. Guides", use_container_width=True):
        st.session_state.page = "Guides"

st.markdown("---")

# --- 5. DATA & LOGIC ---
CLASS_LIST = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Small binary classifier fitted to pooled MobileNetV2 features. It was trained
# on PlantDoc/PlantVillage leaf images and unrelated object, flower, scene and
# document examples. Store signed int8 coefficients inline for offline use.
LEAF_WEIGHTS = np.frombuffer(b64decode(
    "+cbY/Orv9s3mB+kXARjrEQPvAAbb3dn7GANMDDr4CT0UPenbMTrVAPDn0w/92vz77PTp+ecDzCTS49TnzOjd8A4Q"
    "+uXny9oIBw/23/LfA17oZfQKJNzRLBIhDQLi5fUR60gI7tflOgod3+Kh383a5Pb2Vlrd70MfUFMf7d8A5hAi3Ovm"
    "9PwwAfgo1C7O5Qb9++AOBsoQ/+nmBu/0bO0k5uw70sjrJ+TbUB75yArt7T8MO3H/8MtV1+0G+GEH++roHfbZ/8zd"
    "wt/SqvzmKOn/FwozAN4AI/4P+eLk2hbmxvkuw94REBD9wutCNxkb29v8CATnDdoRJObi7gr/+eEO/hT8yg7+1ObV"
    "CvLszRk0LtfuGK3pCwrP+A7fFN00D/3JMOMGTTzt4g78IPjk/evaC9YSBQIM3//W9xTIKOj897VKPvDdDgvQLf/6"
    "2AE78ufFbQ57OPIqDvHhAgf1/gg449go5hT+BBokVAJOMw76FNoQAhJFGe/j7h0a/+X+EeTyH+Fl0fVEDB/y6AH+"
    "2usFCgcZL8jrDSxgL0Hu79HX6+XN4cEG6RoW4ya1AvnOxe/UA/c+AAJVDfMPQjkTA0UFEegM4dDtA+sS3xsm+REa"
    "+dMN8SX9SioFD/c6/OnJE+8uFfIE0BnaQxH+8/npEebY8yT6IxnD5NdSs+n+7+/1JPQORvoE2Ako28oI3M3wKfz9"
    "2eXn9/DwDiwK4eoFoOb4GB8TDi0aC+r5vOqBEAkr6yv1+etA7NsX6/JU7zHa6ScOMBjSK/1L5e8e/AQrAwbpDcv0"
    "/Afz/h3m5rkZFgX/2g7bECDg/d8U6PT/2/P1HyT/3fwYzfjv7gv+R/L+3/sp6fcB7C7P4goSA/AHGhoDCP7u+/sl"
    "zRcAEgHg6PhXCeoR9ywHBunr9TRSKeL9DfLoEdbj+dve9+rO1TwP3gVNBdYO/s4Y1RTN8vja0zI8B99jTQVtBgAl"
    "COlOC9XGwgDO9Rf1JggAH/Hd4eL9Bkch8wgR6BkMOxA4QPw8MvYPLzMe4ucX9b7/898jJBob8gPI8v0EJiLdFQUh"
    "Q+DJ3cziDeMNNNL8OT0AJg+0r8bgDBry1uEaBrsf1ffh8t1sBO/2HrAGhwu9xQAq2fgOAxoa2s/dFNbh1vJ0zjro"
    "FcVJ3iHi6BDx/AUq6lU59Ro07+r2tPnsxN4H9u3lFCck5hoh9isa0dTr/AYX5A0E6AfsJvrc+woQGgcgtQfRP879"
    "B+T12tYat+IQFhzp7bAJCWdQ3TLfu/5v1dX13yQh5/r/5y4DGLoHAyQE5uskH/wKATvR8+UD7Q3/Y2cQCtPR2i0G"
    "3//038ne2P/19f8T+v/XUOgNCQ8T69XTxBUr/Ffx1RXJCP3x+AIbMNT2EEj/99AP5Pa66OTl6RIlPfzqEg7UKvXf"
    "RUrjSarTBbf++uEMYfEVzQPvGuj1YeUMBtHwDR76ofvvIh/43GTfIBkb7CrwQE0BxgTc5wf299Xd3Qcx2uP5ywr4"
    "49/REJ446gcHBCIWTvEoR/D6HczE+Bf94f+f8jLx+RT89gUA/S3d1v0p9/vr6RPkFegcW8s2BxTsBOrENgAIGSM6"
    "ENwlEN7h5we67t7q3wfi3Ov24QP77exK8M8M9AEP2T0H9e3huOMTArjzDw0R7hO8Hys+Svz1PNwX+RL2BfP07AwO"
    "rizs7fz57Cn6wtvoaOfsH0T7AOnf5+L/HBs="
), dtype=np.int8).astype(np.float32) * 0.03763507039416805
LEAF_BIAS = -4.13353426637332
LEAF_CUTOFF = 1.268064783194852


def prepare_mobile_model(path):
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    pool = next((layer for layer in model.layers
                 if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)), None)
    if pool is None or pool.output.shape[-1] != len(LEAF_WEIGHTS):
        raise ValueError("The leaf check requires the project's original MobileNetV2 model.")
    encoder = tf.keras.Model(model.inputs[0], pool.output)

    def leaf_features(batch):
        return encoder(batch, training=False)

    def disease_scores(batch):
        return model(batch, training=False)

    signature = [tf.TensorSpec((1, 224, 224, 3), tf.float32)]
    try:
        fast_features = tf.function(leaf_features, input_signature=signature, autograph=False)
        fast_scores = tf.function(disease_scores, input_signature=signature, autograph=False)
        sample = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        fast_features(sample).numpy()
        fast_scores(sample).numpy()
    except (ValueError, RuntimeError, tf.errors.OpError):
        return leaf_features, disease_scores
    return fast_features, fast_scores


def prepare_disease_model(path):
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)

    def disease_scores(batch):
        return model(batch, training=False)

    signature = [tf.TensorSpec((1, 224, 224, 3), tf.float32)]
    try:
        fast_scores = tf.function(disease_scores, input_signature=signature, autograph=False)
        fast_scores(tf.zeros((1, 224, 224, 3), dtype=tf.float32)).numpy()
    except (ValueError, RuntimeError, tf.errors.OpError):
        return disease_scores
    return fast_scores


@st.cache_resource
def model_loader():
    # Prepare one model at a time to limit CPU and memory pressure on the host.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-warmup")


@st.cache_resource
def prepare_in_background(path, file_version):
    # Including file_version keeps a replaced model from reusing an old future.
    prepare = (
        prepare_mobile_model
        if Path(path).name == "MobileNetV2_model.h5"
        else prepare_disease_model
    )
    return model_loader().submit(prepare, path)


def is_leaf_photo(encoder, image):
    # Preserve the full scene so that background foliage in a portrait does
    # not become the subject of a crop. Gray padding matches gate training.
    preview = image.copy()
    preview.thumbnail((224, 224))
    if np.asarray(preview, dtype=np.float32).std(axis=(0, 1)).max() < 4.0:
        return False
    view = ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR,
                        color=(128, 128, 128))
    batch = np.asarray(view, dtype=np.float32)[None, ...] / 127.5 - 1.0
    features = encoder(batch).numpy()[0]
    length = np.linalg.norm(features)
    if not np.isfinite(length) or length == 0:
        return False
    score = np.dot(features / length, LEAF_WEIGHTS) + LEAF_BIAS
    return bool(score >= LEAF_CUTOFF)


def remember_uploads():
    st.session_state.uploaded_files = list(st.session_state.leaf_uploader or [])


TREATMENTS = {
    "Apple___Apple_scab": {"def": "A fungal disease causing olive-green to black velvety spots on leaves.", "cause": "Venturia inaequalis (fungus)", "steps": ["Apply fungicides early in spring.", "Rake and destroy fallen leaves.", "Prune trees for airflow."]},
    "Apple___Black_rot": {"def": "A fungal disease causing frogeye leaf spots and rotting black fruit.", "cause": "Botryosphaeria obtusa (fungus)", "steps": ["Prune out dead wood.", "Remove mummified fruits.", "Apply fungicides."]},
    "Apple___Cedar_apple_rust": {"def": "A fungal disease causing bright orange-yellow spots on leaves.", "cause": "Gymnosporangium juniperi-virginianae (fungus)", "steps": ["Remove nearby cedar trees.", "Apply fungicides in spring.", "Plant resistant varieties."]},
    "Apple___healthy": {"def": "The apple leaf is healthy with no signs of infection.", "cause": None, "steps": ["Maintain regular watering.", "Monitor for pests."]},
    "Blueberry___healthy": {"def": "The blueberry leaf is healthy and free of disease.", "cause": None, "steps": ["Ensure proper soil pH.", "Maintain hygiene."]},
    "Cherry_(including_sour)___Powdery_mildew": {"def": "A fungal disease creating a white powdery coating on leaves.", "cause": "Podosphaera clandestina (fungus)", "steps": ["Apply sulfur sprays.", "Prune for air circulation.", "Avoid excess nitrogen."]},
    "Cherry_(including_sour)___healthy": {"def": "The cherry leaf is healthy.", "cause": None, "steps": ["Maintain good hygiene.", "Monitor for pests."]},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"def": "A fungal disease causing long, narrow, rectangular gray-to-brown lesions.", "cause": "Cercospora zeae-maydis (fungus)", "steps": ["Use resistant hybrids.", "Rotate crops.", "Apply foliar fungicides."]},
    "Corn_(maize)___Common_rust_": {"def": "A fungal disease producing reddish-brown powdery pustules on leaves.", "cause": "Puccinia sorghi (fungus)", "steps": ["Plant resistant varieties.", "Apply fungicides early.", "Ensure plant spacing."]},
    "Corn_(maize)___Northern_Leaf_Blight": {"def": "A fungal disease causing cigar-shaped gray-green lesions.", "cause": "Exserohilum turcicum (fungus)", "steps": ["Plant resistant hybrids.", "Rotate crops.", "Apply fungicides."]},
    "Corn_(maize)___healthy": {"def": "The corn leaf is healthy.", "cause": None, "steps": ["Maintain soil fertility.", "Ensure proper irrigation."]},
    "Grape___Black_rot": {"def": "A fungal disease causing brown spots on leaves and rotting black fruit.", "cause": "Guignardia bidwellii (fungus)", "steps": ["Remove infected berries.", "Apply fungicides from bud break.", "Keep vines pruned."]},
    "Grape___Esca_(Black_Measles)": {"def": "A complex fungal disease causing leaf discoloration and fruit spotting.", "cause": "Phaeomoniella chlamydospora (fungus)", "steps": ["Remove infected wood.", "Avoid pruning in wet weather.", "Apply wound protectants."]},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"def": "A fungal disease causing irregular dark brown spots on leaves.", "cause": "Pseudocercospora vitis (fungus)", "steps": ["Apply copper fungicides.", "Improve canopy airflow.", "Remove fallen leaves."]},
    "Grape___healthy": {"def": "The grape leaf is healthy.", "cause": None, "steps": ["Maintain regular pruning.", "Monitor for pests."]},
    "Orange___Haunglongbing_(Citrus_greening)": {"def": "A devastating bacterial disease causing blotchy mottle and yellow shoots.", "cause": "Candidatus Liberibacter (bacteria)", "steps": ["Remove infected trees.", "Control citrus psyllid.", "Plant disease-free stock."]},
    "Peach___Bacterial_spot": {"def": "A bacterial disease causing small, dark, water-soaked spots that fall out.", "cause": "Xanthomonas arboricola (bacteria)", "steps": ["Apply copper bactericide.", "Prune to reduce humidity.", "Avoid overhead watering."]},
    "Peach___healthy": {"def": "The peach leaf is healthy.", "cause": None, "steps": ["Thin fruits.", "Prune for open center."]},
    "Pepper,_bell___Bacterial_spot": {"def": "A bacterial disease causing dark, water-soaked spots on leaves.", "cause": "Xanthomonas campestris (bacteria)", "steps": ["Use pathogen-free seeds.", "Apply copper sprays.", "Rotate crops."]},
    "Pepper,_bell___healthy": {"def": "The pepper leaf is healthy.", "cause": None, "steps": ["Maintain consistent watering.", "Monitor for aphids."]},
    "Potato___Early_blight": {"def": "A fungal disease causing dark spots with concentric rings on older leaves.", "cause": "Alternaria solani (fungus)", "steps": ["Apply chlorothalonil.", "Rotate crops.", "Remove infected leaves."]},
    "Potato___Late_blight": {"def": "A highly destructive disease causing large, greasy, black lesions.", "cause": "Phytophthora infestans (water mold)", "steps": ["Apply fungicides immediately.", "Destroy infected plants.", "Avoid overhead irrigation."]},
    "Potato___healthy": {"def": "The potato leaf is healthy.", "cause": None, "steps": ["Use certified seeds.", "Hill soil around stems."]},
    "Raspberry___healthy": {"def": "The raspberry leaf is healthy.", "cause": None, "steps": ["Prune canes after harvest.", "Ensure drainage."]},
    "Soybean___healthy": {"def": "The soybean leaf is healthy.", "cause": None, "steps": ["Rotate crops.", "Maintain weed control."]},
    "Squash___Powdery_mildew": {"def": "A fungal disease creating a white powdery coating on leaves.", "cause": "Podosphaera xanthii (fungus)", "steps": ["Apply neem oil.", "Plant in full sun.", "Ensure air circulation."]},
    "Strawberry___Leaf_scorch": {"def": "A fungal disease causing purple spots and browning leaf edges.", "cause": "Diplocarpon earlianum (fungus)", "steps": ["Remove infected leaves.", "Apply fungicides early.", "Avoid overhead watering."]},
    "Strawberry___healthy": {"def": "The strawberry leaf is healthy.", "cause": None, "steps": ["Renovate beds after harvest.", "Maintain spacing."]},
    "Tomato___Bacterial_spot": {"def": "A bacterial disease causing small, dark, water-soaked spots.", "cause": "Xanthomonas campestris (bacteria)", "steps": ["Apply copper sprays.", "Avoid working when wet.", "Rotate crops."]},
    "Tomato___Early_blight": {"def": "A fungal disease causing dark concentric rings on lower leaves.", "cause": "Alternaria solani (fungus)", "steps": ["Apply chlorothalonil.", "Remove lower infected leaves.", "Mulch to prevent splash."]},
    "Tomato___Late_blight": {"def": "A destructive water mold causing large greasy black lesions.", "cause": "Phytophthora infestans (water mold)", "steps": ["Apply fungicides immediately.", "Destroy infected plants.", "Avoid overhead watering."]},
    "Tomato___Leaf_Mold": {"def": "A fungal disease causing pale green spots on top and olive mold underneath.", "cause": "Passalora fulva (fungus)", "steps": ["Improve air circulation.", "Reduce humidity.", "Apply fungicides."]},
    "Tomato___Septoria_leaf_spot": {"def": "A fungal disease causing small circular spots with gray centers.", "cause": "Septoria lycopersici (fungus)", "steps": ["Remove infected leaves.", "Apply fungicides.", "Mulch around base."]},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"def": "Pest infestation causing yellow stippling and fine webbing on leaves.", "cause": "Tetranychus urticae (mite)", "steps": ["Apply insecticidal soap.", "Increase humidity.", "Remove heavily infested leaves."]},
    "Tomato___Target_Spot": {"def": "A fungal disease causing brown spots with yellow halos.", "cause": "Corynespora cassiicola (fungus)", "steps": ["Apply fungicides.", "Improve airflow.", "Remove infected debris."]},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"def": "A viral disease causing upward curling, yellowing, and stunting.", "cause": "Begomovirus (spread by whiteflies)", "steps": ["Control whiteflies.", "Remove infected plants.", "Use resistant varieties."]},
    "Tomato___Tomato_mosaic_virus": {"def": "A viral disease causing mottled light and dark green patterns.", "cause": "Tobacco mosaic virus", "steps": ["Remove infected plants.", "Disinfect tools.", "Wash hands."]},
    "Tomato___healthy": {"def": "The tomato leaf is healthy.", "cause": None, "steps": ["Maintain even watering.", "Stake plants."]}
}

def get_confidence_word(conf):
    if conf > 90: return "Very High"
    elif conf > 70: return "High"
    elif conf > 50: return "Moderate"
    else: return "Low"

# --- 6. MAIN PAGE RENDER ---
if st.session_state.page == "Guides":
    st.markdown("## 📖 How to Use & Model Guide")
    st.write("Welcome to the Plant Disease Detection System! This application uses Convolutional Neural Networks (CNNs) to identify crop diseases and provide actionable treatments.")
    
    st.markdown("### 🧠 The Models")
    st.write("""
    We have integrated 4 different AI architectures for you to choose from:
    *   **MobileNetV2 (The Default):** This model is incredibly lightweight (13 MB). It was specifically chosen for this app because it runs extremely fast on mobile devices without sacrificing accuracy. This allows farmers in the field to get instant results on their phones.
    *   **ResNet50:** A classic, deep 50-layer network (96 MB). It is very powerful but requires more computational power.
    *   **EfficientNetB0:** A highly optimized modern model (16 MB) that balances accuracy and efficiency beautifully.
    *   **DenseNet121:** A densely connected network (31 MB) that reuses features for robust learning.
    """)
    
    st.markdown("### 📸 Step-by-Step Guide")
    st.markdown("""
    1.  **Select a Model:** Choose an AI architecture from the dropdown (MobileNetV2 is recommended for speed).
    2.  **Upload Images:** Click the green upload button to select one or multiple plant leaf images.
    3.  **Leaf check:** Use a close-up photo with one leaf as the main subject. Unclear photos are skipped.
    4.  **Review Results:** For accepted photos, the selected model suggests a match among its supported crop and disease classes. A weak match is marked uncertain.
    """)
    
    st.info("💡 **Pro Tip:** For the best accuracy, make sure the leaf is clearly visible, well-lit, and takes up most of the image.")

elif st.session_state.page == "Diagnostics":
    
    # Top Controls for Diagnostics
    col_model, col_empty = st.columns([1, 2])
    with col_model:
        selected_model_name = st.selectbox(
            "Select AI Model", 
            ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]
        )
        sizes = {"MobileNetV2": "13 MB", "ResNet50": "96 MB", "EfficientNetB0": "16 MB", "DenseNet121": "31 MB"}
        st.markdown(f'<p class="model-weight">Model Weight: {sizes[selected_model_name]}</p>', unsafe_allow_html=True)

    st.file_uploader(
        "Upload a close-up leaf photo (offline leaf check)",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        key="leaf_uploader", on_change=remember_uploads,
    )
    uploaded_files = st.session_state.uploaded_files
    uploads = [(sha256(file.getbuffer()).digest(), file) for file in (uploaded_files or [])]
    # Keep results only for the photos in this session's upload list.
    previous = st.session_state.get("prediction_cache", {})
    prediction_cache = {key: previous.get(key, {}) for key, _ in uploads}
    st.session_state.prediction_cache = prediction_cache
    mobile_path = Path(__file__).resolve().with_name("MobileNetV2_model.h5")
    try:
        mobile_ready = prepare_in_background(str(mobile_path), mobile_path.stat().st_mtime_ns)
    except OSError:
        mobile_ready = None

    selected_ready = None
    if mobile_ready is not None and selected_model_name != "MobileNetV2":
        model_path = mobile_path.with_name(f"{selected_model_name}_model.h5")
        try:
            selected_ready = prepare_in_background(str(model_path), model_path.stat().st_mtime_ns)
        except OSError:
            pass

    if uploaded_files:
        try:
            if mobile_ready is None:
                raise FileNotFoundError(mobile_path.name)
            with st.spinner("Loading MobileNetV2…"):
                leaf_check, predict_mobile = mobile_ready.result()
        except Exception as exc:
            st.error("Leaf checking is unavailable. Place the original MobileNetV2_model.h5 in the same folder as this app.")
            st.caption(str(exc))
            st.stop()

    if uploaded_files:
        for key, uploaded_file in uploads:
            cached = prediction_cache[key]
            st.markdown("---") 
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.markdown(f"#### Image: {uploaded_file.name}")
                try:
                    with Image.open(BytesIO(uploaded_file.getvalue())) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                    st.warning("This file could not be read. Try a JPG or PNG photo.")
                    continue
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### 🔬 Diagnostic Results")
                
                if "leaf" not in cached:
                    with st.spinner("Checking for a leaf…"):
                        try:
                            cached["leaf"] = is_leaf_photo(leaf_check, image)
                        except Exception as exc:
                            st.error("Leaf checking failed for this image. Diagnosis skipped.")
                            st.caption(str(exc))
                            continue
                if not cached["leaf"]:
                    st.markdown(
                        '<div class="leaf-rejection" role="alert">'
                        "I don't recognize this as a plant leaf.</div>",
                        unsafe_allow_html=True,
                    )
                    continue

                with st.spinner("Checking the leaf condition…"):
                    predictions = cached.get(selected_model_name)
                    if predictions is None:
                        img = image.resize((224, 224), Image.Resampling.NEAREST)
                        img_array = np.array(img, dtype=np.float32)
                    
                        if "MobileNet" in selected_model_name:
                            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                            img_array = preprocess_input(img_array)
                        elif "ResNet" in selected_model_name:
                            from tensorflow.keras.applications.resnet50 import preprocess_input
                            img_array = preprocess_input(img_array)
                        elif "Efficient" in selected_model_name:
                            from tensorflow.keras.applications.efficientnet import preprocess_input
                            img_array = preprocess_input(img_array)
                        elif "Dense" in selected_model_name:
                            from tensorflow.keras.applications.densenet import preprocess_input
                            img_array = preprocess_input(img_array)
                    
                        img_array = np.expand_dims(img_array, axis=0)
                    
                        if selected_model_name == "MobileNetV2":
                            predictions = predict_mobile(img_array).numpy()[0]
                        else:
                            if selected_ready is None:
                                st.markdown(
                                    '<div class="leaf-rejection" role="alert">'
                                    f'{selected_model_name} is unavailable. Add '
                                    f'<code>{selected_model_name}_model.h5</code> next to '
                                    '<code>app.py</code> and redeploy.</div>',
                                    unsafe_allow_html=True,
                                )
                                continue
                            try:
                                predict_selected = selected_ready.result()
                            except Exception as exc:
                                st.error(f"Error loading model: {exc}")
                                st.stop()
                            predictions = predict_selected(img_array).numpy()[0]
                    if predictions.shape != (len(CLASS_LIST),) or not np.isfinite(predictions).all():
                        st.error("The selected model returned invalid scores. Diagnosis skipped.")
                        continue
                    cached[selected_model_name] = predictions
                    best_index = np.argmax(predictions)
                    predicted_class = CLASS_LIST[best_index]
                    confidence = predictions[best_index] * 100
                    
                    display_name = predicted_class.replace("___", " - ").replace("_", " ")
                    conf_word = get_confidence_word(confidence)
                    
                    if confidence < 50:
                        st.warning("Leaf detected. The model is uncertain about its condition. Try a sharper photo of the affected area.")
                        continue

                    info = TREATMENTS.get(predicted_class, {"def": "No definition.", "cause": "Unknown.", "steps": ["No data."]})
                    is_healthy = info["cause"] is None
                        
                    # Prediction Panel (Green Confidence)
                    st.markdown(f'''
                        <div class="panel">
                            <p class="prediction-label">Prediction:</p>
                            <p class="disease-name">{display_name}</p>
                            <p class="definition">{info['def']}</p>
                            <p class="confidence">Confidence: {conf_word} ({confidence:.2f}%)</p>
                        </div>
                    ''', unsafe_allow_html=True)
                        
                    # Recommended Action Plan
                    treatment_html = "".join([f"<div class='treat-list-item'>💊 {t}</div>" for t in info["steps"]])
                    st.markdown(f'''
                        <div class="panel treat">
                            <h3>Recommended Action Plan</h3>
                            <div>{treatment_html}</div>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Cause Panel
                    if not is_healthy:
                        st.markdown(f'''
                            <div class="panel cause">
                                <h3>Cause of Disease</h3>
                                <p style="color: #ffffff !important;">{info["cause"]}</p>
                            </div>
                        ''', unsafe_allow_html=True)
    else:
        st.info("Waiting for image upload to start diagnosis...")