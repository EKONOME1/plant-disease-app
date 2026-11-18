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
        <p class="title-text">PLANT DISEASE DETECTOR</p>
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
