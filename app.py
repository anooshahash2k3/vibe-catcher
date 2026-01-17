import streamlit as st
from fer import FER
from PIL import Image
import numpy as np
from transformers import pipeline

# --- PAGE SETUP ---
st.set_page_config(page_title="Vibe-Catcher", page_icon="🎯")
st.title("🎯 Multimodal Vibe-Catcher")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Use 'distilbert' because it's the gold standard for 'lightweight' NLP
    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # mtcnn=False uses the faster Haar Cascade method for face detection
    vision = FER(mtcnn=False)
    return nlp, vision

sentiment_ai, face_ai = load_models()

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Camera Input")
    camera_img = st.camera_input("Smile or look locked in!")

with col2:
    st.subheader("✍️ Text Input")
    user_text = st.text_input("Tell the AI your current vibe:")

# --- EXECUTION ---
if st.button("Analyze My Vibe"):
    if camera_img and user_text:
        with st.spinner("Fusing Data Streams..."):
            # 1. Vision Processing
            img = Image.open(camera_img)
            img_array = np.array(img)
            face_results = face_ai.detect_emotions(img_array)
            
            # 2. NLP Processing
            text_results = sentiment_ai(user_text)[0]
            
            # 3. Presentation
            st.divider()
            
            # Display Face Result
            if face_results:
                emotions = face_results[0]["emotions"]
                top_face_vibe = max(emotions, key=emotions.get)
                st.write(f"**Face Signal:** {top_face_vibe.capitalize()}")
            else:
                st.write("**Face Signal:** No face detected.")

            # Display Text Result
            st.write(f"**Text Signal:** {text_results['label']} ({round(text_results['score']*100)}% confidence)")
            
            # THE DOCTOR'S FAVORITE: CROSS-MODAL FUSION
            st.info(f"**AI Synthesis:** The system detected a {text_results['label'].lower()} semantic tone combined with a {top_face_vibe if face_results else 'neutral'} visual state.")
    else:
        st.warning("I need both a photo and some text to do the fusion!")
