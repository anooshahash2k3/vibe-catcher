import streamlit as st
from fer import FER
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline

# 1. Setup
st.set_page_config(page_title="Vibe-Catcher AI", page_icon="🎯")
st.title("🎯 Multimodal Vibe-Catcher")
st.markdown("### NLP + Computer Vision Fusion")

# 2. Load Models Safely
@st.cache_resource
def load_nlp_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_vision_model():
    # mtcnn=False makes it run faster on basic servers
    return FER(mtcnn=False)

sentiment_pipe = load_nlp_model()
detector = load_vision_model()

# 3. Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Visual Mode")
    img_file = st.camera_input("Take a photo")

with col2:
    st.subheader("✍️ Text Mode")
    user_text = st.text_input("How's your day going?", placeholder="e.g. It's been a total grind but I'm locked in.")

# 4. Logic & Fusion
if st.button("Run Multimodal Analysis"):
    if img_file and user_text:
        with st.spinner("Calculating Vibe..."):
            # A. Process Vision
            img = Image.open(img_file)
            frame = np.array(img)
            face_data = detector.detect_emotions(frame)
            
            # B. Process NLP
            text_data = sentiment_pipe(user_text)[0]
            
            # C. Display Results
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if face_data:
                    emotions = face_data[0]["emotions"]
                    top_emotion = max(emotions, key=emotions.get)
                    st.metric("Face Emotion", top_emotion.capitalize())
                else:
                    st.warning("No face found in photo.")

            with res_col2:
                st.metric("Text Sentiment", text_data['label'], f"{round(text_data['score']*100)}%")

            # D. The "Dr. of AI" Fusion Logic
            st.subheader("Final System Summary")
            if face_data:
                # This is a heuristic 'Vibe' fusion
                if top_emotion == "happy" and text_data['label'] == "POSITIVE":
                    st.success("Analysis: User is in a 'High-Vibe' state. Neural alignment detected. 🚀")
                elif text_data['label'] == "NEGATIVE":
                    st.error("Analysis: Potential 'Brain Rot' or stress detected. Take a break. 🛑")
                else:
                    st.info("Analysis: Neutral or Mixed signals detected. System requires more data.")
    else:
        st.warning("Please provide BOTH a photo and text for multimodal fusion.")
