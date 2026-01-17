import streamlit as st
from fer import FER
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multimodal Vibe Catcher", page_icon="🧘")
st.title("🧘 Multimodal Vibe Catcher")
st.markdown("Analyzing your **Facial Expression** and **Written Vibe** together.")

# --- LOAD MODELS ---
@st.cache_resource
def load_nlp():
    # Standard Sentiment Analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_ai = load_nlp()
face_detector = FER(mtcnn=True) # Facial Emotion Recognition

# --- CAMERA INPUT ---
st.header("1. Visual Vibe (Camera)")
img_file = st.camera_input("How are you looking today?")

face_vibe = "Unknown"
if img_file:
    # Convert image for the AI
    img = Image.open(img_file)
    frame = np.array(img)
    
    # Detect Emotion
    with st.spinner("Analyzing face..."):
        result = face_detector.detect_emotions(frame)
        if result:
            # Get the top emotion
            emotions = result[0]["emotions"]
            face_vibe = max(emotions, key=emotions.get)
            st.write(f"**Face Detection:** You look **{face_vibe}**")
        else:
            st.warning("No face detected. Try better lighting!")

# --- TEXT INPUT ---
st.header("2. Thought Vibe (Text)")
user_text = st.text_input("How are you actually feeling? (Be honest)")

if st.button("Generate Total Vibe Report"):
    if user_text:
        # NLP Task: Sentiment
        text_analysis = sentiment_ai(user_input=user_text)[0]
        text_vibe = text_analysis['label']
        
        # FINAL MULTIMODAL LOGIC
        st.divider()
        st.subheader("Final AI Diagnosis")
        
        if face_vibe.lower() == "happy" and text_vibe == "POSITIVE":
            st.balloons()
            st.success("Absolute Legend: Your face and thoughts are both in a great place. Keep cooking!")
        elif face_vibe.lower() == "sad" or text_vibe == "NEGATIVE":
            st.info("Vibe Check: You might be feeling a bit 'mid' or cooked. Take a break, fr.")
        else:
            st.warning("Mixed Signals: Your face says one thing, but your words say another. Intriguing...")
            
        st.write(f"**Face:** {face_vibe} | **Words:** {text_vibe}")
    else:
        st.error("Write something first!")
