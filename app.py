import streamlit as st
from transformers import pipeline

# 1. Setup
st.set_page_config(page_title="AI Vibe Engine", page_icon="📑")
st.title("📑 AI Vibe & Summary Engine")
st.markdown("*Stable Multi-Task NLP*")

# 2. Load Models Safely
@st.cache_resource
def load_nlp():
    # 'framework="pt"' is the secret to fixing your ValueError!
    # It forces the use of PyTorch and avoids TensorFlow crashes.
    summary_pipe = pipeline(
        "summarization", 
        model="sshleifer/distilbart-cnn-6-6", 
        framework="pt"
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        framework="pt"
    )
    return summary_pipe, sentiment_pipe

try:
    summarizer, analyzer = load_nlp()

    # 3. User Input
    user_input = st.text_area("Paste text here for analysis:", height=200)

    if st.button("✨ Analyze Vibes"):
        if len(user_input) > 50:
            with st.spinner("AI is thinking..."):
                # A. Analysis
                sentiment = analyzer(user_input[:512])[0]
                summary = summarizer(user_input, max_length=60, min_length=20)[0]['summary_text']
                
                # B. Results Display
                st.divider()
                st.subheader("Summary")
                st.info(summary)
                
                st.subheader("Sentiment Analysis")
                if sentiment['label'] == "POSITIVE":
                    st.success(f"Vibe: {sentiment['label']} ({round(sentiment['score']*100)}%)")
                else:
                    st.error(f"Vibe: {sentiment['label']} ({round(sentiment['score']*100)}%)")
        else:
            st.warning("Please enter at least 50 characters so the AI has enough context.")

except Exception as e:
    st.error("Model loading issue. Please try 'Reboot App' in Streamlit settings.")
    st.write(f"Error Details: {e}")
