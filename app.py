import streamlit as st
from transformers import pipeline

# 1. Setup
st.set_page_config(page_title="AI Vibe Summarizer", page_icon="📑")
st.title("📑 AI Vibe & Summary Engine")
st.markdown("---")

# 2. Load Models (Stable Local Versions)
@st.cache_resource
def load_nlp():
    # Summarization model (Very stable)
    summary_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Sentiment model
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summary_pipe, sentiment_pipe

summarizer, analyzer = load_nlp()

# 3. Input
user_input = st.text_area("Paste a long article or 'yapping' here:", height=200)

if st.button("✨ Analyze & Summarize"):
    if len(user_input) > 20:
        with st.spinner("AI is reading the vibes..."):
            # A. Get Sentiment
            sentiment = analyzer(user_input[:512])[0] # Analyze first 512 tokens
            
            # B. Get Summary
            summary = summarizer(user_input, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
            
            # 4. Display Results with "Dr. Level" Analysis
            st.subheader("Results")
            
            # Color coding based on vibe
            if sentiment['label'] == "POSITIVE":
                st.success(f"**Vibe Check:** Positive Mood ({round(sentiment['score']*100)}%)")
            else:
                st.error(f"**Vibe Check:** Negative/Serious Mood ({round(sentiment['score']*100)}%)")
            
            st.markdown(f"**AI Summary:** {summary}")
            
            # Technical Explanation for the Dr.
            with st.expander("See Technical Metadata"):
                st.write("Model 1: DistilBART (Abstractive Summarization)")
                st.write("Model 2: DistilBERT (Sequence Classification)")
                st.write("Technique: Pipeline Parallelism")
    else:
        st.warning("Please enter more text for a valid analysis.")
