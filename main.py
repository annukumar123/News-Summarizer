import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article
from textblob import TextBlob
import torch
import nltk
import re

# --- 0. NLTK PRE-REQUISITES ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- 1. CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="AI News Insights", layout="wide", page_icon="📰")

# --- 2. CUSTOM BEAUTIFUL CSS ---
st.markdown("""
    <style>
    /* Main Title Styling */
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Style the 'Analyze Article' button */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        height: 3.5em;
        width: 100%;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Style the 'Download' button */
    .stDownloadButton > button {
        background-color: #008CBA !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 100% !important;
        height: 3em !important;
    }
    
    /* Custom info box styling */
    .stInfo {
        border-left: 5px solid #1E3A8A !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">📰 AI News Summarizer & Sentiment Analyzer</p>', unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_ai_model()

# --- 4. INPUT SECTION ---
col1, col2 = st.columns([1, 1])

# Initialize session state for metadata
if 'metadata' not in st.session_state:
    st.session_state.metadata = {"author": "Unknown", "date": "Unknown", "title": "Manual Entry"}

with col1:
    option = st.radio("Choose Input Method:", ("URL", "Manual Text"))
    input_text = ""
    
    if option == "URL":
        url = st.text_input("Paste Link:", placeholder="https://www.bbc.com/news/...")
        if url:
            try:
                article = Article(url)
                article.download()
                article.parse()
                input_text = article.text
                
                # Update Metadata
                st.session_state.metadata = {
                    "author": ", ".join(article.authors) if article.authors else "Not Found",
                    "date": str(article.publish_date.date()) if article.publish_date else "Not Found",
                    "title": article.title
                }
                st.success(f"✅ Article Found: {article.title}")
            except Exception as e:
                st.error(f"❌ Error fetching article: {e}")
    else:
        input_text = st.text_area("Paste Text:", height=200, placeholder="Paste your article content here...")
        st.session_state.metadata = {"author": "User Provided", "date": "N/A", "title": "Manual Entry"}

# --- 5. PROCESSING & DISPLAY ---
if st.button("Analyze Article ✨"):
    if input_text.strip():
        with st.spinner("🧠 AI is analyzing and summarizing..."):
            
            # Clean "Read More" junk (common in Indian news sites like TOI/NDTV)
            clean_text = re.sub(r'Read more at:.*', '', input_text, flags=re.DOTALL)
            
            # A. Summarization (Direct Model Generation)
            inputs = tokenizer(clean_text, max_length=1024, truncation=True, return_tensors="pt").to(device)
            summary_ids = model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=150, 
                min_length=40, 
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # B. Sentiment Analysis
            analysis = TextBlob(clean_text)
            polarity = analysis.sentiment.polarity
            sentiment = "Positive 😊" if polarity > 0 else "Negative 😟" if polarity < 0 else "Neutral 😐"

            # --- C. DISPLAY RESULTS ---
            st.markdown("---")
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.subheader("📝 AI Summary")
                st.info(summary)
                
                # Meta info box
                st.markdown(f"""
                **Article Details:**
                * 👤 **Author:** {st.session_state.metadata['author']}
                * 📅 **Published Date:** {st.session_state.metadata['date']}
                """)
            
            with res_col2:
                st.subheader("📊 Sentiment Insights")
                st.metric(label="Polarity Score", value=round(polarity, 2), delta=sentiment.split()[0])
                st.write(f"Tone: **{sentiment}**")
                st.progress((polarity + 1) / 2) 

                # --- D. DOWNLOAD REPORT ---
                report_text = f"""
NEWS SUMMARY REPORT
-------------------
Title: {st.session_state.metadata['title']}
Author: {st.session_state.metadata['author']}
Date: {st.session_state.metadata['date']}

SUMMARY:
{summary}

SENTIMENT:
{sentiment} (Score: {round(polarity, 2)})
-------------------
Generated by AI News Insights
                """
                
                st.download_button(
                    label="📥 Download Summary Report",
                    data=report_text,
                    file_name="news_summary_report.txt",
                    mime="text/plain"
                )
    else:
        st.warning("⚠️ Please provide some input first!")