import os
import sys
import email
import streamlit as st
import torch
import warnings
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Add project root to sys path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.features.extract_meta import get_url_count, get_keyword_flags

# Suppress HuggingFace/PyTorch warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #007bff;
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .title-text {
        text-align: center;
        color: #343a40;
        font-family: 'Inter', sans-serif;
    }
    .subtitle-text {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading BERT Model...")
def load_model():
    """Loads the fine-tuned BERT model and tokenizer."""
    model_path = os.path.join(project_root, 'models', 'bert')
    
    if not os.path.exists(model_path) or not os.listdir(model_path):
        st.error(f"Model not found at {model_path}. Please ensure you have downloaded and extracted the BERT model files.")
        st.stop()
        
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_resource(show_spinner="Loading ML Model...")
def load_sklearn_model():
    """Loads the traditional ML (regression) model."""
    model_path = os.path.join(project_root, 'models', 'phishing_model.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"ML Model not found at {model_path}.")
        st.stop()
        
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load ML model: {e}")
        st.stop()

def analyze_email(text, tokenizer, model):
    """Tokenizes input and runs inference."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    prediction_idx = torch.argmax(probabilities).item()
    confidence = probabilities[prediction_idx].item() * 100
    
    return prediction_idx, confidence

def get_top_phishing_words(text, pipeline):
    """Extracts top words contributing to a phishing prediction."""
    try:
        tfidf_step = pipeline.named_steps['tfidf']
        model_step = pipeline.named_steps['model']
        
        tfidf_vec = tfidf_step.transform([text])
        feature_indices = tfidf_vec.nonzero()[1]
        
        if len(feature_indices) == 0:
            return []
            
        feature_names = tfidf_step.get_feature_names_out()
        
        if model_step.coef_.shape[0] == 1:
            coefs = model_step.coef_[0]
        else:
            idx_1 = list(model_step.classes_).index(1)
            coefs = model_step.coef_[idx_1]
            
        impacts = []
        for idx in feature_indices:
            word = feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = coefs[idx]
            impact = tfidf_val * weight
            if impact > 0:
                impacts.append((word, impact))
                
        impacts.sort(key=lambda x: x[1], reverse=True)
        return [w[0] for w in impacts[:5]]
    except Exception as e:
        return []

# Main App Interface
st.markdown("<h1 class='title-text'>🛡️ Phishing Email Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Analyze an email to determine if it's safe or a phishing attempt.</p>", unsafe_allow_html=True)

tokenizer, model = load_model()
sklearn_model = load_sklearn_model()

# Input Methods
input_type = st.radio("Choose Input Method:", ["Paste Text", "Upload .eml File"], horizontal=True)

email_text = ""
attachment_count = 0

if input_type == "Paste Text":
    email_text = st.text_area("Email Content", height=200, placeholder="Paste the email text here...", label_visibility="collapsed")
else:
    uploaded_file = st.file_uploader("Upload an .eml file", type=["eml", "txt"])
    if uploaded_file is not None:
        try:
            raw_data = uploaded_file.read().decode('utf-8', errors='ignore')
            msg = email.message_from_string(raw_data)
            
            # Extract body and count attachments
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body_parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                
                if part.get_content_maintype() != 'multipart' and part.get('Content-Disposition') is not None:
                    attachment_count += 1
                    
            email_text = "\n".join(body_parts) if body_parts else raw_data
            st.text_area("Extracted Email Content", value=email_text, height=200, disabled=True)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please provide some text to analyze.")
    else:
        with st.spinner("Analyzing with multiple models..."):
            prediction_idx, confidence = analyze_email(email_text, tokenizer, model)
            
            # Sklearn Model Prediction
            sklearn_pred = sklearn_model.predict([email_text])[0]
            sklearn_prob = None
            if hasattr(sklearn_model, "predict_proba"):
                try:
                    prob_idx = list(sklearn_model.classes_).index(sklearn_pred)
                    sklearn_prob = sklearn_model.predict_proba([email_text])[0][prob_idx] * 100
                except (ValueError, IndexError):
                    sklearn_prob = None
            
            st.markdown("---")
            st.subheader("Analysis Results")
            
            col_bert, col_ml = st.columns(2)
            
            with col_bert:
                st.markdown("#### Deep Learning (BERT)")
                if prediction_idx == 1:
                    st.error("🚨 **PHISHING EMAIL**")
                else:
                    st.success("✅ **SAFE EMAIL**")
                    
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                if prediction_idx == 1:
                    st.progress(int(confidence), text="Phishing Probability")
                else:
                    st.progress(int(confidence), text="Safe Probability")

            with col_ml:
                st.markdown("#### Machine Learning (Regression)")
                if sklearn_pred == 1:
                    st.error("🚨 **PHISHING EMAIL**")
                else:
                    st.success("✅ **SAFE EMAIL**")
                
                if sklearn_prob is not None:
                    st.metric(label="Confidence Score", value=f"{sklearn_prob:.2f}%")
                    if sklearn_pred == 1:
                        st.progress(int(sklearn_prob), text="Phishing Probability")
                    else:
                        st.progress(int(sklearn_prob), text="Safe Probability")
                else:
                    st.info("Confidence score unavailable for this model.")
                    
            # Threat Indicators Section
            st.markdown("---")
            with st.expander("🔍 View Threat Indicators", expanded=True):
                st.markdown("These indicators highlight why the models might classify this email as suspicious.")
                
                ind_col1, ind_col2, ind_col3 = st.columns(3)
                
                with ind_col1:
                    st.markdown("**🔗 Links & Attachments**")
                    url_count = get_url_count(email_text)
                    st.write(f"- URLs Found: `{url_count}`")
                    st.write(f"- Attachments: `{attachment_count}`")
                    
                with ind_col2:
                    st.markdown("**🚩 Suspicious Keywords**")
                    flags = get_keyword_flags(email_text)
                    active_flags = [k.replace('has_', '').capitalize() for k, v in flags.items() if v == 1]
                    if active_flags:
                        for flag in active_flags:
                            st.write(f"- 🔴 {flag}")
                    else:
                        st.write("- ✅ None detected")
                        
                with ind_col3:
                    st.markdown("**🧠 ML Top Contributors**")
                    top_words = get_top_phishing_words(email_text, sklearn_model)
                    if top_words:
                        for word in top_words:
                            st.write(f"- 📈 `{word}`")
                    else:
                        st.write("- ✅ No strong indicators")
