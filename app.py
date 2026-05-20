import os
import pickle
import streamlit as st
import numpy as np

from src.preprocessing import clean_text

st.set_page_config(
    page_title="GuardianAI - Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_ml_assets():
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_ml_assets()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply clean font to main elements */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title style with elegant gradient */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .header-subtitle {
        color: #718096;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Sleek card styling */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Glowing visual feedback containers for classification results */
    .result-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .result-box-ham {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.2) 100%);
        border: 2px solid #38A169;
        color: #2F855A;
    }
    
    .result-box-spam {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(229, 62, 62, 0.2) 100%);
        border: 2px solid #E53E3E;
        color: #9B2C2C;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Animation helper */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sample Button styling overrides */
    div.stButton > button:first-child {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #5a67d8;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11);
    }
    
    /* Secondary/Sample Button stylings */
    .sample-btn-label {
        font-size: 0.85rem;
        color: #A0AEC0;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-title'>🛡️ GuardianAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Machine Learning Spam Email & SMS Classifier</p>", unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error("🚨 Trained Model components were not found! Please run the training pipeline first: `python train_model.py` in your terminal.")
    st.stop()

if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

def apply_sample(sample_text):
    st.session_state.current_input = sample_text

SPAM_SAMPLE = "URGENT! Your mobile number has been selected for a free £2000 prize draw! Call 09061701461 to claim your reward. T&Cs apply."
HAM_SAMPLE = "Hey, are we still meeting for lunch at 12:30? Let me know if you need a ride, I can pick you up."

col1, col2 = st.columns([6, 5], gap="large")

with col1:
    st.subheader("📝 Analyze a Message")
    st.write("Type or paste the contents of an email or SMS below to analyze it for malicious spam content.")
    
    input_text = st.text_area(
        label="Message Contents",
        value=st.session_state.current_input,
        placeholder="Paste your email or SMS text here...",
        height=220,
        label_visibility="collapsed",
        key="text_area_input"
    )
    
    char_count = len(input_text)
    word_count = len(input_text.split()) if char_count > 0 else 0
    
    stat_cols = st.columns(3)
    stat_cols[0].metric("Characters", char_count)
    stat_cols[1].metric("Words", word_count)
    
    predict_clicked = st.button("🔍 Analyze Message", use_container_width=True)

    st.write("")
    st.markdown("<p class='sample-btn-label'>💡 QUICK TESTING PRESETS:</p>", unsafe_allow_html=True)
    preset_cols = st.columns(2)
    if preset_cols[0].button("🟢 Safe Message Example", use_container_width=True):
        apply_sample(HAM_SAMPLE)
        st.rerun()
    if preset_cols[1].button("🔴 Spam Message Example", use_container_width=True):
        apply_sample(SPAM_SAMPLE)
        st.rerun()

with col2:
    st.subheader("📊 Classifier Report")
    
    current_text = st.session_state.text_area_input.strip()
    
    if current_text:
    
        cleaned = clean_text(current_text)
        
        vectorized = vectorizer.transform([cleaned])
        
        prediction = model.predict(vectorized)[0]
        
        probabilities = model.predict_proba(vectorized)[0]
        class_labels = model.classes_
        
        ham_conf = probabilities[np.where(class_labels == 'ham')[0][0]] * 100
        spam_conf = probabilities[np.where(class_labels == 'spam')[0][0]] * 100
        
        prediction_confidence = spam_conf if prediction == 'spam' else ham_conf
        
        if prediction == "spam":
            st.markdown(f"""
                <div class='result-box result-box-spam'>
                    <div class='result-title'>⚠️ HIGH RISK SPAM</div>
                    <div class='result-text'>This message shares high statistical similarities with known fraudulent, scam, or marketing campaigns. Do not click any links!</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.metric(label="🎯 Prediction Confidence", value=f"{prediction_confidence:.2f}%")
            st.progress(prediction_confidence / 100.0)
            
        else:
            st.markdown(f"""
                <div class='result-box result-box-ham'>
                    <div class='result-title'>🟢 SAFE MESSAGE (HAM)</div>
                    <div class='result-text'>This message has been identified as safe, legitimate conversation. No suspicious marketing or phishing indicators were found.</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.metric(label="🎯 Prediction Confidence", value=f"{prediction_confidence:.2f}%")
            st.progress(prediction_confidence / 100.0)
            
        with st.expander("🛠️ Advanced NLP Feature Inspection"):
            st.markdown("##### Preprocessing Trace")
            st.write("Below is what the model actually **sees** after stripping punctuation, lowercasing, and removing stopwords:")
            
            if cleaned:
                st.info(f"**Cleaned Tokens:** `{cleaned}`")
            else:
                st.warning("⚠️ The input text contained only stopwords/punctuation! The cleaning pipeline returned an empty list.")
                
            st.markdown("##### Confidence Probability Distribution")
            p_cols = st.columns(2)
            p_cols[0].metric(label="Safe Probability", value=f"{ham_conf:.2f}%")
            p_cols[1].metric(label="Spam Probability", value=f"{spam_conf:.2f}%")
            
    else:
        st.info("👈 Enter message text or click one of the quick-presets on the left to trigger the machine learning analysis.")
        
        st.markdown("""
        ### How does GuardianAI work?
        1. **NLP Preprocessing**: Your raw text is cleaned to extract only root keywords. Emojis, exclamation marks, and grammar filler words are removed.
        2. **TF-IDF Vectorization**: Words are converted into numerical scores representing how important each word is in the dataset.
        3. **Naive Bayes Classifier**: A mathematical model calculates the probability of the message being **Spam** vs. **Safe** based on the frequency patterns learned during training.
        """)

st.markdown("---")
st.markdown("### ⚙️ Model Information & Pipeline Diagnostics")

metric_cols = st.columns(4)
metric_cols[0].metric("Model Type", "Multinomial Naive Bayes")
metric_cols[1].metric("Accuracy Score", "96.68%")
metric_cols[2].metric("Dataset Rows", "5,567 Items")
metric_cols[3].metric("Feature Set Size", "5,000 Vectors")

with st.expander("🔍 Detailed Model Performance Metrics (Confusion Matrix & Classification Report)"):
    st.write("These metrics were generated on unseen test data during model training:")
    
    col_rep, col_conf = st.columns(2)
    
    with col_rep:
        st.markdown("**Classification Report**")
        st.code("""
              precision    recall  f1-score   support

         ham       0.96      1.00      0.98       965
        spam       1.00      0.75      0.86       149

    accuracy                           0.97      1114
   macro avg       0.98      0.88      0.92      1114
weighted avg       0.97      0.97      0.96      1114
        """)
        
    with col_conf:
        st.markdown("**Confusion Matrix (Test Split)**")
        st.code("""
Actual / Predicted    |   Predicted Ham   |   Predicted Spam
---------------------------------------------------------
Actual Ham (965)      |       965         |         0
Actual Spam (149)     |        37         |       112
        """)
        
        st.write("""
        * **965** True Negatives: Legitimate messages correctly classified as safe.
        * **112** True Positives: Real spam messages correctly identified.
        * **37** False Negatives: Spam messages that slipped through the filter.
        * **0** False Positives: **Zero** legitimate emails were blocked as spam! (Highly desirable!)
        """)
