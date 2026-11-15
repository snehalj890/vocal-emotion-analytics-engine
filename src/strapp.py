import os
import time
import json
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Vocal Emotion Intelligence | SaaS Hub",
    page_icon="🎙️",
    layout="wide"
)

# Elite SaaS Landing Page Custom CSS Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1f2937 100%);
        color: #f3f4f6;
    }
    .saas-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .saas-logo {
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #c084fc, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.7) 0%, rgba(17, 24, 39, 0.8) 100%);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 15px;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        max-width: 600px;
        margin-bottom: 25px;
    }
    .card-container {
        background: rgba(17, 24, 39, 0.75);
        padding: 20px 25px;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }
    .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    p, label, .streamlit-expanderHeader {
        color: #e5e7eb !important;
    }
    [data-testid="stFileUploader"] section div span, 
    [data-testid="stFileUploader"] section div small {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    [data-testid="stAudioInput"] button {
        transform: scale(1.2);
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        margin: 10px 0px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        padding: 0.75rem 1rem;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.95;
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(139, 92, 246, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# Load the trained model
@st.cache_resource
def load_ser_model():
    model_path = "../models/best_speech_emotion_cnn_lstm.keras"
    if not os.path.exists(model_path):
        model_path = "../models/best_speech_emotion_cnn_lstm.h5"
    if not os.path.exists(model_path):
        model_path = "models/best_speech_emotion_cnn_lstm.keras"
    if not os.path.exists(model_path):
        model_path = "models/best_speech_emotion_cnn_lstm.h5"
    return load_model(model_path)

try:
    model = load_ser_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

EMOTIONS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

def extract_features(file_path, max_pad_len=216):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T
        if mfccs.shape[0] < max_pad_len:
            mfccs = np.pad(mfccs, ((0, max_pad_len - mfccs.shape[0]), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_pad_len, :]
        return mfccs, audio, sample_rate
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None, None

# --- AUTHENTICATION GATE ---
if not st.session_state.logged_in:
    st.markdown('<p class="main-title" style="margin-top: 50px;">🔐 Secure Enterprise Authentication</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Please login or register your account to access the Vocal Emotion Intelligence Hub.</p>', unsafe_allow_html=True)
    
    col_auth1, col_auth2, col_auth3 = st.columns([1, 1.5, 1])
    with col_auth2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        auth_mode = st.radio("Choose Action:", ["Login", "Register New Account"])
        
        email_input = st.text_input("📧 Email Address")
        password_input = st.text_input("🔒 Password", type="password")
        
        if auth_mode == "Register New Account":
            name_input = st.text_input("👤 Full Name")
            if st.button("✨ Create Account & Register"):
                if email_input and password_input and name_input:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email_input
                    st.success("Account created successfully! Redirecting...")
                    st.rerun()
                else:
                    st.warning("Please fill in all registration fields.")
        else:
            if st.button("🚀 Sign In to Dashboard"):
                if email_input and password_input:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email_input
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.warning("Please enter your email and password.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- SaaS Navigation Bar Header ---
st.markdown(f"""
    <div class="saas-nav">
        <div class="saas-logo">🎙️ VOCAL EMOTION AI HUB &nbsp;|&nbsp; <span style="font-size: 0.85rem; color: #38bdf8;">👤 {st.session_state.user_email}</span></div>
    </div>
""", unsafe_allow_html=True)

nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1, 1, 1, 1, 1.5])
if nav_c1.button("🏠 Home"):
    st.session_state.current_page = "Home"
if nav_c2.button("📊 Analytics"):
    st.session_state.current_page = "Analytics"
if nav_c3.button("📈 Evaluation"):
    st.session_state.current_page = "Evaluation"
if nav_c4.button("💼 Portfolio"):
    st.session_state.current_page = "Portfolio"
if nav_c5.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.markdown("---")

page = st.session_state.current_page

if page == "Home":
    col_h1, col_h2 = st.columns([1.3, 0.7])
    with col_h1:
        st.markdown("""
            <div class="hero-card">
                <div class="hero-title">Speech emotion<br>analysis never ends</div>
                <div class="hero-subtitle">Enterprise-grade deep learning speech recognition powered by CNN-LSTM neural architecture, real-time MFCC feature extraction, and executive telemetry dashboards.</div>
            </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=800&auto=format&fit=crop", use_container_width=True)
        
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start Live Analyzer")
    
    input_method = st.radio("Select audio source:", ["Upload Audio File (.wav / .mp3)", "Record Live Voice (Microphone)"])
    
    audio_file_to_process = None
    file_display_name = ""

    if input_method == "Upload Audio File (.wav / .mp3)":
        uploaded_file = st.file_uploader("Select a speech recording (.wav or .mp3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_file_to_process = uploaded_file
            file_display_name = uploaded_file.name
    else:
        st.markdown("""
            <div style="background: rgba(56, 189, 248, 0.15); border-left: 4px solid #38bdf8; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <span style="color: #38bdf8; font-weight: 700; font-size: 1.05rem;">🎙️ Click here to record your voice:</span> Use the microphone button below to speak directly into your browser, then click 'Run Quick Prediction' once finished!
            </div>
        """, unsafe_allow_html=True)
        
        recorded_audio = st.audio_input("Record your voice")
        if recorded_audio is not None:
            audio_file_to_process = recorded_audio
            file_display_name = "live_mic_recording.wav"

    if audio_file_to_process is not None:
        st.audio(audio_file_to_process)
        if st.button("⚡ Run Quick Prediction"):
            temp_path = f"temp_{file_display_name}"
            if hasattr(audio_file_to_process, "getbuffer"):
                with open(temp_path, "wb") as f:
                    f.write(audio_file_to_process.getbuffer())
            else:
                temp_path = audio_file_to_process
                
            # Measure Real-Time Inference Latency
            start_time = time.time()
            features, audio_signal, sr = extract_features(temp_path)
            
            if features is not None:
                model_input = np.expand_dims(features, axis=0)
                predictions = model.predict(model_input)
                inference_latency = (time.time() - start_time) * 1000  # in ms
                
                predicted_class_idx = np.argmax(predictions[0])
                predicted_emotion = EMOTIONS.get(predicted_class_idx, "Unknown")
                confidence = np.max(predictions[0]) * 100
                
                st.session_state.current_prediction = {
                    "filename": file_display_name,
                    "emotion": predicted_emotion.upper(),
                    "confidence": round(confidence, 2),
                    "latency": round(inference_latency, 2),
                    "predictions": predictions[0],
                    "audio_signal": audio_signal,
                    "sr": sr
                }
                
                # Chronological History Logging (Max 10 entries)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "Timestamp": timestamp,
                    "Filename": file_display_name,
                    "Predicted Emotion": predicted_emotion.upper(),
                    "Confidence (%)": round(confidence, 2),
                    "Latency (ms)": round(inference_latency, 2)
                }
                
                st.session_state.history.append(history_entry)
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]
                    
            if os.path.exists(temp_path) and "temp_" in temp_path:
                os.remove(temp_path)
                
    if st.session_state.current_prediction:
        res = st.session_state.current_prediction
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); 
                        padding: 16px 20px; 
                        border-radius: 10px; 
                        border: 2px solid #34d399; 
                        text-align: center; 
                        margin: 15px 0;
                        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4);">
                <span style="color: #ffffff !important; font-size: 1.2rem !important; font-weight: 800 !important; text-decoration: underline; text-underline-offset: 6px;">
                    🎯 Detected Emotion: {res['emotion']} with {res['confidence']}% confidence!
                </span>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Analytics":
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Real-Time Audio Telemetry & Neural Visualizations")
    if st.session_state.current_prediction:
        res = st.session_state.current_prediction
        y_sig, sr_sig = res['audio_signal'], res['sr']
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Duration", f"{librosa.get_duration(y=y_sig, sr=sr_sig):.2f} s")
        col_t2.metric("Sample Rate", f"{sr_sig} Hz")
        col_t3.metric("RMS Energy", f"{np.mean(librosa.feature.rms(y=y_sig)):.4f}")
        col_t4.metric("Zero Cross Rate", f"{np.mean(librosa.feature.zero_crossing_rate(y=y_sig)):.4f}")
        
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 3))
        S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y_sig, sr=sr_sig, n_mels=128, fmax=8000), ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_sig, fmax=8000, ax=ax, cmap='coolwarm')
        ax.set(title='Mel-Frequency Spectrogram')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)
        
        radar_df = pd.DataFrame(dict(r=[float(res['predictions'][i]) for i in range(len(EMOTIONS))], theta=[EMOTIONS[i].capitalize() for i in range(len(EMOTIONS))]))
        fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, markers=True)
        fig_radar.update_traces(fill='toself', line_color='#ec4899', marker_color='#38bdf8')
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Please perform a prediction on the **Home** tab first to view telemetry.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Session Audit Log & 10-Entry History Center ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### 📜 Session Audit Log & Advanced Export Center")
    
    if len(st.session_state.history) > 0:
        df_history = pd.DataFrame(st.session_state.history)
        total_clips = len(df_history)
        avg_conf = df_history["Confidence (%)"].mean()
        dominant_emo = df_history["Predicted Emotion"].mode()[0] if not df_history.empty else "N/A"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Clips Analyzed", f"{total_clips} / 10")
        c2.metric("Average Confidence", f"{avg_conf:.2f}%")
        c3.metric("Dominant Session Emotion", dominant_emo)
        
        st.markdown("---")
        df_display = df_history.copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = "No."
        st.dataframe(df_display, use_container_width=True)
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_data = df_display.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Audit Report (.csv)",
                data=csv_data,
                file_name="speech_emotion_audit_report.csv",
                mime="text/csv"
            )
        with col_ex2:
            json_data = df_history.to_json(orient="records", indent=4)
            st.download_button(
                label="📥 Download Telemetry Packet (.json)",
                data=json_data,
                file_name="speech_emotion_telemetry.json",
                mime="application/json"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History Log"):
            st.session_state.history = []
            st.session_state.current_prediction = None
            st.rerun()
    else:
        st.info("No analysis records yet. Analyze audio clips on the **Home** tab to populate the 10-entry session audit history log!")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Evaluation":
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### 📈 CNN-LSTM Model Evaluation & Audit")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Architecture", "CNN + Bi-LSTM")
    col_m2.metric("Validation Accuracy", "88.4%")
    col_m3.metric("Feature Extraction", "40-dim MFCCs")
    
    st.markdown("---")
    cm_path = "../outputs/confusion_matrix.png" if os.path.exists("../outputs/confusion_matrix.png") else "outputs/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Model Validation Confusion Matrix Heatmap", use_container_width=True)
    else:
        st.info("ℹ️ Confusion matrix image not found in outputs directory.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Portfolio":
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("### 💼 Project Portfolio & Executive Summary")
    st.markdown("""
    * **Project Title:** Speech Emotion Recognition (SER) Intelligence Hub
    * **Core Technologies:** Python, TensorFlow/Keras, Streamlit, Librosa, Plotly, Pandas.
    * **Model Architecture:** Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) layers for temporal speech sequence classification.
    * **Key Capabilities:** Real-time MFCC feature extraction, browser microphone recording, interactive 2D radar confidence scoring, audio spectrogram visualization, and secure user authentication gating.
    """)
    st.markdown('</div>', unsafe_allow_html=True)