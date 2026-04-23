import streamlit as st
import os
import subprocess
from PIL import Image

# ===================================================================
# APP CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="NeuroDecode — Brain-Computer Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background-color: #080812;
        color: #c8c8d8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #0e0e1c;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        color: #808098;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom-color: #ff66aa !important;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .metric-card {
        background-color: #0e0e1c;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333355;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
OUT_DIR = "output"

def load_image(filename):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

def display_figure(title, filename, description=""):
    img = load_image(filename)
    if img:
        st.subheader(title)
        if description:
            st.markdown(f"*{description}*")
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Figure `{filename}` not found. Please run the backend pipeline first.")

# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.title("🧠 NeuroDecode")
    st.markdown("**Brain-Computer Interface System**")
    st.markdown("---")
    
    st.markdown("""
    This dashboard visualizes an advanced machine learning pipeline that decodes human intent (imagining left vs right hand movement) directly from 64-channel EEG brain signals.
    
    **Core Technology Stack:**
    - Common Spatial Patterns (CSP)
    - Support Vector Machines (SVM)
    - Continuous Wavelet Transforms (CWT)
    - Butterworth Digital Filters
    """)
    st.markdown("---")

    st.subheader("Data Status")
    st.info("The dashboard is serving validated results from real human neural data.")
    
    st.markdown("---")
    st.caption("Powered by Neural Intelligence")


# ===================================================================
# MAIN CONTENT
# ===================================================================
st.title("Neural Decoding: Motor Imagery")
st.markdown("Predicting left vs. right hand movement directly from brain oscillations.")

if not os.path.exists(OUT_DIR) or len(os.listdir(OUT_DIR)) == 0:
    st.error("No analysis results found! Please ensure pipeline output exists.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Brainwave Feed", 
    "Spectral Activity", 
    "Neural Oscillations", 
    "Intent Recognition", 
    "System Accuracy"
])

# -------------------------------------------------------------------
# TAB 1: Raw Signals & Filtering
# -------------------------------------------------------------------
with tab1:
    st.header("Live Feed & Filtering")
    st.markdown("""
    Raw EEG data contains millions of overlapping brain oscillations. We use **zero-phase digital bandpass filters** to safely decompose the signal into core neural frequency bands (Delta, Theta, Alpha/Mu, Beta, Gamma) without destroying the temporal sequence.
    """)
    display_figure("Raw EEG Traces", "01_raw_eeg.png", "Raw, unfiltered microvolt signals recorded from the motor cortex.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        display_figure("Filter Bank Responses", "03_filter_bank.png", "Frequency tuning curves for our digital filters.")
    with col2:
        display_figure("Signal Component Extraction", "02_band_decomposition.png", "A single brainwave channel broken computationally into sub-bands.")

# -------------------------------------------------------------------
# TAB 2: Spectral Analysis
# -------------------------------------------------------------------
with tab2:
    st.header("Frequency-Domain Activity")
    st.markdown("""
    A pure time-domain view is insufficient for decoding brain waves. Fourier transforms allow us to view the exact composition of frequencies present during different intents.
    """)
    display_figure("Power Spectral Density (PSD)", "04_psd.png", "A noise-robust quantification of brainwave power at each frequency step.")
    st.markdown("---")
    display_figure("STFT Spectrogram", "05_spectrogram_C3.png", "A sliding map showing how frequency power clusters over time.")

# -------------------------------------------------------------------
# TAB 3: Wavelet Transforms
# -------------------------------------------------------------------
with tab3:
    st.header("Dynamic Neural Oscillations (CWT)")
    st.markdown("""
    The **Continuous Wavelet Transform** using the **Morlet wavelet** offers dynamic resolution. It adapts natively to neural oscillations, capturing sharp sudden anomalies while perfectly resolving low sustained brainwaves.
    """)
    display_figure("High-Resolution Neural Power Map", "06_cwt_C3.png", "Time-frequency magnitude of the brain wave signals.")

# -------------------------------------------------------------------
# TAB 4: Neural Dynamics
# -------------------------------------------------------------------
with tab4:
    st.header("Motor Intent Recognition")
    st.markdown("""
    This is the core neural phenomenon that powers BCI control. When someone imagines moving their physical hand, the **Mu band (8-12 Hz)** activity physically *decreases* in the opposite hemisphere of the brain. We extract this dip mathematically to use for classification.
    """)
    display_figure("Hemispheric Motor Response", "07_erd_comparison.png", "Notice the significant dip in electrical power compared to rest states.")

# -------------------------------------------------------------------
# TAB 5: Decoding Performance
# -------------------------------------------------------------------
with tab5:
    st.header("Machine Learning Classification Results")
    st.markdown("""
    Our classifier utilizes **Common Spatial Patterns (CSP)** to process all 64 electrodes simultaneously. CSP finds optimal spatial filters that amplify the target signal (e.g. Left Movement) while suppressing the opposite signal (Right Movement), drastically outperforming single-electrode methods.
    """)
    
    display_figure("System Performance Summary", "11_summary_dashboard.png")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        display_figure("Spatial Filter Characteristics", "08_csp_patterns.png", "The physical brain regions being mathematically amplified by our model to separate intent.")
    with col2:
        display_figure("Classification Confusion Matrix", "09_confusion_matrix.png", "Model accuracy breakdown across classes.")
