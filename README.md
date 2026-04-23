# NeuroDecode

> **Brain-Computer Interface for decoding human intent natively from live EEG brain signals**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![MNE](https://img.shields.io/badge/MNE--Python-1.6-009688)](https://mne.tools/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

NeuroDecode is an advanced machine learning pipeline that translates human intent directly from 64-channel EEG recordings in real-time. By tracking sensory rhythms and event-related synchronizations across the motor cortex, it acts as a foundational architecture for controlling physical applications using just thought.

## Tech Stack

- **Core Signal Processing:** MNE-Python, SciPy (Butterworth Filters, Power Spectral Density)
- **Machine Learning:** Scikit-Learn (Common Spatial Patterns, Random Forest Classifier)
- **Mathematical Modeling:** NumPy, PyWavelets (Continuous Wavelet Transforms)
- **Frontend Dashboard:** Streamlit
- **Visualization:** Matplotlib

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankitgpt18/NeuroDecode.git
   cd NeuroDecode
   ```

2. **Environment Setup**
   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Analysis Metrics**
   Run the backend pipeline to calculate models and export high-resolution dashboard visuals:
   ```bash
   python main.py
   ```

4. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
   - **App:** http://localhost:8501

## License
This project is licensed under the MIT License.
