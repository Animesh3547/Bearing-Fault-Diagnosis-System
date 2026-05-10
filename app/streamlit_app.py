import sys
import os

sys.path.append(

    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


import streamlit as st
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torch
from torchvision import transforms
from PIL import Image

from src.model import BearingCNN

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

# ==========================================
# LOAD MODEL
# ==========================================

model = BearingCNN().to(device)

model.load_state_dict(
    torch.load(
        "../models/bearing_cnn.pth",
        map_location=device
    )
)

model.eval()

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "ball",
    "inner",
    "normal",
    "outer"
]


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Bearing Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================
# TITLE SECTION
# ==========================================

st.title(
    "Bearing Fault Diagnosis & "
    "Predictive Maintenance System"
)

st.markdown("""
Vibration-based fault diagnosis using:

- STFT Spectrogram Analysis
- CNN-Based Classification
- Predictive Maintenance Concepts
- Cross-Domain Robustness Evaluation
""")

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("System Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload .mat vibration signal",
    type=["mat"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.info("""
Model: CNN

Input:
128 × 128 Grayscale Spectrogram

Classes:
- Normal
- Inner Race
- Ball Fault
- Outer Race
""")

st.sidebar.markdown("---")

st.sidebar.warning("""
Recommended Input:

- 12 kHz vibration signals
- Similar preprocessing conditions
- CWRU-like signal characteristics
""")

# ==========================================
# MAIN DASHBOARD
# ==========================================

# ------------------------------------------
# SECTION 1 — SIGNAL INFO
# ------------------------------------------

st.header("Signal Information")

signal_col1, signal_col2 = st.columns(2)

with signal_col1:

    st.metric(
        label="Sampling Frequency",
        value="12 kHz"
    )

with signal_col2:

    st.metric(
        label="Window Size",
        value="12000"
    )

st.divider()



# ==========================================
#   SECTION 2 — LOAD + PROCESS SIGNAL
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # LOAD .MAT FILE
    # --------------------------------------

    mat_data = sio.loadmat(uploaded_file)

    # --------------------------------------
    # FIND SIGNAL KEY
    # --------------------------------------

    signal_key = None

    for key in mat_data.keys():

        if "DE" in key:

            signal_key = key
            break

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    if signal_key is None:

        st.error(
            "Could not find vibration signal "
            "inside uploaded .mat file."
        )

    else:

        # --------------------------------------
        # EXTRACT SIGNAL
        # --------------------------------------

        signal = mat_data[signal_key]

        signal = signal.flatten()

        # --------------------------------------
        # SIGNAL INFO
        # --------------------------------------

        st.write(
            f"Loaded Signal Key: {signal_key}"
        )

        st.write(
            f"Signal Length: "
            f"{len(signal)} samples"
        )

        # --------------------------------------
        # SIGNAL WINDOW
        # --------------------------------------

        window_size = 12000

        signal_window = signal[:window_size]

        # --------------------------------------
        # STFT
        # --------------------------------------

        frequencies, times, Sxx = spectrogram(

            signal_window,

            fs=12000,

            nperseg=256,

            noverlap=128

        )

        # --------------------------------------
        # LOG SCALE
        # --------------------------------------

        Sxx_log = 10 * np.log10(
            Sxx + 1e-10
        )

        # ==========================================
        # SIDE-BY-SIDE VISUALIZATION
        # ==========================================

        signal_col, spec_col = st.columns(2)

        # --------------------------------------
        # RAW SIGNAL
        # --------------------------------------

        with signal_col:

            st.subheader(
                "Raw Vibration Signal"
            )

            fig, ax = plt.subplots(
                figsize=(6, 3)
            )

            ax.plot(signal_window)

            ax.set_xlabel(
                "Samples"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            st.pyplot(fig)

        # --------------------------------------
        # SPECTROGRAM
        # --------------------------------------

        with spec_col:

            st.subheader(
                "STFT Spectrogram"
            )

            fig2, ax2 = plt.subplots(
                figsize=(6, 3)
            )

            im = ax2.pcolormesh(

                times,

                frequencies,

                Sxx_log,

                shading='gouraud'

            )

            ax2.set_xlabel(
                "Time"
            )

            ax2.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig2)

else:

    st.info(
        "Upload a .mat file to "
        "visualize vibration signal "
        "and spectrogram."
    )


st.divider()

# ------------------------------------------
# SECTION 4 — PREDICTION PANEL
# ------------------------------------------

st.header("Fault Prediction")

if uploaded_file is not None and signal_key is not None:

    # ==========================================
    # CREATE TEMP SPECTROGRAM IMAGE
    # ==========================================

    fig3, ax3 = plt.subplots(
        figsize=(4, 4)
    )

    ax3.pcolormesh(

        times,

        frequencies,

        Sxx_log,

        shading='gouraud'

    )

    ax3.axis("off")

    fig3.savefig(

        "temp_spectrogram.png",

        bbox_inches='tight',

        pad_inches=0

    )

    plt.close(fig3)

    # ==========================================
    # IMAGE TRANSFORM
    # ==========================================

    transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize((128, 128)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    # ==========================================
    # LOAD IMAGE
    # ==========================================

    image = Image.open(
        "temp_spectrogram.png"
    )

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    # ==========================================
    # MODEL INFERENCE
    # ==========================================

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    # ==========================================
    # RESULTS
    # ==========================================

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    # ==========================================
    # DISPLAY
    # ==========================================

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:

        st.metric(
            label="Predicted Fault",
            value=predicted_class
        )

    with pred_col2:

        st.metric(
            label="Confidence",
            value=f"{confidence_score:.2f}%"
        )

    # ==========================================
    # PROBABILITY DISPLAY
    # ==========================================

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        prob = (
            probabilities[0][i].item()
            * 100
        )

        st.progress(
            int(prob)
        )

        st.write(
            f"{class_name}: "
            f"{prob:.2f}%"
        )

else:

    st.info(
        "Upload signal to run "
        "CNN inference."
    )




st.divider()

# ------------------------------------------
# SECTION 5 — ENGINEERING DETAILS
# ------------------------------------------

with st.expander(
    " Technical Details"
):

    st.markdown("""
### Signal Processing

- Sliding-window segmentation
- STFT spectrogram generation
- Log scaling
- Grayscale preprocessing

### CNN Pipeline

- Convolutional feature extraction
- Spectrogram texture learning
- Multi-class fault classification

### Evaluation

- Same-domain testing
- Noise robustness analysis
- Cross-domain IMS validation
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Model trained primarily on "
    "CWRU bearing dataset. "
    "Cross-domain industrial "
    "generalization remains an "
    "active research challenge."
)