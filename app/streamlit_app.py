import streamlit as st

import torch

from torchvision import transforms

import tempfile

import sys

# ==========================================
# ADD SRC PATH
# ==========================================

sys.path.append("../src")

from model import BearingCNN

from config import *

from signal_to_spectrogram import (
    generate_spectrogram_from_mat
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Bearing Fault Diagnosis",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title(
    "Bearing Fault Diagnosis System"
)

st.write(
    "CNN-based predictive maintenance "
    "using vibration signal analysis."
)

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource

def load_model():

    model = BearingCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_SAVE_PATH,
            map_location=device
        )
    )

    model.eval()

    return model

model = load_model()

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload MATLAB Vibration File (.mat)",
    type=["mat"]
)

# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # SAVE TEMP FILE
    # --------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mat"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        temp_path = tmp_file.name

    # --------------------------------------
    # GENERATE SPECTROGRAM
    # --------------------------------------

    spectrogram_image = (
        generate_spectrogram_from_mat(
            temp_path
        )
    )

    # --------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------

    st.subheader("Generated Spectrogram")

    st.image(
        spectrogram_image,
        use_container_width=True
    )

    # --------------------------------------
    # PREPROCESS
    # --------------------------------------

    input_tensor = transform(
        spectrogram_image
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # --------------------------------------
    # INFERENCE
    # --------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    # --------------------------------------
    # RESULTS
    # --------------------------------------

    st.subheader("Prediction")

    st.success(
        f"Fault Type: "
        f"{predicted_class}"
    )

    st.info(
        f"Confidence: "
        f"{confidence_score:.2f}%"
    )

    # --------------------------------------
    # PROBABILITIES
    # --------------------------------------

    st.subheader("Class Probabilities")

    probs = (
        probabilities.squeeze()
        .cpu()
        .numpy()
    )

    for cls, prob in zip(
        CLASS_NAMES,
        probs
    ):

        st.write(
            f"{cls}: "
            f"{prob*100:.2f}%"
        )