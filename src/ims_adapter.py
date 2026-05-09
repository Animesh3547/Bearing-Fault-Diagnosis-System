import pandas as pd
import numpy as np

from scipy.signal import resample
from scipy.signal import spectrogram

import matplotlib.pyplot as plt

from PIL import Image

import io

# ==========================================
# TRAINING PARAMETERS
# ==========================================

TARGET_FS = 12000

WINDOW_SIZE = 12000

NPERSEG = 256
NOVERLAP = 128

# ==========================================
# IMS ORIGINAL SETTINGS
# ==========================================

IMS_FS = 20000

# ==========================================
# FUNCTION
# ==========================================

def generate_ims_spectrogram(
    file_path,
    channel=4
):

    # --------------------------------------
    # LOAD ASCII FILE
    # --------------------------------------

    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )

    # --------------------------------------
    # SELECT CHANNEL
    # --------------------------------------

    signal = data[channel].values

    # --------------------------------------
    # RESAMPLE
    # 20k -> 12k
    # --------------------------------------

    target_length = int(
        len(signal)
        * TARGET_FS
        / IMS_FS
    )

    signal = resample(
        signal,
        target_length
    )

    # --------------------------------------
    # USE FIRST WINDOW
    # --------------------------------------

    signal = signal[:WINDOW_SIZE]

    # --------------------------------------
    # GENERATE SPECTROGRAM
    # --------------------------------------

    frequencies, times, Sxx = spectrogram(
        signal,
        fs=TARGET_FS,
        nperseg=NPERSEG,
        noverlap=NOVERLAP
    )

    # --------------------------------------
    # LOG SCALE
    # --------------------------------------

    Sxx = np.log(
        Sxx + 1e-10
    )

    # --------------------------------------
    # PLOT
    # --------------------------------------

    fig = plt.figure(figsize=(4,4))

    plt.pcolormesh(
        times,
        frequencies,
        Sxx,
        shading='gouraud'
    )

    plt.axis('off')

    plt.tight_layout()

    # --------------------------------------
    # SAVE TO BUFFER
    # --------------------------------------

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format='png',
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)

    buffer.seek(0)

    image = Image.open(buffer)

    return image