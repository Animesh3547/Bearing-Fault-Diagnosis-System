import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

from PIL import Image

import io

# ==========================================
# PARAMETERS
# ==========================================

fs = 12000

window_size = 12000
step_size = 9000

nperseg = 256
noverlap = 128

# ==========================================
# FUNCTION
# ==========================================

def generate_spectrogram_from_mat(
    mat_path
):

    # --------------------------------------
    # LOAD MATLAB FILE
    # --------------------------------------

    data = sio.loadmat(mat_path)

    # --------------------------------------
    # FIND DE_time KEY
    # --------------------------------------

    de_key = None

    for key in data.keys():

        if "DE_time" in key:

            de_key = key
            break

    if de_key is None:

        raise ValueError(
            "DE_time key not found."
        )

    # --------------------------------------
    # EXTRACT SIGNAL
    # --------------------------------------

    signal = data[de_key].squeeze()

    # --------------------------------------
    # USE FIRST WINDOW
    # --------------------------------------

    segment = signal[:window_size]

    # --------------------------------------
    # GENERATE SPECTROGRAM
    # --------------------------------------

    frequencies, times, Sxx = spectrogram(
        segment,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap
    )

    # --------------------------------------
    # LOG SCALE
    # --------------------------------------

    Sxx = np.log(Sxx + 1e-10)

    # --------------------------------------
    # PLOT TO MEMORY
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

    # --------------------------------------
    # RETURN PIL IMAGE
    # --------------------------------------

    image = Image.open(buffer)

    return image