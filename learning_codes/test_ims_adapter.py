from ims_adapter import (
    generate_ims_spectrogram
)

import matplotlib.pyplot as plt

# ==========================================
# FILE PATH
# ==========================================

file_path = (
    "../data_external/ims/"
    "1st_test/"
    "2003.11.25.12.27.32"
)

# ==========================================
# GENERATE SPECTROGRAM
# ==========================================

image = generate_ims_spectrogram(
    file_path=file_path,
    channel=4
)

# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(6,6))

plt.imshow(image)

plt.axis('off')

plt.title(
    "IMS Generated Spectrogram"
)

plt.show()