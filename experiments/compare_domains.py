from ims_adapter import (
    generate_ims_spectrogram
)

from PIL import Image

import matplotlib.pyplot as plt

# ==========================================
# LOAD CWRU IMAGES
# ==========================================

cwru_normal = Image.open(
    "../dataset_gray/train/normal/normal_00001.png"
)

cwru_inner = Image.open(
    "../dataset_gray/train/inner/inner_00001.png"
)

cwru_ball = Image.open(
    "../dataset_gray/train/ball/ball_00001.png"
)

# ==========================================
# IMS FILE
# ==========================================

ims_file = (
    "../data_external/ims/"
    "1st_test/"
    "2003.11.25.23.39.56"
)

# ==========================================
# GENERATE IMS SPECTROGRAMS
# ==========================================

ims_normal = generate_ims_spectrogram(
    file_path=ims_file,
    channel=0
)

ims_inner = generate_ims_spectrogram(
    file_path=ims_file,
    channel=4
)

ims_ball = generate_ims_spectrogram(
    file_path=ims_file,
    channel=6
)

# ==========================================
# PLOT COMPARISON
# ==========================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12,8)
)

# ------------------------------------------
# TOP ROW - CWRU
# ------------------------------------------

axes[0,0].imshow(cwru_normal)
axes[0,0].set_title("CWRU Normal")
axes[0,0].axis('off')

axes[0,1].imshow(cwru_inner)
axes[0,1].set_title("CWRU Inner")
axes[0,1].axis('off')

axes[0,2].imshow(cwru_ball)
axes[0,2].set_title("CWRU Ball")
axes[0,2].axis('off')

# ------------------------------------------
# BOTTOM ROW - IMS
# ------------------------------------------

axes[1,0].imshow(ims_normal)
axes[1,0].set_title("IMS Bearing 1")
axes[1,0].axis('off')

axes[1,1].imshow(ims_inner)
axes[1,1].set_title("IMS Bearing 3")
axes[1,1].axis('off')

axes[1,2].imshow(ims_ball)
axes[1,2].set_title("IMS Bearing 4")
axes[1,2].axis('off')

# ------------------------------------------
# FINALIZE
# ------------------------------------------

plt.tight_layout()

plt.show()