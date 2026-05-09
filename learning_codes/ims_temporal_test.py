import os

import torch

from torchvision import transforms

from model import BearingCNN

from config import *

from ims_adapter import (
    generate_ims_spectrogram
)

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
        MODEL_SAVE_PATH,
        map_location=device
    )
)

model.eval()

print("\nModel loaded.")

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
# IMS DIRECTORY
# ==========================================

ims_dir = (
    "../data_external/ims/1st_test"
)

all_files = sorted(
    os.listdir(ims_dir)
)

# ==========================================
# SELECT FILES
# ==========================================

early_files = all_files[:10]

middle_start = len(all_files)//2

middle_files = all_files[
    middle_start:middle_start+10
]

late_files = all_files[-10:]

# ==========================================
# TEST FUNCTION
# ==========================================

def run_stage_test(
    files,
    stage_name,
    channel,
    expected_fault
):

    print("\n================================")
    print(f"{stage_name} STAGE")
    print(
        f"Expected: {expected_fault}"
    )
    print("================================")

    for file_name in files:

        file_path = os.path.join(
            ims_dir,
            file_name
        )

        # ----------------------------------
        # GENERATE SPECTROGRAM
        # ----------------------------------

        image = generate_ims_spectrogram(
            file_path=file_path,
            channel=channel
        )

        # ----------------------------------
        # PREPROCESS
        # ----------------------------------

        tensor = transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)

        # ----------------------------------
        # INFERENCE
        # ----------------------------------

        with torch.no_grad():

            outputs = model(tensor)

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

        # ----------------------------------
        # PRINT
        # ----------------------------------

        print(f"\nFile: {file_name}")

        print(
            f"Prediction: "
            f"{predicted_class}"
        )

        print(
            f"Confidence: "
            f"{confidence_score:.2f}%"
        )

# ==========================================
# BEARING 3 TEST
# INNER-LIKE
# ==========================================

print("\n\n############################")
print("IMS BEARING 3 ANALYSIS")
print("############################")

run_stage_test(
    early_files,
    "EARLY",
    channel=4,
    expected_fault="normal/healthy-ish"
)

run_stage_test(
    middle_files,
    "MIDDLE",
    channel=4,
    expected_fault="degrading"
)

run_stage_test(
    late_files,
    "LATE",
    channel=4,
    expected_fault="inner-like"
)

# ==========================================
# BEARING 4 TEST
# BALL-LIKE
# ==========================================

print("\n\n############################")
print("IMS BEARING 4 ANALYSIS")
print("############################")

run_stage_test(
    early_files,
    "EARLY",
    channel=6,
    expected_fault="normal/healthy-ish"
)

run_stage_test(
    middle_files,
    "MIDDLE",
    channel=6,
    expected_fault="degrading"
)

run_stage_test(
    late_files,
    "LATE",
    channel=6,
    expected_fault="ball-like"
)