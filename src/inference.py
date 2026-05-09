import torch

from PIL import Image

from torchvision import transforms

from model import BearingCNN

from config import *

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

# ==========================================
# LOAD MODEL
# ==========================================

model = BearingCNN().to(device)

model.load_state_dict(
    torch.load(MODEL_SAVE_PATH)
)

model.eval()

print("\nModel loaded successfully.")

import os
import random

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ==========================================
# SELECT TEST TYPE
# ==========================================

test_type = "hard"

# ==========================================
# DATASET PATH
# ==========================================

if test_type == "easy":

    base_path = "../dataset_gray/test"

elif test_type == "medium":

    base_path = "../dataset_medium"

elif test_type == "hard":

    base_path = "../dataset_hard"

else:

    raise ValueError("Invalid test type.")

# ==========================================
# RANDOM TESTING
# ==========================================

print("\n==============================")
print(f"RANDOM {test_type.upper()} TESTING")
print("==============================")

num_tests = 10

correct = 0

for i in range(num_tests):

    # --------------------------------------
    # RANDOM CLASS
    # --------------------------------------

    true_class = random.choice(CLASS_NAMES)

    class_folder = os.path.join(
        base_path,
        true_class
    )

    # --------------------------------------
    # RANDOM IMAGE
    # --------------------------------------

    image_name = random.choice(
        os.listdir(class_folder)
    )

    image_path = os.path.join(
        class_folder,
        image_name
    )

    # --------------------------------------
    # LOAD IMAGE
    # --------------------------------------

    image = Image.open(image_path)

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    # --------------------------------------
    # PREDICTION
    # --------------------------------------

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

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    # --------------------------------------
    # CORRECTNESS
    # --------------------------------------

    is_correct = (
        predicted_class == true_class
    )

    if is_correct:

        correct += 1

    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print(f"\nTest {i+1}")

    print(f"Image: {image_name}")

    print(f"Actual: {true_class}")

    print(f"Predicted: {predicted_class}")

    print(
        f"Confidence: "
        f"{confidence_score:.2f}%"
    )

    print(f"Correct: {is_correct}")

# ==========================================
# FINAL SUMMARY
# ==========================================

accuracy = (correct / num_tests) * 100

print("\n==============================")
print("FINAL RANDOM TEST RESULTS")
print("==============================")

print(
    f"Correct Predictions: "
    f"{correct}/{num_tests}"
)

print(
    f"Accuracy: "
    f"{accuracy:.2f}%"
)

print("==============================")