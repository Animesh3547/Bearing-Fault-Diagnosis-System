
# import os
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt

# from scipy.signal import spectrogram

# # =========================================================
# # PARAMETERS
# # =========================================================

# fs = 12000

# window_size = 12000
# step_size = 9000

# nperseg = 256
# noverlap = 128

# # =========================================================
# # INPUT / OUTPUT DIRECTORIES
# # =========================================================

# input_base_dir = "../data"
# output_base_dir = "../dataset"

# classes = ["normal", "inner", "ball", "outer"]
# splits = ["train", "test"]

# # =========================================================
# # CREATE OUTPUT FOLDERS
# # =========================================================

# for split in splits:

#     for cls in classes:

#         os.makedirs(
#             os.path.join(output_base_dir, split, cls),
#             exist_ok=True
#         )

# # =========================================================
# # IMAGE COUNTER
# # =========================================================

# image_counter = {
#     split: {cls: 0 for cls in classes}
#     for split in splits
# }

# # =========================================================
# # PROCESS TRAIN + TEST DATA
# # =========================================================

# for split in splits:

#     print(f"\n==============================")
#     print(f"PROCESSING {split.upper()} DATA")
#     print(f"==============================")

#     for cls in classes:

#         print(f"\nClass: {cls}")

#         input_folder = os.path.join(
#             input_base_dir,
#             split,
#             cls
#         )

#         output_folder = os.path.join(
#             output_base_dir,
#             split,
#             cls
#         )

#         # -------------------------------------------------
#         # GET MATLAB FILES
#         # -------------------------------------------------

#         mat_files = sorted([
#             f for f in os.listdir(input_folder)
#             if f.endswith(".mat")
#         ])

#         print(f"Found {len(mat_files)} files")

#         # -------------------------------------------------
#         # PROCESS EACH FILE
#         # -------------------------------------------------

#         for file_name in mat_files:

#             file_path = os.path.join(
#                 input_folder,
#                 file_name
#             )

#             print(f"Processing: {file_name}")

#             # ---------------------------------------------
#             # LOAD MATLAB FILE
#             # ---------------------------------------------

#             data = sio.loadmat(file_path)

#             # ---------------------------------------------
#             # FIND DRIVE-END SIGNAL KEY
#             # ---------------------------------------------

#             de_key = None

#             for key in data.keys():

#                 if "DE_time" in key:

#                     de_key = key
#                     break

#             if de_key is None:

#                 print(f"DE_time not found in {file_name}")
#                 continue

#             # ---------------------------------------------
#             # EXTRACT SIGNAL
#             # ---------------------------------------------

#             signal = data[de_key].squeeze()

#             # ---------------------------------------------
#             # TRIM SIGNAL
#             # ---------------------------------------------
#             if cls == "normal":
#                 signal = signal[:240000]
#             else:
#                 signal = signal[:120000]

#             # ---------------------------------------------
#             # WINDOW COUNT
#             # ---------------------------------------------

#             num_windows = (
#                 (len(signal) - window_size)
#                 // step_size
#             ) + 1

#             print(f"Generated {num_windows} windows")

#             # ---------------------------------------------
#             # GENERATE WINDOWS
#             # ---------------------------------------------

#             for i in range(num_windows):

#                 start = i * step_size
#                 end = start + window_size

#                 segment = signal[start:end]

#                 # -----------------------------------------
#                 # SPECTROGRAM
#                 # -----------------------------------------

#                 frequencies, times, Sxx = spectrogram(
#                     segment,
#                     fs=fs,
#                     nperseg=nperseg,
#                     noverlap=noverlap
#                 )

#                 # -----------------------------------------
#                 # LOG SCALE
#                 # -----------------------------------------

#                 Sxx = np.log(Sxx + 1e-10)

#                 # -----------------------------------------
#                 # IMAGE COUNT
#                 # -----------------------------------------

#                 image_counter[split][cls] += 1

#                 save_name = (
#                     f"{cls}_"
#                     f"{image_counter[split][cls]:05d}.png"
#                 )

#                 save_path = os.path.join(
#                     output_folder,
#                     save_name
#                 )

#                 # -----------------------------------------
#                 # SAVE IMAGE
#                 # -----------------------------------------

#                 plt.figure(figsize=(4,4))

#                 plt.pcolormesh(
#                     times,
#                     frequencies,
#                     Sxx,
#                     shading='gouraud'
#                 )

#                 plt.axis('off')

#                 plt.tight_layout()

#                 plt.savefig(
#                     save_path,
#                     bbox_inches='tight',
#                     pad_inches=0
#                 )

#                 plt.close()

# # =========================================================
# # FINAL SUMMARY
# # =========================================================

# print("\n===================================")
# print("DATASET GENERATION COMPLETE")
# print("===================================")

# for split in splits:

#     print(f"\n{split.upper()} DATASET")

#     total_images = 0

#     for cls in classes:

#         count = image_counter[split][cls]

#         total_images += count

#         print(f"{cls}: {count}")

#     print(f"Total: {total_images}")







# import os
# from PIL import Image

# # ==========================================
# # INPUT / OUTPUT
# # ==========================================

# input_base = "../dataset"
# output_base = "../dataset_gray"

# classes = ["normal", "inner", "ball", "outer"]
# splits = ["train", "test"]

# # ==========================================
# # CREATE OUTPUT DIRECTORIES
# # ==========================================

# for split in splits:

#     for cls in classes:

#         os.makedirs(
#             os.path.join(output_base, split, cls),
#             exist_ok=True
#         )

# # ==========================================
# # CONVERT IMAGES
# # ==========================================

# for split in splits:

#     for cls in classes:

#         input_folder = os.path.join(
#             input_base,
#             split,
#             cls
#         )

#         output_folder = os.path.join(
#             output_base,
#             split,
#             cls
#         )

#         for file_name in os.listdir(input_folder):

#             if not file_name.endswith(".png"):
#                 continue

#             input_path = os.path.join(
#                 input_folder,
#                 file_name
#             )

#             output_path = os.path.join(
#                 output_folder,
#                 file_name
#             )

#             # ---------------------------------
#             # LOAD IMAGE
#             # ---------------------------------

#             img = Image.open(input_path)

#             # ---------------------------------
#             # FORCE SINGLE CHANNEL GRAYSCALE
#             # ---------------------------------

#             gray_img = img.convert("L")

#             # ---------------------------------
#             # SAVE
#             # ---------------------------------

#             gray_img.save(output_path)

# print("\nGrayscale conversion complete.")





import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# ==========================================
# TRANSFORMS
# ==========================================

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x + 0.02 * torch.randn_like(x)
    ),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ==========================================
# LOAD TRAIN DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    root="../dataset_gray/train",
    transform=transform
)

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    root="../dataset_gray/test",
    transform=transform
)

# ==========================================
# CLASS NAMES
# ==========================================

print(train_dataset.class_to_idx)

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# ==========================================
# CHECK SHAPES
# ==========================================

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

import torch
import torch.nn as nn

# ==========================================
# CNN MODEL
# ==========================================

class BearingCNN(nn.Module):

    def __init__(self):

        super(BearingCNN, self).__init__()

        # ----------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------

        self.features = nn.Sequential(

            # Conv Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # Conv Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        # ----------------------------------
        # CLASSIFIER
        # ----------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 32 * 32, 128),

            nn.ReLU(),

            nn.Linear(128, 4)
        )

    # ======================================
    # FORWARD PASS
    # ======================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ==========================================
# CREATE MODEL
# ==========================================

model = BearingCNN()

print(model)

dummy_input = torch.randn(16, 1, 128, 128)

output = model(dummy_input)

print(output.shape)


import torch.optim as optim

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

# ==========================================
# MOVE MODEL
# ==========================================

model = model.to(device)

# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Setup complete.")

# ==========================================
# TRAINING LOOP
# ==========================================

num_epochs = 10

for epoch in range(num_epochs):

    # --------------------------------------
    # TRAIN MODE
    # --------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # --------------------------------------
    # LOOP THROUGH BATCHES
    # --------------------------------------

    for images, labels in train_loader:

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------
        # RESET GRADIENTS
        # ----------------------------------

        optimizer.zero_grad()

        # ----------------------------------
        # FORWARD PASS
        # ----------------------------------

        outputs = model(images)

        # ----------------------------------
        # LOSS
        # ----------------------------------

        loss = criterion(outputs, labels)

        # ----------------------------------
        # BACKPROPAGATION
        # ----------------------------------

        loss.backward()

        # ----------------------------------
        # UPDATE WEIGHTS
        # ----------------------------------

        optimizer.step()

        # ----------------------------------
        # TRACK LOSS
        # ----------------------------------

        running_loss += loss.item()

        # ----------------------------------
        # PREDICTIONS
        # ----------------------------------

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    # --------------------------------------
    # EPOCH METRICS
    # --------------------------------------

    epoch_loss = running_loss / len(train_loader)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


print(len(test_dataset))




from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# EVALUATION MODE
# ==========================================

model.eval()

all_labels = []
all_predictions = []

correct = 0
total = 0

# ==========================================
# DISABLE GRADIENTS
# ==========================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------
        # FORWARD PASS
        # ----------------------------------

        outputs = model(images)

        # ----------------------------------
        # PREDICTIONS
        # ----------------------------------

        _, predicted = torch.max(outputs, 1)

        # ----------------------------------
        # STORE RESULTS
        # ----------------------------------

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # ----------------------------------
        # ACCURACY
        # ----------------------------------

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

# ==========================================
# FINAL TEST ACCURACY
# ==========================================

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

class_names = [
    "ball",
    "inner",
    "normal",
    "outer"
]

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)