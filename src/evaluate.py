import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from config import *

from model import BearingCNN

from dataloader import test_loader

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

# ==========================================
# EVALUATION
# ==========================================

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

# ==========================================
# ACCURACY
# ==========================================

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================
# REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES
    )
)