import torch
import torch.nn as nn
import torch.optim as optim

from config import *

from model import BearingCNN

from dataloader import train_loader

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

# ==========================================
# MODEL
# ==========================================

model = BearingCNN().to(device)

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print("\nModel saved successfully.")