import torch

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from config import *

# ==========================================
# TRAIN TRANSFORM
# ==========================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
# TEST TRANSFORM
# ==========================================

test_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ==========================================
# DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# INFO
# ==========================================

if __name__ == "__main__":

    print(train_dataset.class_to_idx)

    print("Train:", len(train_dataset))
    print("Test:", len(test_dataset))