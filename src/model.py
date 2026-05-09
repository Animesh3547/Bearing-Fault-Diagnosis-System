import torch
import torch.nn as nn

# ==========================================
# CNN MODEL
# ==========================================

class BearingCNN(nn.Module):

    def __init__(self):

        super(BearingCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 32 * 32, 128),

            nn.ReLU(),

            nn.Linear(128, 4)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x