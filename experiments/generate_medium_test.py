import os
import torch

from PIL import Image

from torchvision import transforms

# ==========================================
# PATHS
# ==========================================

input_base = "../dataset_gray/test"
output_base = "../dataset_medium"

classes = ["normal", "inner", "ball", "outer"]

# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

for cls in classes:

    os.makedirs(
        os.path.join(output_base, cls),
        exist_ok=True
    )

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x + 0.02 * torch.randn_like(x)
    )
])

to_pil = transforms.ToPILImage()

# ==========================================
# PROCESS IMAGES
# ==========================================

for cls in classes:

    input_folder = os.path.join(
        input_base,
        cls
    )

    output_folder = os.path.join(
        output_base,
        cls
    )

    for file_name in os.listdir(input_folder):

        if not file_name.endswith(".png"):
            continue

        image_path = os.path.join(
            input_folder,
            file_name
        )

        image = Image.open(image_path)

        tensor = transform(image)

        tensor = torch.clamp(tensor, 0, 1)

        noisy_image = to_pil(tensor)

        save_path = os.path.join(
            output_folder,
            file_name
        )

        noisy_image.save(save_path)

print("\nMedium test dataset generated.")