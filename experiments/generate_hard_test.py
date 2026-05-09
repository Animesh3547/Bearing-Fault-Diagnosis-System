import os
import torch

from PIL import Image, ImageFilter, ImageEnhance

from torchvision import transforms

# ==========================================
# PATHS
# ==========================================

input_base = "../dataset_gray/test"
output_base = "../dataset_hard"

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
# NOISE TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x + 0.06 * torch.randn_like(x)
    )
])

to_pil = transforms.ToPILImage()

# ==========================================
# PROCESS
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

        # ----------------------------------
        # BLUR
        # ----------------------------------

        image = image.filter(
            ImageFilter.GaussianBlur(radius=1)
        )

        # ----------------------------------
        # CONTRAST REDUCTION
        # ----------------------------------

        enhancer = ImageEnhance.Contrast(image)

        image = enhancer.enhance(0.7)

        # ----------------------------------
        # ADD NOISE
        # ----------------------------------

        tensor = transform(image)

        tensor = torch.clamp(tensor, 0, 1)

        hard_image = to_pil(tensor)

        save_path = os.path.join(
            output_folder,
            file_name
        )

        hard_image.save(save_path)

print("\nHard test dataset generated.")