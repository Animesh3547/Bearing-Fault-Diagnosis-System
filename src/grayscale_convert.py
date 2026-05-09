
import os
from PIL import Image

# ==========================================
# INPUT / OUTPUT
# ==========================================

input_base = "../dataset"
output_base = "../dataset_gray"

classes = ["normal", "inner", "ball", "outer"]
splits = ["train", "test"]

# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

for split in splits:

    for cls in classes:

        os.makedirs(
            os.path.join(output_base, split, cls),
            exist_ok=True
        )

# ==========================================
# CONVERT IMAGES
# ==========================================

for split in splits:

    for cls in classes:

        input_folder = os.path.join(
            input_base,
            split,
            cls
        )

        output_folder = os.path.join(
            output_base,
            split,
            cls
        )

        for file_name in os.listdir(input_folder):

            if not file_name.endswith(".png"):
                continue

            input_path = os.path.join(
                input_folder,
                file_name
            )

            output_path = os.path.join(
                output_folder,
                file_name
            )

            # ---------------------------------
            # LOAD IMAGE
            # ---------------------------------

            img = Image.open(input_path)

            # ---------------------------------
            # FORCE SINGLE CHANNEL GRAYSCALE
            # ---------------------------------

            gray_img = img.convert("L")

            # ---------------------------------
            # SAVE
            # ---------------------------------

            gray_img.save(output_path)

print("\nGrayscale conversion complete.")