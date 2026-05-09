import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

# =========================================================
# PARAMETERS
# =========================================================

fs = 12000

window_size = 12000
step_size = 9000

nperseg = 256
noverlap = 128

# =========================================================
# INPUT / OUTPUT DIRECTORIES
# =========================================================

input_base_dir = "../data"
output_base_dir = "../dataset"

classes = ["normal", "inner", "ball", "outer"]
splits = ["train", "test"]

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

for split in splits:

    for cls in classes:

        os.makedirs(
            os.path.join(output_base_dir, split, cls),
            exist_ok=True
        )

# =========================================================
# IMAGE COUNTER
# =========================================================

image_counter = {
    split: {cls: 0 for cls in classes}
    for split in splits
}

# =========================================================
# PROCESS TRAIN + TEST DATA
# =========================================================

for split in splits:

    print(f"\n==============================")
    print(f"PROCESSING {split.upper()} DATA")
    print(f"==============================")

    for cls in classes:

        print(f"\nClass: {cls}")

        input_folder = os.path.join(
            input_base_dir,
            split,
            cls
        )

        output_folder = os.path.join(
            output_base_dir,
            split,
            cls
        )

        # -------------------------------------------------
        # GET MATLAB FILES
        # -------------------------------------------------

        mat_files = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith(".mat")
        ])

        print(f"Found {len(mat_files)} files")

        # -------------------------------------------------
        # PROCESS EACH FILE
        # -------------------------------------------------

        for file_name in mat_files:

            file_path = os.path.join(
                input_folder,
                file_name
            )

            print(f"Processing: {file_name}")

            # ---------------------------------------------
            # LOAD MATLAB FILE
            # ---------------------------------------------

            data = sio.loadmat(file_path)

            # ---------------------------------------------
            # FIND DRIVE-END SIGNAL KEY
            # ---------------------------------------------

            de_key = None

            for key in data.keys():

                if "DE_time" in key:

                    de_key = key
                    break

            if de_key is None:

                print(f"DE_time not found in {file_name}")
                continue

            # ---------------------------------------------
            # EXTRACT SIGNAL
            # ---------------------------------------------

            signal = data[de_key].squeeze()

            # ---------------------------------------------
            # TRIM SIGNAL
            # ---------------------------------------------
            if cls == "normal":
                signal = signal[:240000]
            else:
                signal = signal[:120000]

            # ---------------------------------------------
            # WINDOW COUNT
            # ---------------------------------------------

            num_windows = (
                (len(signal) - window_size)
                // step_size
            ) + 1

            print(f"Generated {num_windows} windows")

            # ---------------------------------------------
            # GENERATE WINDOWS
            # ---------------------------------------------

            for i in range(num_windows):

                start = i * step_size
                end = start + window_size

                segment = signal[start:end]

                # -----------------------------------------
                # SPECTROGRAM
                # -----------------------------------------

                frequencies, times, Sxx = spectrogram(
                    segment,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap
                )

                # -----------------------------------------
                # LOG SCALE
                # -----------------------------------------

                Sxx = np.log(Sxx + 1e-10)

                # -----------------------------------------
                # IMAGE COUNT
                # -----------------------------------------

                image_counter[split][cls] += 1

                save_name = (
                    f"{cls}_"
                    f"{image_counter[split][cls]:05d}.png"
                )

                save_path = os.path.join(
                    output_folder,
                    save_name
                )

                # -----------------------------------------
                # SAVE IMAGE
                # -----------------------------------------

                plt.figure(figsize=(4,4))

                plt.pcolormesh(
                    times,
                    frequencies,
                    Sxx,
                    shading='gouraud'
                )

                plt.axis('off')

                plt.tight_layout()

                plt.savefig(
                    save_path,
                    bbox_inches='tight',
                    pad_inches=0
                )

                plt.close()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===================================")
print("DATASET GENERATION COMPLETE")
print("===================================")

for split in splits:

    print(f"\n{split.upper()} DATASET")

    total_images = 0

    for cls in classes:

        count = image_counter[split][cls]

        total_images += count

        print(f"{cls}: {count}")

    print(f"Total: {total_images}")
