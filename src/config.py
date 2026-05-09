# ==========================================
# PATHS
# ==========================================

TRAIN_DIR = "../dataset_gray/train"
TEST_DIR = "../dataset_gray/test"

MODEL_SAVE_PATH = "../models/bearing_cnn.pth"

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMAGE_SIZE = 128

# ==========================================
# TRAINING SETTINGS
# ==========================================

BATCH_SIZE = 16

LEARNING_RATE = 0.001

NUM_EPOCHS = 10

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "ball",
    "inner",
    "normal",
    "outer"
]