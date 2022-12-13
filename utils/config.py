import os

# Project location
# PROJECT_PATH = "./"
# Input dataset directory
RAW_INPUT_PATH = "./dataset/raw"
# Result dataset directory
SPLIT_INPUT_PATH = "./dataset/split"
# Output directory
OUTPUT_PATH = "./output"

# Training testing, validation
TRAIN_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "training"])
VAL_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "validation"])
TEST_PATH = os.path.sep.join([SPLIT_INPUT_PATH, "testing"])

# Data splitting
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Parameters
CLASSES = ["benign","malignant"]
BATCH_SIZE = 24
INIT_LR = 1e-3
EPOCHS = 220

# Path to serialized model after training
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "CarcinomaPrediction.model"])

# Setting the path for the output training history plots for the custom model
PLOT_PATH_CM = os.path.sep.join([OUTPUT_PATH, "TrainingHistoryCM.png"])

# Setting the path for the output training history plots for the ResNet model
PLOT_PATH_RN = os.path.sep.join([OUTPUT_PATH, "TrainingHistoryRN.png"])
