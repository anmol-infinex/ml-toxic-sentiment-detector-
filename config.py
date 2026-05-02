from pathlib import Path

# Base directory = the folder this config.py lives in.
# Using __file__ means all paths are absolute and always correct,
# no matter which directory you run your scripts from.
BASE_DIR = Path(__file__).resolve().parent

# Paths
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = str(BASE_DIR / "train.csv")
TEST_FILE = str(BASE_DIR / "test.csv")
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"

# Data config
TEXT_COLUMN = "text"   # Column name containing your text data
LABEL_COLUMN = "label" # Column name containing labels ("bad" or "good")

# Model config
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
