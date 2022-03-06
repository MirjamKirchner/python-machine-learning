import pandas as pd
import numpy as np
import os
from config import PATH_TO_DATA
FILE_RAW_DATA = "raw_iris.data"
COLNAMES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "target"]


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_TO_DATA, FILE_RAW_DATA), header=None, names=COLNAMES)


def preprocess_target(training_data: pd.DataFrame):
    y = training_data["target"].values
    return np.where(y == "Iris-setosa", -1, 1)


def extract_features(training_data: pd.DataFrame, features: list = None):
    if features is None:
        return training_data.to_numpy()
    else:
        return training_data[features].to_numpy()


if __name__ == "__main__":
    training_data = load_training_data()
    examples = extract_features(training_data, ["sepal_length_cm", "petal_length_cm"])
    target = preprocess_target(training_data)
    np.save(os.path.join(PATH_TO_DATA, 'training_examples_iris.npy'), examples)
    np.save(os.path.join(PATH_TO_DATA, 'training_target_iris.npy'), target)