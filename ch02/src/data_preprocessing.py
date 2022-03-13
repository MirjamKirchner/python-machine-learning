import pandas as pd
import numpy as np
import os
from config import PATH_TO_DATA
FILE_RAW_DATA = "raw_iris.data"
COLNAMES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "target"]


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_TO_DATA, FILE_RAW_DATA), header=None, names=COLNAMES)


def preprocess_target(training_data: pd.DataFrame, target_value="Iris-setosa"):
    y = training_data["target"].values
    return np.where(y == target_value, -1, 1)


def extract_features(training_data: pd.DataFrame, features: list = None):
    if features is None:
        return training_data.to_numpy()
    else:
        return training_data[features].to_numpy()


def preprocess_ova_targets(training_data: pd.DataFrame):
    unique_target_values = training_data["target"].unique()
    list_y = [preprocess_target(training_data, target_value=target_value) for target_value in unique_target_values]
    return np.array([list_y, unique_target_values], dtype=object)


if __name__ == "__main__":
    training_data = load_training_data()

    examples = extract_features(training_data, ["sepal_length_cm", "petal_length_cm"])
    bcp_target = preprocess_target(training_data)
    ova_targets = preprocess_ova_targets(training_data)

    np.save(os.path.join(PATH_TO_DATA, 'training_examples_iris.npy'), examples)
    np.save(os.path.join(PATH_TO_DATA, 'training_target_iris_setosa.npy'), bcp_target)
    np.save(os.path.join(PATH_TO_DATA, 'training_targets_ova.npy'), ova_targets)
