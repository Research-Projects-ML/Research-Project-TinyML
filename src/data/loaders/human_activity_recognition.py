import pandas as pd
import os

def load_data(root_dir):

    def load_split(split):
        split_dir = os.path.join(root_dir, split)

        x = pd.read_csv(
            os.path.join(split_dir, f"X_{split}.txt"),
            sep=r"\s+",
            header=None
        ).values

        y = pd.read_csv(
            os.path.join(split_dir, f"y_{split}.txt"),
            header=None
        ).values.squeeze()

        return x, y

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")

    return x_train, y_train, x_test, y_test