import pandas as pd
import os

def load_data(root_dir):

    train_path = os.path.join(root_dir, "UNSW_NB15_training-set.parquet")
    test_path  = os.path.join(root_dir, "UNSW_NB15_testing-set.parquet")

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    # Targets
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # Drop non-feature columns
    drop_cols = ["label"]
    if "attack_cat" in train_df.columns:
        drop_cols.append("attack_cat")

    x_train = train_df.drop(columns=drop_cols)
    x_test  = test_df.drop(columns=drop_cols)

    # One-hot encode categorical columns
    x_train = pd.get_dummies(x_train)
    x_test  = pd.get_dummies(x_test)

    # Ensure same feature space
    x_train, x_test = x_train.align(
        x_test, join="left", axis=1, fill_value=0
    )

    return (
        x_train.values.astype("float32"),
        y_train.astype("int64"),
        x_test.values.astype("float32"),
        y_test.astype("int64"),
    )