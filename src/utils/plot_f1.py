import os

import matplotlib.pyplot as plt


def plot_f1(history, filename):
    if "f1" not in history.history:
        raise ValueError(
            f"F1 metric not found. Available keys: {list(history.history.keys())}"
        )

    save_path = os.path.join("plots", filename)

    plt.figure()
    plt.plot(history.history["f1"], label="Train F1")
    plt.plot(history.history["val_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score During Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
