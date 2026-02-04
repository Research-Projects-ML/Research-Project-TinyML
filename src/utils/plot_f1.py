import os
import matplotlib.pyplot as plt

def plot_metrics(history, filename):
    # Check keys
    available_keys = list(history.history.keys())
    if "f1" not in available_keys or "accuracy" not in available_keys:
        raise ValueError(
            f"Required metrics not found. Available keys: {available_keys}"
        )

    save_path = os.path.join("plots", filename)

    plt.figure(figsize=(8, 5))

    # Plot Accuracy
    plt.plot(history.history["accuracy"], label="Train Accuracy", linestyle='-')
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle='-')

    # # Plot F1
    # plt.plot(history.history["f1"], label="Train F1", linestyle='--')
    # plt.plot(history.history["val_f1"], label="Validation F1", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Metrics During Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
