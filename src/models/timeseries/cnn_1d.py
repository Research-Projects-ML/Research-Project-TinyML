from tensorflow.keras import layers, models

def build():
    return models.Sequential([
        layers.Input(shape=(784, 1)),

        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])