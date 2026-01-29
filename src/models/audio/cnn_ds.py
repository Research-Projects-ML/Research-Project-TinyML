from tensorflow.keras import layers, models

def build():
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.DepthwiseConv2D((3, 3), activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.DepthwiseConv2D((3, 3), activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])