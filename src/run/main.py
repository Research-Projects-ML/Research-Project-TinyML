import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Load model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load data
x_test = np.load("data/x_test.npy").astype(np.float32)  # cast to float32
y_test = np.load("data/y_test.npy")

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("Model input details:", input_details)

# Run model
outputs = []
for x in x_test:
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x, axis=0))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    outputs.append(out)

outputs = np.array(outputs)           # shape (2947, 1, 10)
outputs = np.squeeze(outputs, axis=1) # shape (2947, 10)

# Get predicted class labels
y_pred = np.argmax(outputs, axis=1)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
loss = log_loss(y_test, outputs)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Log Loss: {loss:.4f}")
