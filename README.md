# MNIST Handwritten Digit Classification Using TensorFlow/Keras

## 1. Aim

To build a neural network using TensorFlow/Keras to classify handwritten digits from **0 to 9** using the MNIST dataset.

## 2. Objectives

1. Load MNIST dataset.
2. Preprocess images.
3. Build a neural network.
4. Train and test the model.
5. Check accuracy.
6. Predict five images.
7. Change hidden neurons and compare results.

## 3. Introduction

MNIST is a dataset of handwritten digit images from **0 to 9**. Each image is **28 × 28 pixels**. A neural network learns patterns from these images and predicts the correct digit.

## 4. Dataset

| Property        | Value     |
| --------------- | --------- |
| Training images | 60,000    |
| Testing images  | 10,000    |
| Classes         | 10        |
| Image size      | 28 × 28   |
| Type            | Grayscale |
| Pixel range     | 0–255     |

## 5. Theory

**Neural Network:** Learns patterns using connected neurons.

**ReLU:**

$$
ReLU(x)=max(0,x)
$$

**Softmax:** Gives probability for each digit.

**Normalization:**

$$
X_{new}=X/255
$$

**Loss:** Sparse Categorical Cross-Entropy.

**Optimizer:** Adam.

## 6. Model Architecture

```text
28 × 28 Image
     ↓
Flatten (784)
     ↓
Dense (128)
     ↓
ReLU
     ↓
Dense (10)
     ↓
Softmax
     ↓
Digit 0–9
```

## 7. Algorithm

1. Import libraries.
2. Load MNIST dataset.
3. Display sample images.
4. Normalize pixel values.
5. Create neural network.
6. Compile the model.
7. Train for 10 epochs.
8. Test the model.
9. Plot accuracy and loss.
10. Predict five images.
11. Change neurons from 128 to 256.
12. Compare results.

# 8. Implementation

### Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
```

### Load Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
```

**Output:**

```text
(60000, 28, 28)
(10000, 28, 28)
```

### Display Images

```python
plt.figure(figsize=(10,4))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.show()
```

### Normalize Data

```python
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
```

### Build Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

### Compile Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Train Model

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1
)
```

### Evaluate Model

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

**Test Accuracy:** ______

**Test Loss:** ______

## 9. Accuracy Graph

```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.show()
```

## 10. Loss Graph

```python
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])
plt.show()
```

### Observation

Accuracy increases and loss generally decreases during training.

## 11. Predict Five Images

```python
np.random.seed(42)

indices = np.random.choice(len(x_test), 5, replace=False)

predictions = model.predict(x_test[indices], verbose=0)
predicted = np.argmax(predictions, axis=1)

for i in range(5):
    print("Actual:", y_test[indices[i]],
          "Predicted:", predicted[i])
```

### Prediction Table

| Image | Actual | Predicted | Result |
| ----- | -----: | --------: | ------ |
| 1     |    ___ |       ___ | ___    |
| 2     |    ___ |       ___ | ___    |
| 3     |    ___ |       ___ | ___    |
| 4     |    ___ |       ___ | ___    |
| 5     |    ___ |       ___ | ___    |

# 12. Experiment: 256 Neurons

Change the hidden layer from **128 to 256 neurons**.

```python
experiment_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

experiment_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

experiment_model.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.1
)

loss, accuracy = experiment_model.evaluate(x_test, y_test)

print("Experimental Accuracy:", accuracy)
```

## 13. Comparison

| Model      | Neurons | Test Accuracy |
| ---------- | ------: | ------------: |
| Original   |     128 |        ______ |
| Experiment |     256 |        ______ |

### Observation

256 neurons provide more capacity to learn patterns. The actual accuracy should be compared with the results obtained.

# 14. Results

* Training images: **60,000**
* Testing images: **10,000**
* Image size: **28 × 28**
* Hidden neurons: **128**
* Output neurons: **10**
* Epochs: **10**
* Optimizer: **Adam**
* Test Accuracy: **______**
* Experimental neurons: **256**
* Experimental Accuracy: **______**

# 15. Advantages

1. Easy to implement.
2. Good accuracy.
3. Simple neural network.
4. Fast training.
5. Easy to understand.

# 16. Limitations

1. Uses a basic neural network.
2. Does not use CNN.
3. Spatial information is reduced by Flatten.
4. MNIST is simpler than real-world handwriting.

# 17. Future Improvements

* Use CNN.
* Add Dropout.
* Use data augmentation.
* Try different hidden layers.
* Tune hyperparameters.
* Test real handwritten images.

# 18. Conclusion

A neural network was successfully built using TensorFlow/Keras to classify MNIST handwritten digits. The model was trained, tested, visualized, and used to predict five images. Changing the hidden neurons from **128 to 256** demonstrated how model architecture can affect performance.

# 19. Technologies Used

**Python · TensorFlow · Keras · NumPy · Matplotlib · Google Colab/Jupyter Notebook**

# 20. Repository Structure

```text
MNIST-Digit-Classification/
├── MNIST_Digit_Classification.ipynb
├── MNIST_Report.md
├── README.md
└── requirements.txt
```

# 21. References

1. TensorFlow Documentation
2. Keras Documentation
3. MNIST Dataset
4. NumPy Documentation
5. Matplotlib Documentation
