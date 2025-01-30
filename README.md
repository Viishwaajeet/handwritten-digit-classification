# 🖋️ Handwritten Digit Classification with TensorFlow and Keras

This project demonstrates a handwritten digit classification system using Deep Learning techniques, particularly a Neural Network model trained on the MNIST dataset. The model classifies digits (0-9) from images, achieving high accuracy with TensorFlow and Keras.

## 🚀 Project Overview

This project utilizes the MNIST dataset (Modified National Institute of Standards and Technology), which contains 28x28 pixel grayscale images of handwritten digits. The goal is to build a deep learning model capable of recognizing these digits with high accuracy. The project uses Keras for building the model and TensorFlow as the backend for training and inference.

The model achieved an impressive 98% test accuracy after training on 60,000 images, making it highly effective for digit recognition tasks.

## 📊 Dataset

The model is trained on the MNIST dataset, which consists of:

- 60,000 training images of handwritten digits.
- 10,000 test images for evaluating the model’s performance.
- Each image is a 28x28 grayscale image, where the digits are centered in the image.

You can download the dataset directly from Keras, as it is included in the library.

## 🧠 Model Architecture

The model consists of the following layers:

- **Input Layer:** Flattens the 28x28 pixel image into a 784-dimensional vector.
- **Hidden Layer:** A fully connected Dense layer with 512 neurons and ReLU activation.
- **Output Layer:** A Dense layer with 10 neurons (one for each digit), using the softmax activation function to output probabilities for each class.

### Model Summary:
```python
model = models.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

## 🏋️ Training Process

The model was trained using the categorical cross-entropy loss function and the RMSprop optimizer.

### Training Configuration:
- **Epochs:** 5
- **Batch size:** 128
- **Optimizer:** RMSprop
- **Loss function:** Categorical Crossentropy

### Performance:
- **Training Accuracy:** ~98.90%
- **Test Accuracy:** ~98.04%

## 📈 Results

The model achieved 98.04% accuracy on the test dataset, making it highly effective for recognizing handwritten digits.
