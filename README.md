# Own MNIST Neural Network

This project implements a simple 3-layer Neural Network from scratch in Python to classify handwritten digits from the MNIST dataset.

## Project Overview

*   **Goal:** Classify MNIST digits (0-9) with high accuracy.
*   **Architecture:** 3-layer Neural Network (Input -> Hidden -> Output).
*   **Implementation:** Pure Python + NumPy (No deep learning frameworks like TensorFlow or PyTorch used for the core logic).
*   **Current Accuracy:** **93.85%** on the test dataset.

## Files

*   `train.py`: The main script containing the `NeuralNetworks` class and the training/testing loop.
*   `data_preprocessing.py`: Helper functions for loading and processing the CSV data.
*   `test_all_output.json`: Contains the detailed results of the test run, including predictions for every image and the final accuracy.
*   `mnist_train.csv` / `mnist_test.csv`: The dataset files (expected in the root directory).

## How it Works

The network uses a standard Feedforward architecture with Backpropagation for training:

1.  **Input Layer:** 784 nodes (representing 28x28 pixel images).
2.  **Hidden Layer:** 100 nodes.
3.  **Output Layer:** 10 nodes (representing digits 0-9).
4.  **Activation Function:** Sigmoid function.
5.  **Optimization:** Gradient Descent.

## Usage

Run the training script:

```bash
python train.py
```

This will:
1.  Initialize the network.
2.  Train on `mnist_train.csv` for 2 epochs.
3.  Test on `mnist_test.csv`.
4.  Save the results to `test_all_output.json`.

## Results

The model achieves **93.85%** accuracy. Detailed predictions for each test sample can be found in `test_all_output.json`.
