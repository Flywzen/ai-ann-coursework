# Artificial Intelligence Coursework

## Overview

This repository contains a coursework implementation of an Artificial Neural Network (ANN) developed for an Artificial Intelligence class.

The purpose of this project is to understand the core mechanics of feedforward neural networks, including forward propagation, backpropagation, loss computation, and parameter updates through gradient-based optimization.

---

## Problem Description

Brief description of the supervised learning task:

* Task type: 
* Dataset: 
* Number of features:
* Target variable:

---

## Model Architecture

The implemented model follows a standard feedforward neural network structure:

* Input layer: n features
* Hidden layer(s): configurable
* Activation function: (ReLU / Sigmoid / Tanh)
* Output layer: (Softmax / Linear)

Mathematical formulation:

Forward propagation:
[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
]
[
a^{(l)} = f(z^{(l)})
]

Loss function:

* Mean Squared Error (for regression) 
* Cross-Entropy Loss (for classification)

---

## Training Configuration

* Optimizer: (SGD / Adam / manual gradient descent)
* Learning rate:
* Epochs:
* Batch size:
* Weight initialization method:

---

## Results

Summarize the training outcome:

* Final training loss:
* Validation accuracy:
* Observations regarding convergence behavior
* Notes on overfitting or underfitting

---

## Repository Structure

```
ai-ann-coursework/
│
├── notebooks/      # Jupyter notebooks
├── src/            # Core implementation 
├── data/           # Dataset 
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository:

   ```
   git clone https://github.com/Flywzen/ai-ann-coursework.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```
   jupyter notebook
   ```

---

## Dependencies

* Python 3.x
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Notes

This project is intended for academic learning purposes and focuses on understanding fundamental neural network behavior rather than production-level optimization.
