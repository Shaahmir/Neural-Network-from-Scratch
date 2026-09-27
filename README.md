# Neural Network From Scratch

A fully-connected neural network built **entirely from scratch using NumPy** - no PyTorch, no TensorFlow, no autograd. Every component - forward pass, backpropagation, and gradient descent - is implemented manually to build a from-the-ground-up understanding of how neural networks actually learn.

Trained on the [MNIST digit classification dataset](https://www.kaggle.com/competitions/digit-recognizer) (Kaggle's Digit Recognizer competition), reaching **~97.8% validation accuracy**.

## Architecture

- **Input layer:** 784 units (28×28 flattened grayscale pixels)
- **Hidden layer:** 128 units, ReLU activation
- **Output layer:** 10 units, Softmax activation (digit classes 0–9)
- **Loss:** Cross-entropy (via one-hot encoded labels)
- **Optimizer:** Full-batch gradient descent
- **Weight init:** He initialization (`sqrt(2 / fan_in)`), zero-initialized biases

## Results

| Metric | Value |
|---|---|
| Train Accuracy | 98.22% |
| Validation Accuracy | 97.76% |

## What's implemented from scratch

- `forward()` - manual forward propagation (matrix multiplications, ReLU, Softmax)
- `backpropagation()` - manual gradient computation via the chain rule
- `gradient_descent()` - manual parameter update rule
- `init_parameters()` - He-initialized weights, zero-initialized biases
- Training loop with live train/validation accuracy tracking (`tqdm`)

## Hyperparameter tuning process

Several hyperparameter configurations were tested before settling on the final setup, tracking both validation accuracy and the train/validation gap to catch under and overfitting along the way:

| lr | epochs | bias init | train acc | valid acc | gap |
|---|---|---|---|---|---|
| 0.1 | 1000 | random | 94.19% | 93.86% | 0.33% |
| 0.3 | 3000 | random | 99.21% | 96.43% | 2.78% |
| 0.5 | 1000 | random | 98.23% | 96.86% | 1.37% |
| 0.6 | 1000 | random | 98.81% | 96.76% | 2.05% |
| **0.5** | **1000** | **zeros** | **98.22%** | **97.76%** | **0.46%** |

Key fixes along the way:
- Corrected a bias-gradient bug (`db1`/`db2` were collapsing across the wrong axis, forcing all biases in a layer to update identically)
- Switched from naive uniform weight init to He initialization, matched to each layer's correct fan-in
- Diagnosed and resolved an underfitting issue (too few gradient updates under full-batch GD) before addressing overfitting
- Zero-initialized biases instead of randomizing them, tightening the train/validation generalization gap

## Usage

```bash
git clone https://github.com/Shaahmir/Neural-Network-from-Scratch.git
cd Neural-Network-from-Scratch
pip install numpy pandas matplotlib tqdm kagglehub
```

Run the notebook (`Neural_Network_from_Scratch.ipynb`) top to bottom. It will:
1. Download the MNIST digit-recognizer dataset via `kagglehub`
2. Train the network
3. Report validation accuracy
4. Visualize sample predictions
5. Generate a Kaggle-ready `submission.csv`

## Why from scratch?

Frameworks like PyTorch and TensorFlow abstract away the math that actually makes neural networks work. This project strips that away - every gradient, every matrix multiplication, every parameter update is derived and coded by hand. It's a companion, lower-level project to [PlutoAI](https://github.com/Shaahmir/PlutoAI), a 125M-parameter transformer LLM also built from scratch.

## License

This project is released under the MIT License. See the `LICENSE` file for additional information.
