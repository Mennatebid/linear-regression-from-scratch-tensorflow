# Linear Regression from Scratch with TensorFlow

A modular implementation of linear regression from scratch using TensorFlow, exploring the mathematical foundations of regression, automatic differentiation, stochastic gradient descent, and model evaluation.

The project also includes a high-level Keras implementation to demonstrate how deep learning frameworks abstract the underlying mathematical and computational operations.

## Project Objectives

- Implement linear regression without relying on high-level Keras training utilities.
- Generate synthetic datasets with known ground-truth parameters.
- Implement custom minibatch data loaders.
- Perform forward propagation using tensor operations.
- Implement the squared error loss function and stochastic gradient descent.
- Compute gradients using TensorFlow's automatic differentiation.
- Build custom training and validation loops.
- Visualize learning curves and evaluate parameter recovery.
- Compare the from-scratch implementation with TensorFlow/Keras.

## Mathematical Foundations

### Linear Regression Model

The model assumes a linear relationship between input features and target values:

$$\hat{y} = Xw + b$$

Where:

- $X$: input feature matrix.
- $w$: trainable weight vector.
- $b$: trainable bias.
- $\hat{y}$: predicted values.

### Synthetic Data Generation

The synthetic dataset is generated according to:

$$y = Xw + b + \epsilon$$

Where the noise follows a Gaussian distribution:

$$\epsilon \sim \mathcal{N}(0, 0.01^2)$$

The ground-truth parameters are:

$$w = [2.0, -3.4]^T, \qquad b = 4.2$$

### Loss Function

The model minimizes the half mean squared error:

$$\mathcal{L}(w,b)=\frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$$

### Stochastic Gradient Descent

The model parameters are updated using minibatch stochastic gradient descent:

$$w \leftarrow w-\eta\nabla_w\mathcal{L}$$

$$b \leftarrow b-\eta\nabla_b\mathcal{L}$$

Where $\eta$ is the learning rate.

Gradients are computed using TensorFlow's `GradientTape`, while parameter updates are implemented manually using `assign_sub()`.

## Dataset Configuration

| Parameter | Value |
|---|---:|
| Total observations | 2,000 |
| Training observations | 1,000 |
| Validation observations | 1,000 |
| Input features | 2 |
| Output features | 1 |
| Noise standard deviation | 0.01 |

## Model Configuration

| Hyperparameter | Value |
|---|---:|
| Batch size | 32 |
| Learning rate | 0.03 |
| Epochs | 10 |
| Weight initialization | Gaussian distribution |
| Weight initialization standard deviation | 0.01 |
| Bias initialization | Zero |
| Optimizer | Minibatch SGD |
| Trainable parameters | 3 |

## Project Structure

```text
linear-regression-from-scratch-tensorflow/
│
├── synthetic_data.py    # Synthetic dataset generation
├── model.py             # Parameter initialization and forward propagation
├── loss.py              # Custom squared error loss function
├── optimizer.py         # Custom SGD implementation
├── train.py             # Training, validation and visualization
├── keras_model.py       # High-level Keras implementation
├── .gitignore
└── README.md
```

## Training Pipeline

1. Generate synthetic observations and ground-truth labels.
2. Split the dataset into training and validation sets.
3. Shuffle and partition training observations into minibatches.
4. Initialize trainable model parameters.
5. Perform forward propagation.
6. Calculate the prediction loss.
7. Compute gradients using automatic differentiation.
8. Update parameters using stochastic gradient descent.
9. Evaluate the model on validation data.
10. Record and visualize training and validation losses.

Training and validation losses are aggregated using sample-weighted averages to account for minibatches of different sizes.

## From Scratch vs. Keras

| Component | From Scratch | Keras |
|---|---|---|
| Model architecture | Tensor operations | Sequential and Dense |
| Parameters | `tf.Variable` | Layer-managed variables |
| Forward propagation | `tf.matmul()` | Model call |
| Loss | Custom implementation | Custom or built-in loss |
| Differentiation | `tf.GradientTape` | Automatic |
| Optimization | Custom SGD | `tf.keras.optimizers.SGD` |
| Training | Custom training loop | `model.fit()` |
| Validation | Custom evaluation function | Built-in validation |
| Loss tracking | Python lists | Keras History object |

## Installation

```bash
pip install tensorflow 
```

## Usage

### From-Scratch Implementation

```bash
python train.py
```

### Keras Implementation

```bash
python keras_model.py
```

Both implementations train a linear regression model on synthetic observations and visualize training and validation losses.

## Model Evaluation

Model performance is evaluated using:

- Training loss.
- Validation loss.
- Recovery of ground-truth weights and bias.

The expected outcome is convergence toward the ground-truth parameters, with decreasing training and validation losses.

## Technologies

- Python
- TensorFlow
- Keras

## Key Concepts

Linear Regression · Supervised Learning · Automatic Differentiation · Stochastic Gradient Descent · Tensor Operations · Minibatch Training · Parameter Estimation · Model Validation · Numerical Optimization

## References

This project is inspired by Chapter 3 of [Dive into Deep Learning](https://d2l.ai/), particularly:

- [3.3. Synthetic Regression Data](https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html)
- [3.4. Linear Regression Implementation from Scratch](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html)
- [3.5. Concise Implementation of Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression-concise.html)

The implementation is developed independently of the D2L package using TensorFlow.
