# Multilayer Perceptron for Breast Cancer Classification

A neural network implementation from scratch using Python and NumPy for binary classification of breast cancer data from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 🎯 Project Overview

This project implements a multilayer perceptron (MLP) neural network to classify breast cancer tumors as malignant or benign. The implementation includes:

- **Custom neural network** built from scratch (no TensorFlow/PyTorch)
- **Backpropagation algorithm** with gradient descent optimization
- **Multiple hidden layers** with configurable architecture
- **Data preprocessing** with standardization/normalization
- **Model persistence** for saving and loading trained networks
- **Automatic recovery** for poor hyperparameter configurations

## 📊 Dataset

- **Source**: Wisconsin Diagnostic Breast Cancer (WDBC) dataset
- **Features**: 30 real-valued features computed from digitized images
- **Classes**: 2 (Malignant=M, Benign=B)
- **Samples**: ~569 instances
- **Format**: CSV with ID, diagnosis, and 30 feature columns

## 🏗️ Network Architecture

- **Input Layer**: 30 neurons (one per feature)
- **Hidden Layers**: Configurable (default: [64, 32, 16] neurons)
- **Output Layer**: 2 neurons with softmax activation
- **Activation Functions**: 
  - Hidden layers: Sigmoid
  - Output layer: Softmax
- **Loss Function**: Binary cross-entropy

## 📁 Project Structure

```
multilayer_perceptron/
├── data.csv              # Wisconsin breast cancer dataset
├── train.py              # Training script with model saving
├── predict.py            # Model loading and prediction
├── split.py              # Data splitting utility
├── rules.py              # Core mathematical functions
├── main.py               # Simple all-in-one training script
├── README.md             # This file
└── saved_model.npy       # Trained model (generated after training)
```

## 🚀 Quick Start

### 1. Split the Data
```bash
python3 split.py --input data.csv --train train.csv --valid valid.csv --seed 42
```

### 2. Train the Model
```bash
python3 train.py --train train.csv --valid valid.csv
```

### 3. Make Predictions
```bash
python3 predict.py --model saved_model.npy --test valid.csv
```

## ⚙️ Configuration Options

### Training Parameters
- `--layer`: Hidden layer sizes (e.g., `64 32 16`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Mini-batch size (default: 8)
- `--learning_rate`: Learning rate (default: 0.0314)
- `--seed`: Random seed for reproducibility (default: 42)

### Data Split Options
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--test_ratio`: Test set ratio (default: 0.0)
- `--seed`: Random seed for reproducible splits (default: 42)

## 📈 Expected Results

With proper hyperparameters, the model typically achieves:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~97-98%
- **Binary Cross-Entropy**: < 0.08

Example training output:
```
x_train shape : (456, 30)
x_valid shape : (113, 30)
epoch   1/100 - loss: 0.5473 - val_loss: 0.5482
epoch   2/100 - loss: 0.4800 - val_loss: 0.4765
...
epoch 100/100 - loss: 0.0421 - val_loss: 0.0456

> saved model and scaler in saved_model.npy
> learning_curves.png saved
```

## 🔧 Advanced Usage

### Custom Architecture Training
```bash
# Train with deeper network
python3 train.py --train train.csv --valid valid.csv \
                 --layer 128 64 32 16 \
                 --epochs 150 \
                 --learning_rate 0.02

# Train with wider network
python3 train.py --train train.csv --valid valid.csv \
                 --layer 256 128 \
                 --epochs 80 \
                 --learning_rate 0.025
```

### Batch Prediction with Output File
```bash
python3 predict.py --model saved_model.npy --test valid.csv --output predictions.csv
```

### Different Data Splits
```bash
# 70% train, 20% valid, 10% test
python3 split.py --input data.csv \
                 --train train.csv --valid valid.csv --test test.csv \
                 --val_ratio 0.2 --test_ratio 0.1 \
                 --seed 42
```

## 🧠 Implementation Details

### Neural Network Features
- **Weight Initialization**: He uniform initialization for stable training
- **Backpropagation**: Full gradient computation with chain rule
- **Optimization**: Stochastic Gradient Descent (SGD) with mini-batches
- **Regularization**: Input standardization (z-score normalization)

### Data Preprocessing
- **Normalization**: Z-score standardization (μ=0, σ=1)
- **Label Encoding**: M→1 (malignant), B→0 (benign)
- **One-hot Encoding**: Binary classification output
- **Format Handling**: Flexible CSV parsing (with/without ID column)

### Model Persistence
- **Single File**: Everything saved in `saved_model.npy`
- **Contains**: Weights, biases, architecture, training history, normalization parameters
- **Format**: NumPy dictionary for fast loading

### Automatic Recovery System
- **Detection**: Monitors validation loss > 0.08
- **Recovery**: Automatically tries proven architectures
- **Fallback**: Uses reliable hyperparameters if initial training fails

## 📚 Mathematical Background

### Forward Propagation
```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = σ(z^(l))  [sigmoid for hidden layers]
a^(L) = softmax(z^(L))  [softmax for output layer]
```

### Backpropagation
```
∂L/∂W^(l) = δ^(l) ⊗ (a^(l-1))^T
∂L/∂b^(l) = δ^(l)
δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^(l))
```

### Loss Function
```
Binary Cross-Entropy: L = -∑[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

## 🛠️ Requirements

- Python 3.7+
- NumPy
- Matplotlib (for learning curves)
- CSV module (built-in)

## 🎓 Educational Value

This project demonstrates:
- **Neural network fundamentals** from first principles
- **Backpropagation algorithm** step-by-step implementation
- **Gradient descent optimization** techniques
- **Binary classification** methodologies
- **Data preprocessing** best practices
- **Model evaluation** and persistence

## 🔍 Troubleshooting

### Common Issues

**Training Loss Too High (> 0.08):**
```bash
# Try proven architecture
python3 train.py --train train.csv --valid valid.csv --layer 64 32 16
```

**Training Takes Too Long:**
```bash
# Use smaller network or higher learning rate
python3 train.py --train train.csv --valid valid.csv --layer 32 16 --learning_rate 0.05
```

**Overfitting (train_acc >> val_acc):**
```bash
# Use smaller network or more data
python3 train.py --train train.csv --valid valid.csv --layer 32 16
```

### Recommended Configurations

**Most Reliable:**
```bash
python3 train.py --train train.csv --valid valid.csv --layer 64 32 16 --learning_rate 0.02
```

**Fastest Training:**
```bash
python3 train.py --train train.csv --valid valid.csv --layer 48 24 --learning_rate 0.03
```

**Highest Accuracy:**
```bash
python3 train.py --train train.csv --valid valid.csv --layer 128 64 32 --epochs 150 --learning_rate 0.015
```

## 📄 Files Description

- **`train.py`**: Main training script with full functionality
- **`predict.py`**: Model loading and prediction with detailed output
- **`split.py`**: Data splitting utility with flexible ratios
- **`rules.py`**: Mathematical functions (sigmoid, softmax, etc.)
- **`main.py`**: Simple all-in-one training (alternative to train.py)
- **`data.csv`**: Wisconsin Diagnostic Breast Cancer dataset


```bash
# Complete workflow
python3 split.py --input data.csv --train train.csv --valid valid.csv --seed 42
python3 train.py --train train.csv --valid valid.csv
python3 predict.py --model saved_model.npy --test valid.csv

# Custom training
python3 train.py --train train.csv --valid valid.csv \
                 --layer 64 32 16 \
                 --epochs 100 \
                 --batch_size 8 \
                 --learning_rate 0.02 \
                 --seed 42
```