# Multilayer Perceptron for Breast Cancer Classification

A neural network implementation from scratch using Python and NumPy for binary classification of breast cancer data from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 🎯 Project Overview

This project implements a multilayer perceptron (MLP) neural network to classify breast cancer tumors as malignant or benign. The implementation includes:

- **Custom neural network** built from scratch (no TensorFlow/PyTorch)
- **Backpropagation algorithm** with gradient descent optimization
- **Multiple hidden layers** with configurable architecture
- **Data preprocessing** with standardization/normalization
- **Model persistence** for saving and loading trained networks

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
├── main.py               # Simple training script (all-in-one)
├── train.py              # Advanced training with model saving
├── predict.py            # Model loading and prediction
├── split.py              # Data splitting utility
├── README.md             # This file
└── en.subject.pdf        # Project requirements
```

## 🚀 Quick Start

### 1. Train with Simple Script
```bash
python3 main.py
```

### 2. Advanced Training Pipeline

**Step 1: Split the data**
```bash
python3 split.py --input data.csv --train train.csv --valid valid.csv --val_ratio 0.2 --seed 42
```

**Step 2: Train the model**
```bash
python3 train.py --train train.csv --valid valid.csv \
                 --layer 64 32 16 \
                 --epochs 50 \
                 --batch_size 8 \
                 --learning_rate 0.02
```

**Step 3: Make predictions**
```bash
python3 predict.py --model saved_model.npy --test valid.csv
```

## ⚙️ Configuration Options

### Training Parameters
- `--layer`: Hidden layer sizes (e.g., `64 32 16`)
- `--epochs`: Number of training epochs (default: 70)
- `--batch_size`: Mini-batch size (default: 8)
- `--learning_rate`: Learning rate (default: 0.0314)

### Data Split Options
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--test_ratio`: Test set ratio (default: 0.0)
- `--seed`: Random seed for reproducibility (default: 42)

## 📈 Expected Results

With proper hyperparameters, the model typically achieves:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~97-98%
- **Training Loss**: <0.1 (cross-entropy)

Example training output:
```
epoch  1/20 - loss: 0.6234 - val_loss: 0.5891
epoch  2/20 - loss: 0.4123 - val_loss: 0.3987
...
epoch 20/20 - loss: 0.0821 - val_loss: 0.0934

Validation accuracy: 98.246% on 114 samples
```

## 🔧 Implementation Details

### Neural Network Features
- **Weight Initialization**: He/Glorot initialization
- **Backpropagation**: Full gradient computation
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Regularization**: Batch normalization via standardization

### Data Preprocessing
- **Normalization**: Z-score standardization (μ=0, σ=1)
- **Label Encoding**: M→1 (malignant), B→0 (benign)
- **One-hot Encoding**: Binary classification output

### Model Persistence
- **Weights & Biases**: Saved as NumPy arrays (.npy)
- **Scaler Parameters**: Mean and std saved (.npz)
- **Learning Curves**: Training plots (.png)

## 🛠️ Requirements

- Python 3.7+
- NumPy
- Matplotlib (for training plots)
- CSV module (built-in)

## 📝 Usage Examples

### Training with Custom Architecture
```bash
# Train a deeper network
python3 train.py --train train.csv --valid valid.csv \
                 --layer 128 64 32 16 8 \
                 --epochs 100 \
                 --learning_rate 0.01
```

### Batch Prediction
```bash
# Predict on multiple samples
python3 predict.py --model saved_model.npy --test test.csv --output predictions.csv
```

## 🎓 Educational Value

This project demonstrates:
- **Neural network fundamentals** from first principles
- **Backpropagation algorithm** implementation
- **Gradient descent optimization**
- **Binary classification techniques**
- **Data preprocessing pipelines**
- **Model evaluation metrics**

## 📚 Mathematical Background

The implementation includes:
- **Forward propagation**: `z = Wx + b`, `a = σ(z)`
- **Backpropagation**: `∂L/∂W = δ⊗aᵀ`, `∂L/∂b = δ`
- **Cross-entropy loss**: `L = -Σ(y*log(ŷ))`
- **Softmax output**: `σ(z)ᵢ = eᶻⁱ/Σeᶻʲ`
