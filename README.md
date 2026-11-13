# 2026AJ05998_DNN_Assignment2
# Sequence Modeling with RNN, LSTM, and GRU Networks
# Student Information

    Name: SAKTHIVEL S.
    Course: Deep Neural Networks
    Assignment: Assignment 2
    Topic: Human Activity Recognition and Sequence Learning using RNN, LSTM, and GRU

Project Overview

This assignment investigates recurrent neural network architectures for sequential data modeling. The project consists of two major tasks:

    Human Activity Recognition (HAR)
        Classification of human activities using smartphone sensor data.
        Comparison of RNN, LSTM, and GRU architectures.

    Adding Problem
        Synthetic sequence learning benchmark.
        Evaluation of long-term dependency learning capabilities.
        Analysis of vanishing gradient effects.

Dataset
Human Activity Recognition (HAR)

Dataset Source:

    UCI Machine Learning Repository
    Human Activity Recognition Using Smartphones Dataset

Activities:
Class ID	Activity
1	Walking
2	Walking Upstairs
3	Walking Downstairs
4	Sitting
5	Standing
6	Laying

Dataset Shape:
Split	Samples	Sequence Length	Features
Training	7352	128	9
Testing	2947	128	9

Sensor Signals:

    Body Acceleration (X, Y, Z)
    Body Gyroscope (X, Y, Z)
    Total Acceleration (X, Y, Z)

Installation

Install required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision torchaudio

Project Structure

.
├── assignment2.ipynb
├── README.md
├── results.csv
├── class_balance.png
├── confusion_matrix.png
├── learning_curve_T200.png
├── mse_vs_T.png
├── gradient_norm.png
├── hidden_size_sweep.png
└── har/

Methodology
Data Preprocessing

    Download HAR dataset.
    Load inertial sensor signals.
    Normalize features using training statistics.
    Create custom PyTorch datasets and dataloaders.

Models Implemented
Simple RNN

Standard recurrent neural network:

h_t = tanh(Wx_t + Uh_(t-1))

Characteristics:

    Simple architecture
    Fast training
    Suffers from vanishing gradients

LSTM

Long Short-Term Memory network:

Components:

    Input Gate
    Forget Gate
    Output Gate
    Cell State

Advantages:

    Handles long-term dependencies
    Reduces vanishing gradient problem

GRU

Gated Recurrent Unit:

Components:

    Update Gate
    Reset Gate

Advantages:

    Fewer parameters than LSTM
    Faster training
    Competitive performance

Experiments
Experiment 1: HAR Classification

Models Compared:

    RNN
    LSTM
    GRU

Evaluation Metrics:

    Accuracy
    Macro F1-Score
    Confusion Matrix

Experiment 2: Adding Problem

Sequence Lengths:

T = [50, 100, 200, 400]

Objective: Predict the sum of two marked positions in a sequence.

Metric:

    Mean Squared Error (MSE)

Purpose:

    Evaluate memory retention
    Observe performance degradation with increasing sequence length

Visualizations
1. Class Distribution

File:

class_balance.png

Shows the distribution of activity classes in the HAR training set.
2. Confusion Matrix

File:

confusion_matrix.png

Displays classification performance for the best-performing model.
3. Learning Curve

File:

learning_curve_T200.png

Compares training losses of:

    RNN
    LSTM
    GRU

for sequence length T = 200.
4. MSE vs Sequence Length

File:

mse_vs_T.png

Demonstrates model robustness as sequence length increases.
5. Gradient Norm Analysis

File:

gradient_norm.png

Illustrates the vanishing gradient phenomenon.
6. Hidden Size Sweep

File:

hidden_size_sweep.png

Investigates the impact of hidden layer size:

[16, 32, 64, 128]

on model performance.
Results

Results are stored in:

results.csv

Columns:
Field	Description
model	Model type
accuracy	Classification accuracy
macro_f1	Macro averaged F1 score
Key Findings

    LSTM generally achieves the highest classification performance.
    GRU offers a strong trade-off between accuracy and computational efficiency.
    Vanilla RNN performance degrades for long sequences due to vanishing gradients.
    LSTM and GRU maintain better gradient flow and memory retention.
    Increasing hidden size improves performance up to a threshold after which gains become marginal.

Technologies Used

    Python
    PyTorch
    NumPy
    Pandas
    Matplotlib
    Seaborn
    Scikit-Learn

References

    UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.

    Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder.

    PyTorch Documentation: https://pytorch.org/

Author

SAKTHIVEL S.

Deep Neural Networks Assignment 2



## Links

- **GitHub Repository:** (https://github.com/ssakthiece-wq/SAKTHIVELS)
