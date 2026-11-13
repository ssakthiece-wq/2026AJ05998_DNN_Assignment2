# 2026AJ05998_DNN_Assignment2
# Problem 1: Sequence Modeling with RNN, LSTM, and GRU Networks
# Problem 2: The Transformer 
# Student Information

    Name: SAKTHIVEL S.
    Course: Deep Neural Networks
    Assignment: Assignment 2
    Topic: Human Activity Recognition and Sequence Learning using RNN, LSTM, and GRU

# Project Overview

This assignment investigates recurrent neural network architectures for sequential data modeling. The project consists of two major tasks:

    **Human Activity Recognition (HAR)**
        Classification of human activities using smartphone sensor data.
        Comparison of RNN, LSTM, and GRU architectures.

    **Adding Problem 1**
        Synthetic sequence learning benchmark.
        Evaluation of long-term dependency learning capabilities.
        Analysis of vanishing gradient effects.

# Dataset
Human Activity Recognition (HAR)

**1.	Data pipelines :**

1) Input Dimensions: 128 sequence steps across 9 inertial signal channels (𝑁𝑡𝑟𝑎𝑖𝑛=7,352, 𝑁𝑡𝑒𝑠𝑡=2,947).
2) Validation Split: 4 subjects were held out from the 21 training subjects using np.random.default_rng(seed).choice(train_subjects, 4, replace=False) for each of the 3 seeds (42, 123, 999).
3) Preprocessing: Per-channel mean and standard deviation were calculated strictly on the training partition of each split to prevent data leakage.

# Dataset Shape:
Split	Samples	Sequence Length	Features

Training	7352	128	9
Testing	2947	128	9
**Dataset URL:** url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
Sensor Signals:

    Body Acceleration (X, Y, Z)
    Body Gyroscope (X, Y, Z)
    Total Acceleration (X, Y, Z)

# 2.	Cells from scratch 

**Install required libraries:**

pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision torchaudio

Project Structure
.
├── 2026AJ05998_DNN_Assignment
├── README.md
├── requirements.txt
├── report.pdf
├── results.csv
├── problem1.ipynb
├── figures/
├── problem2.ipynb
├── figures/

# 3.	HAR classification
**Data Preprocessing**

    Download HAR dataset.
    Load inertial sensor signals.
    Normalize features using training statistics.
    Create custom PyTorch datasets and dataloaders.

Models Implemented
**Simple RNN**

Standard recurrent neural network:

h_t = tanh(Wx_t + Uh_(t-1))

Characteristics:

    Simple architecture
    Fast training
    Suffers from vanishing gradients

**LSTM**

Long Short-Term Memory network:
Components:

    Input Gate
    Forget Gate
    Output Gate
    Cell State

Advantages:

    Handles long-term dependencies
    Reduces vanishing gradient problem

**GRU**

Gated Recurrent Unit:

Components:

    Update Gate
    Reset Gate

Advantages:

    Fewer parameters than LSTM
    Faster training
    Competitive performance

Experiments
## Experiment 1: HAR Classification

Models Compared:

    RNN
    LSTM
    GRU

Evaluation Verification:

    Unit Testing & Numerical Agreement: All three cells built from basic PyTorch matrix operations were verified against torch.nn.RNNCell, torch.nn.LSTMCell, and torch.nn.GRUCell. After copying layer parameters, maximum absolute tensor differences across random input batches were strictly <1×10−5.
●
Theoretical Parameter Derivations vs. Code (𝒅=𝟗, 𝒉=𝟔𝟒):
●
RNN Cell:
𝑃𝑎𝑟𝑎𝑚𝑠=ℎ⋅(𝑑+ℎ+1)=64⋅(9+64+1)=4,736
●
LSTM Cell:
𝑃𝑎𝑟𝑎𝑚𝑠=4⋅ℎ⋅(𝑑+ℎ+1)=4⋅64⋅(9+64+1)=18,944
●
GRU Cell:
𝑃𝑎𝑟𝑎𝑚𝑠=3⋅ℎ⋅(𝑑+ℎ+1)=3⋅64⋅(9+64+1)=14,208
●
PyTorch Parameter Count Reconciliation: PyTorch built-in layers report 4,800 (RNN), 19,200 (LSTM), and 14,400 (GRU) parameters because PyTorch maintains two independent bias vectors per gate (𝑏𝑖ℎ and 𝑏ℎℎ), adding 1⋅ℎ, 4⋅ℎ, and 3⋅ℎ extra parameters respectively. Custom cells fuse these into a single unified bias vector.

**Experiment 2: Adding Problem**

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

** HAR Classification Results**
All models used an Adam optimizer (𝑙𝑟=1×10−3), hidden size ℎ=64, batch size 64, and identical epoch budgets across 3 seeds.
Table 1: Performance Summary on UCI HAR Test Set


| Model | Parameters| Mean Test Accuracy (%) |Mean Macro F1|Sec / Epoch |
| :--- | :---: | ---: | ---: | ---: |
| Vanilla RNN | 5,190 | 77.57% ± 4.71% |0.7708 ± 0.0507|0.277s|


| Model | Parameters| Mean Test Accuracy (%) |Mean Macro F1|Sec / Epoch |
| :--- | :---: | ---: | ---: | ---: |
|LSTM| 19,590| 89.06% ± 0.88%| 0.8909 ± 0.0090| 0.280s|
|GRU| 14,790| 90.77% ± 0.83%| 0.9085 ± 0.0087| 0.277s|

Detailed Per-Seed Breakdown (from results.csv):

**● RNN:**

* Seed 42: Acc 77.81%, Macro F1 0.7741, 0.314s/epoch
*Seed 123: Acc 82.15%, Macro F1 0.8197, 0.270s/epoch
*Seed 999: Acc 72.75%, Macro F1 0.7184, 0.248s/epoch
●
LSTM:
○
Seed 42: Acc 88.06%, Macro F1 0.8806, 0.258s/epoch
○
Seed 123: Acc 89.45%, Macro F1 0.8946, 0.303s/epoch
○
Seed 999: Acc 89.68%, Macro F1 0.8975, 0.281s/epoch
●
GRU:
○
Seed 42: Acc 90.16%, Macro F1 0.9018, 0.254s/epoch
○
Seed 123: Acc 90.43%, Macro F1 0.9053, 0.295s/epoch
○
Seed 999: Acc 91.72%, Macro F1 0.9183, 0.282s/epoch Adding Problem Benchmark & Solved Runs
Runs are marked as solved if the final Test MSE is <0.05 (compared to baseline MSE ≈0.167).
Table 2: Test MSE and Solved Run Statistics across Sequence Lengths (𝑻)
Sequence Length (𝑇)
Model
Mean Test MSE
Solved Runs / Total
𝑻=𝟓𝟎
Vanilla RNN
0.1745 ± 0.0005
0 / 3
LSTM
0.0703 ± 0.0894
2 / 3
GRU
0.0058 ± 0.0035
3 / 3
𝑻=𝟏𝟎𝟎
Vanilla RNN
0.1701 ± 0.0005
0 / 3
LSTM
0.1700 ± 0.0013
0 / 3
GRU
0.0043 ± 0.0007
3 / 3
𝑻=𝟐𝟎𝟎
Vanilla RNN
0.1696 ± 0.0020
0 / 3
LSTM
0.1696 ± 0.0026
0 / 3
GRU
0.0082 ± 0.0036
3 / 3
𝑻=𝟒𝟎𝟎
Vanilla RNN
0.1573 ± 0.0055
0 / 3
LSTM
0.1566 ± 0.0055
0 / 3
GRU
0.1503 ± 0.0018
0 / 3



on model performance.
**5.	Gradients, ablations and analysis** 
5(a) Gradient Norm Analysis: Plotting |𝜕𝐿/𝜕ℎ𝑡| versus step index 𝑡 at 𝑇=100 demonstrates severe exponential decay in Vanilla RNNs prior to convergence. GRU architectures sustain consistent non-zero gradient values across earlier time steps, facilitating long-range temporal credit assignment.

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

    Python|    PyTorch|    NumPy|   Pandas|    Matplotlib|    Seaborn |    Scikit-Learn

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
