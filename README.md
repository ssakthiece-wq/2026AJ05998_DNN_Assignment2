# 2026AJ05998_DNN_Assignment2

1.	GitHub Repository Link

GitHub Repository: https://github.com/ssakthiece-wq/2026AJ05998_DNN_Assignment2

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

### 1.	Data pipelines :

1) Input Dimensions: 128 sequence steps across 9 inertial signal channels (Ntrain=7,352, Ntest=2,947).
2) Validation Split: 4 subjects were held out from the 21 training subjects using np.random.default_rng(seed).choice(train_subjects, 4, replace=False) for each of the 3 seeds (42, 123, 999).
3) Preprocessing: Per-channel mean and standard deviation were calculated strictly on the training partition of each split to prevent data leakage.

# Dataset URL:
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
1. Class balance Distribution

<img width="600" height="500" alt="har_confusion_matrix" src="https://github.com/user-attachments/assets/3abcdd8d-8e31-42f2-87fe-4e2fe4e4d160" />

Shows the distribution of activity classes in the HAR training set.

2. Confusion Matrix

<img width="700" height="400" alt="har_class_balance" src="https://github.com/user-attachments/assets/aff73a14-27ae-4a31-b13b-0ee95fa86844" />

Displays classification performance for the best-performing model.

3. Learning Curve

<img width="700" height="400" alt="adding_learning_curves_t200" src="https://github.com/user-attachments/assets/c46c9cda-7455-4325-8323-43cba2144c3d" />

<img width="800" height="400" alt="gradient_norms_t100" src="https://github.com/user-attachments/assets/9b7d16d4-854d-48cb-b867-b5a83532bc19" />

Compares training losses of:

    RNN
    LSTM
    GRU

for sequence length T = 200.

4. MSE vs Sequence Length

<img width="700" height="400" alt="ablation_study" src="https://github.com/user-attachments/assets/c81d2965-7c97-4174-8a43-ffac37e0043d" />

Demonstrates model robustness as sequence length increases.

5. Gradient Norm Analysis

<img width="800" height="400" alt="gradient_norms_t100" src="https://github.com/user-attachments/assets/52c7e855-21c8-4945-913b-133e32caaf7e" />


Illustrates the vanishing gradient phenomenon.

6. Hidden Size Sweep

<img width="700" height="400" alt="hidden_size_sweep" src="https://github.com/user-attachments/assets/cf40069e-953a-488d-ac71-440ee062e5b3" />


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
* Seed 123: Acc 82.15%, Macro F1 0.8197, 0.270s/epoch
* Seed 999: Acc 72.75%, Macro F1 0.7184, 0.248s/epoch
* 
**●LSTM:**
  
* Seed 42: Acc 88.06%, Macro F1 0.8806, 0.258s/epoch
* Seed 123: Acc 89.45%, Macro F1 0.8946, 0.303s/epoch
* Seed 999: Acc 89.68%, Macro F1 0.8975, 0.281s/epoch

**● GRU:**
* Seed 42: Acc 90.16%, Macro F1 0.9018, 0.254s/epoch
* Seed 123: Acc 90.43%, Macro F1 0.9053, 0.295s/epoch
* Seed 999: Acc 91.72%, Macro F1 0.9183, 0.282s/epoch

Adding Problem Benchmark & Solved Runs
Runs are marked as solved if the final Test MSE is <0.05 (compared to baseline MSE ≈0.167).
Table 2: Test MSE and Solved Run Statistics across Sequence Lengths (𝑻)

|Sequence Length (𝑇)| Mode| Mean Test MSE| Solved Runs Total|
| :--- | :---: | ---: | ---: | 
𝑻=𝟓𝟎| Vanilla RNN| 0.1745 ± 0.0005| 0 / 3
| | LSTM| 0.0703 ± 0.0894| 2 / 3
| |GRU| 0.0058 ± 0.0035| 3 / 3
𝑻=𝟏𝟎𝟎| Vanilla RNN| 0.1701 ± 0.0005| 0 / 3
| | LSTM| 0.1700 ± 0.0013| 0 / 3
| | GRU| 0.0043 ± 0.0007 | 3 / 3
𝑻=𝟐𝟎𝟎| Vanilla RNN| 0.1696 ± 0.0020| 0 / 3
| | LSTM| 0.1696 ± 0.0026| 0 / 3
| | GRU| 0.0082 ± 0.0036| 3 / 3
𝑻=𝟒𝟎𝟎 |Vanilla RNN| 0.1573 ± 0.0055| 0 / 3
| | LSTM| 0.1566 ± 0.0055| 0 / 3
| | GRU| 0.1503 ± 0.0018| 0 / 3



on model performance.

**5.	Gradients, ablations and analysis** 
5(a) Gradient Norm Analysis: Plotting |𝜕𝐿/𝜕ℎ𝑡| versus step index 𝑡 at 𝑇=100 demonstrates severe exponential decay in Vanilla RNNs prior to convergence. GRU architectures sustain consistent non-zero gradient values across earlier time steps, facilitating long-range temporal credit assignment.

**5(b) Hidden-Size Sweep 𝟏𝟔,𝟑𝟐,𝟔𝟒,𝟏𝟐𝟖 on HAR:**

Increasing ℎ from 16 to 64 steadily improves classification macro F1 score. Beyond ℎ=64, performance gains plateau while runtime increases linearly.

**5(c) Individual Ablation (LSTM Forget-Gate Bias Init 𝟎 vs. 𝟏 at 𝑻=𝟐𝟎𝟎):**

Initialising the forget-gate bias to 1.0 keeps the gate open early in training, preventing premature gradient vanishing and enabling convergence on the adding problem. Initialising to 0.0 leads to complete training stagnation (MSE ≈0.167).

Results are stored in:

results.csv

**Key Findings**

Gradient Dynamics & Long Memory: Vanilla RNNs collapse beyond sequence length 𝑇=50 due to vanishing gradients in unrolled backpropagation. GRUs maintain long-term memory via additive update paths:

ℎ𝑡=(1−𝑧𝑡)⊙ℎ𝑡−1+𝑧𝑡⊙ℎ̃𝑡

**● Wearable System Deployment Recommendation: For battery-powered wearable devices running HAR classification, GRU is the superior model:**

1.Accuracy & Stability: Highest mean accuracy (90.77%) and lowest seed variance (±0.83%).
2.Parameter Footprint: Uses 24.5% fewer parameters than LSTM (14,790 vs. 19,590), directly conserving memory access bandwidth and battery power.

    LSTM generally achieves the highest classification performance.
    GRU offers a strong trade-off between accuracy and computational efficiency.
    Vanilla RNN performance degrades for long sequences due to vanishing gradients.
    LSTM and GRU maintain better gradient flow and memory retention.
    Increasing hidden size improves performance up to a threshold after which gains become marginal.

Technologies Used

    Python|    PyTorch|    NumPy|   Pandas|    Matplotlib|    Seaborn |    Scikit-Learn

***2. Problem 2: Transformer from Scratch***

**Attention and Encoder-Block Unit Tests**

●Scaled Dot-Product Attention: Row softmax sum Σ𝑗𝐴𝑖𝑗=1.0 verified for all rows. Numerical outputs matched torch.nn.functional.scaled_dot_product_attention within tolerance <1×10−5.

●Causal Masking: Verified that applying upper-triangular −∞ logit masks sets attention weight to strictly 0.0 for all future positions (𝑗>𝑖).

**Parameter Count Derivation (𝒅=𝟏𝟐𝟖, 𝒅𝒇𝒇=𝟓𝟏𝟐)**

Theoretical derivation for a single Transformer Encoder block:

𝑃𝑟𝑜𝑗𝑒𝑐𝑡𝑖𝑜𝑛𝑊𝑒𝑖𝑔ℎ𝑡𝑠(𝑊𝑄,𝑊𝐾,𝑊𝑉,𝑊𝑂)=4⋅𝑑2=4⋅(128)2=65,536
𝐹𝑒𝑒𝑑−𝐹𝑜𝑟𝑤𝑎𝑟𝑑𝑁𝑒𝑡𝑤𝑜𝑟𝑘(𝑊1,𝑊2)=2⋅𝑑⋅𝑑𝑓𝑓=2⋅128⋅512=131,072
𝑇𝑜𝑡𝑎𝑙𝐵𝑙𝑜𝑐𝑘𝑊𝑒𝑖𝑔ℎ𝑡𝑃𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠=65,536+131,072=196,608

Reconciliation with PyTorch parameter count includes LayerNorm gains/biases (4𝑑) and projection layer biases
(4𝑑+𝑑𝑓𝑓+𝑑), matching sum(p.numel()) exactly.

**Task A: Transformer Encoder Classifier on UCI HAR Data**
**Table 3: Positional Encoding Ablation on HAR Dataset**

|Positional Encoding Variant| Parameters| Mean Test Accuracy (%)| Mean Macro F1
| :--- | :---: | ---: | ---: |
None (No Position Info)| 398,726| 87.89% ± 2.73%| 0.8765
Sinusoidal Encoding| 398,726| 88.51% ± 0.39%| 0.8836
Learned Encoding| 415,238| 88.79% ± 1.08%| 0.8853

●Analysis: Omitting positional information increases test variance across seeds (accuracy ranges from 85.58% to 90.91%), confirming that explicit positional awareness stabilizes temporal feature extraction. 

**Task B: Character Language Model on Tiny Shakespeare**

A 4-layer Causal Transformer Language Model (𝑑=128,ℎ=4, context length 128, total parameters = 810,049) was trained on Tiny Shakespeare.

** ●Validation Loss:** 1.7741𝑛𝑎𝑡𝑠 (Run 1: 1.7741, Run 2: 1.7748).
**Bits Per Character (BPC):**

𝐵𝑃𝐶=𝐿𝑜𝑠𝑠(𝑛𝑎𝑡𝑠)𝑙𝑛2=1.77410.693147=2.5595𝑏𝑝𝑐
● Training Throughput: ≈15.53𝑠𝑒𝑐/𝑒𝑝𝑜𝑐ℎ on Colab T4 GPU.
● Temperature Generation Analysis (𝑵=𝟓𝟎𝟎 characters)
● **Temperature 0.2:** Highly deterministic, repetitive phrasing ("the see the shall come the be...").
● **Temperature 0.8:** Balanced entropy, structured character relationships, realistic Shakespearean formatting.
● **Temperature 1.5:** High entropy, frequent gibberish character combinations and broken syntax ("trotuncnene, thekfuics doke...").

<img width="700" height="400" alt="lm_learning_curve" src="https://github.com/user-attachments/assets/f8634c0e-6db0-427d-9f9e-a7d7f155707b" />

**Analysis and Ablations**
●
Attention Heatmaps: Visualization shows Head 1 specializing in adjacent character transitions, while Head 2
tracks long-range syntactic patterns like line breaks and character names.
●
Causal Mask Removal Experiment: Removing the causal mask allows information leakage across time steps during training. Training loss drops immediately to ≈2.0𝑛𝑎𝑡𝑠 because tokens can directly inspect future answers. At test time, autoregressive generation completely fails because the network never learned causal dependency tracking.

<img width="700" height="450" alt="no_mask_comparison" src="https://github.com/user-attachments/assets/0633c5ce-6825-481c-a68b-6f5569a81c94" />

<img width="1200" height="500" alt="attention_heatmaps" src="https://github.com/user-attachments/assets/174281b1-a669-4f76-a49c-b0e3632f87b7" />

<img width="700" height="450" alt="heads_ablation" src="https://github.com/user-attachments/assets/b22c129f-4cfe-4680-9009-eafe4e2518e1" />


References

    UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.

    Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder.

    PyTorch Documentation: https://pytorch.org/

Author

SAKTHIVEL S.

Deep Neural Networks Assignment 2


## Links

- **GitHub Repository:** (https://github.com/ssakthiece-wq/2026AJ05998_DNN_Assignment2/)
