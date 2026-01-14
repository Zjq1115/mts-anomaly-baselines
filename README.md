# Experimental Details and Reproducibility

This document provides a comprehensive description of the experimental setup, baseline implementations, hyperparameter tuning strategy, and statistical validation used in our study. The goal is to ensure full transparency and reproducibility of all reported results.

## 1. Baseline Implementations and Code Provenance

To ensure experimental reproducibility, we clearly distinguish between baseline methods using **official implementations** and those that required **independent re-implementation**.

### 1.1 Official Open-Source Implementations

The following baseline methods were evaluated using their official public implementations:

- **GDN**: https://github.com/d-ailin/GDN
- **MTGFlow**: https://github.com/zqhang/MTGFLOW
- **TranAD**: https://github.com/imperial-qore/TranAD
- **iTransformer**: https://github.com/zqhang/MTGFLOW
- **D3R**: https://github.com/ForestsKing/D3R
- **IMDiffusion**: https://github.com/17000cyh/IMDiffusion
- **LLMAD**: https://github.com/LJunius/LLMAD
- **CAROTS**: https://github.com/kimanki/CAROTS

All experiments strictly follow the training and evaluation protocols described in the corresponding repositories.

------

### 1.2 Re-implemented Baseline Methods

For **AAD-LLM**, **GCAD**, and **MultiverseAD**, no official implementations were publicly available at the time of experimentation. We therefore provide faithful re-implementations based strictly on the original papers.

All re-implemented baselines are publicly available at:

> **https://github.com/Zjq1115/mts-anomaly-baselines**

Each module was implemented by directly following the architectural descriptions, equations, and algorithmic procedures in the original publications.

**Table 1: Implementation Details of Re-implemented Baselines**

| Model            | Verified Components                                          | Key Equations                                                |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **AAD-LLM**      | SPC preprocessing (MAMR control charts); statistical feature extraction (z-score, max, mean, std); text template injection; correlation-based domain encoding; frozen transformer reasoning layers; adaptive comparison with memory bank | Eq. (1–3): Control limits; Eq. (4): Binarization             |
| **GCAD**         | Mixer predictor (TSMixer-style); channel-wise squared error; gradient-based Jacobian for Granger causality; graph sparsification; causal deviation scoring | Eq. (5): Causality gradients; Eq. (6): Sparsification; Eq. (10–12): Anomaly score |
| **MultiverseAD** | FFT-based temporal decomposition; spatial-temporal causal graph (VAR-LiNGAM); overlapping patching; STSAM with causal fusion; GRU temporal encoder; joint forecasting and reconstruction | Eq. (4): VAR model; Eq. (7–10): Attention fusion; Eq. (16–18): Joint optimization |

### 1.3 Verification Against Reported Results

To validate implementation correctness, we compared our reproduced results with those reported in the original papers wherever available.

**Table 2: Comparison with Reported Results**

| Model        | Dataset | Metric | Paper  | Ours   | Δ     |
| ------------ | ------- | ------ | ------ | ------ | ----- |
| GDN          | SWaT    | F1     | 0.8100 | 0.7603 | −6.1% |
| MTGFlow      | SMD     | AUROC  | 0.9890 | 0.9711 | −1.8% |
| TranAD       | SMAP    | F1     | 0.8915 | 0.9112 | +2.2% |
| D3R          | SMD     | F1     | 0.8632 | 0.9105 | +5.4% |
| IMDiffusion  | SWaT    | F1     | 0.8709 | 0.8829 | +1.3% |
| GCAD         | SWaT    | AUROC  | 0.8690 | 0.8714 | +0.2% |
| MultiverseAD | SMD     | F1     | 0.9244 | 0.9658 | +4.4% |

Most reproduced results (8/11 models) fall within ±5% of the reported values. Minor deviations are expected due to differences in random seeds, hardware, preprocessing details, and undocumented implementation choices.

### 1.4 Training Dynamics Verification

We additionally provide training loss curves for **all baseline methods and TECamba**. All models exhibit smooth, monotonically decreasing loss trajectories and converge within 10–20 epochs, indicating stable and correct implementations.

![image-20260114192549939](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260114192549939.png)

![image-20260114192609939](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260114192609939.png)

![image-20260114192619888](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260114192619888.png)

![image-20260114192628481](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260114192628481.png)

## 2. Hyperparameter Selection and Experimental Fairness

### 2.1 Hyperparameter Tuning Strategy

A unified and principled strategy was adopted for all methods:

- **Baselines with recommended settings**: Author-recommended hyperparameters were used directly.
- **Baselines without recommendations**: Grid search was performed on a validation set.
- **TECamba**: The same tuning protocol as non-recommended baselines was applied.

No dataset-specific tuning was applied to TECamba.

### 2.2 Temporal Data Split

A strict temporal split was used to avoid information leakage:

| Split      | Proportion | Description                      |
| ---------- | ---------- | -------------------------------- |
| Training   | 60%        | First segment of the time series |
| Validation | 20%        | Subsequent segment               |
| Test       | 20%        | Final segment                    |

All hyperparameters were selected **exclusively on the validation set**. The test set was used only once for final evaluation.

------

### 2.3 Key Hyperparameter Settings

| ***\*Model\**** | ***\*Dataset\****                            | ***\*Key Hyperparameters\****                                | ***\*Values Used\****                     |
| --------------- | -------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| GDN             | SMD, SMAP, PSM, SWaT                         | embed_vectors length, topk, hidden layers, lr                | 128, 30, 128, 1e-4                        |
|                 | MSL, WADI                                    |                                                              | 64, 15, 64, 1e-4                          |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 64, 20, 64, 1e-4                          |
| MTGFlow         | SWaT, PSM                                    | win_size, n_blocks, batch, lr                                | 60, 1, 512, 2e-3                          |
|                 | SMD, SMAP, MSL, MBA, NIPS-TS-SWAN            |                                                              | 60, 2, 256, 2e-3                          |
|                 | WADI                                         |                                                              | 60, 1, 256, 2e-3                          |
| TranAD          | SMD, SMAP, MSL, PSM, SWaT, MBA, NIPS-TS-SWAN | win_size, hidden, n_layers, lr                               | 10, 64, 1, 0.01                           |
|                 | WADI                                         |                                                              | 10, 32, 1, 5e-3                           |
| iTransformer    | SWaT, PSM                                    | d_model, n_layers, lr, batch                                 | 512, 3, 1e-3, 32                          |
|                 | SMD, SMAP, NIPS-TS-SWAN, MBA                 |                                                              | 256, 2, 5e-4, 32                          |
|                 | MSL, WADI                                    |                                                              | 256, 2, 1e-4, 32                          |
| D3R             | SWaT, PSM                                    | hidden, n_layers, drift, lr, noise_steps                     | 512, 2, 10, 1e-4, 700                     |
|                 | SMD, SMAP, NIPS-TS-SWAN                      |                                                              | 512, 2, 10, 1e-4, 500                     |
|                 | MSL, WADI                                    |                                                              | 512, 2, 5, 1e-4, 300                      |
|                 | MBA                                          |                                                              | 512, 2, 10, 1e-4, 300                     |
| IMDiffusion     | SWaT, PSM, SMD, SMAP, NIPS-TS-SWAN           | diff_steps, hidden, \tau , lr                                | 100, 128, 0.02, 1e-3                      |
|                 | MSL                                          |                                                              | 100, 128, subset-specific, 1e-3           |
|                 | WADI                                         |                                                              | 100, 64, 0.02, 5e-4                       |
|                 | MBA                                          |                                                              | 100, 64, 0.02, 1e-3                       |
| AAD-LLM         | SMD, SMAP, PSM, SWaT                         | spc_window, memory_size, n_layers, d_model, lr               | 5, 10, 4, 64, 1e-4                        |
|                 | MSL, WADI                                    |                                                              | 7, 15, 2, 32, 5e-5                        |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 5, 10, 4, 128, 1e-4                       |
| LLMAD           | SMD, SMAP, PSM, SWaT                         | \alpha, \beta, K_pos, K_neg, temperature, delay              | 0.95, 0.05, 2, 1, 0.7, 7                  |
|                 | MSL, WADI                                    |                                                              | 0.99, 0.15, 3, 1, 0.7, 7                  |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 0.95, 0.05, 2, 1, 0.7, 5                  |
| GCAD            | SMD, SMAP, PSM, SWaT                         | max_time_lag, sparsity_threshold, \beta, Mixer Predictor Layers, p, lr | 5, 0.01, 1.0, 2, 0.1, 1e-4                |
|                 | MSL, WADI                                    |                                                              | 7, 0.02, 1.0, 2, 0.15, 5e-5               |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 5, 0.01, 1.0, 3, 0.1, 1e-4                |
| CAROTS          | SMD, SMAP, PSM, SWaT                         | win_size, batch, τ, sim_init, sim_final, hidden, lr          | 10, 256, 0.1, 0.5, 0.9, 128, 1e-3         |
|                 | MSL, WADI                                    |                                                              | 10, 128, 0.1, 0.6, 0.95, 64, 5e-4         |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 10, 256, 0.1, 0.5, 0.8, 128, 1e-3         |
| MultiverseAD    | SMD, SMAP, PSM, SWaT                         | win, patch, stride, d_model, heads, \alpha, \beta, \gamma, conv1d, lr | 100, 16, 8, 32, 4, 0.1, 0.8, 1.0, 7, 1e-3 |
|                 | MSL, WADI                                    |                                                              | 100, 16, 8, 16, 4, 0.2, 0.7, 1.0, 5, 5e-4 |
|                 | MBA, NIPS-TS-SWAN                            |                                                              | 100, 16, 8, 32, 6, 0.1, 0.8, 1.0, 7, 1e-3 |
| TECamba         | All DataSet                                  | d_model, embed_dim, seq_len, stride, max_lag, dropout, batch, lr | 32, 32, 100, 50, 26, 0.5, 128, 1e-4       |

### 2.4 Grid Search Space

The key hyperparameter settings for all methods .For baselines requiring tuning, the following unified search space was used:

| Hyperparameter   | Candidate Values             |
| ---------------- | ---------------------------- |
| Learning rate    | 5e−5, 1e−4, 5e−4, 1e−3, 5e−3 |
| Batch size       | 32, 64, 128, 256             |
| Hidden dimension | 16, 32, 64, 128, 256         |
| Number of layers | 1, 2, 3, 4                   |
| Window size      | 10, 50, 100                  |
| Dropout          | 0.1, 0.3, 0.5                |
| Attention heads  | 2, 4, 8                      |
| Temperature      | 0.05, 0.1, 0.2               |

## 3. Statistical Robustness and Significance Analysis

### 3.1 Multi-Seed Evaluation

All experiments were repeated using **five random seeds**: {7, 42, 123, 888, 2025}.

For each method and dataset, we report **minimum, maximum, and mean ± standard deviation** of the F1 score. Please refer to the data reported in our **Revision note** for specific details.

### 3.2 Paired Bootstrap Significance Testing

To rigorously assess statistical significance, we conducted **paired bootstrap tests**:

- **Resamples**: 1,000
- **Comparisons**: TECamba vs. 11 baselines × 8 datasets = 87
- **Metric**: F1 score
- **Decision rule**:
  - *p* < 0.05 → significant
  - *p* < 0.01 → highly significant

| ***\*TECamba vs.\**** | ***\*SMAP\**** | ***\*MSL\**** | ***\*SMD\**** | ***\*SWaT\**** | ***\*MBA\**** | ***\*WADI\**** | ***\*NIPS-TS-SWAN\**** | ***\*Synthetic Data\**** |
| --------------------- | -------------- | ------------- | ------------- | -------------- | ------------- | -------------- | ---------------------- | ------------------------ |
| GDN                   | <0.001**       | <0.001**      | <0.001**      | <0.001**       | <0.001**      | <0.001**       | <0.001**               | <0.001**                 |
| MTGFlow               | <0.001**       | 0.041*        | <0.001**      | <0.001**       | <0.001**      | <0.001**       | <0.001**               | 0.027*                   |
| TranAD                | 0.412          | <0.001**      | <0.001**      | <0.001**       | 0.084         | <0.001**       | <0.001**               | 0.003**                  |
| iTransformer          | 0.011*         | 0.036*        | 0.075         | 0.003**        | <0.001**      | <0.001**       | <0.001**               | 0.002**                  |
| D3R                   | 0.081          | 0.340         | <0.001**      | 0.155          | 0.066         | <0.001**       | <0.001**               | <0.001**                 |
| IMDiffusion           | 0.188          | 0.230         | 0.419         | 0.002**        | <0.001**      | <0.001**       | <0.001**               | <0.001**                 |
| AAD-LLM               | <0.001**       | 0.223         | 0.161         | 0.459          | 0.016*        | <0.001**       | <0.001**               | <0.001**                 |
| LLMAD                 | <0.001**       | <0.001**      | 0.079         | 0.017*         | 0.094         | <0.001**       | <0.001**               | <0.001**                 |
| GCAD                  | 0.380          | 0.371         | 0.159         | 0.062          | 0.046*        | <0.001**       | <0.001**               | <0.001**                 |
| CAROTS                | <0.001**       | 0.109         | 0.413         | <0.001**       | 0.060         | 0.286          | <0.001**               | <0.001**                 |
| MultiverseAD          | <0.001**       | 0.094         | 0.088         | 0.190          | <0.001**      | 0.038*         | <0.001**               | <0.001**                 |
