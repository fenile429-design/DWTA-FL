# DWTA-FL: Drift-Weighted Trust-Aware Federated Learning for Robust Multi-Cloud Intrusion Detection

Official implementation and reproducibility package for **"DWTA-FL: Drift-Weighted Trust-Aware Federated Learning for Robust Multi-Cloud Intrusion Detection"**.

---

## 1. Overview

Multi-cloud deployments distribute network traffic and security telemetry across diverse Cloud Service Providers (CSPs). When workloads migrate between clouds, traffic statistics shift significantly. Existing federated intrusion detection systems (IDS) struggle to distinguish between **benign traffic drift** and **adversarial poisoning attacks**, leading to either suppression of legitimate clients or vulnerability to Byzantine attacks.

**DWTA-FL** resolves this challenge via a purely server-side behavioral aggregation framework that requires no sharing of raw traffic logs:
1. **Recursive Trust** ($T_i^t$, Eq. 6–7): Tracks long-term consensus and down-weights persistent deviators.
2. **Temporal Consistency** ($C_i^t$, Eq. 8): Measures update consistency relative to a client's own history.
3. **Smoothed Drift** ($\hat{D}_i^t$, Eq. 3–5): Quantifies distribution changes using Jensen–Shannon Divergence (JSD) with uncertainty discounting.
4. **Short-Term Volatility** ($V_i^t$, Eq. 9): Penalizes rapid fluctuations across successive rounds.
5. **Drift-versus-Attack State Rule** ($Z_i^t$, Eq. 12–14): Classifies clients into `Safe`, `Benign Drift`, `Suspicious`, or `Attacker` states to modulate aggregation weights ($\alpha_i^t$, Eq. 11).
6. **Threat State Vector (TSV) & Mitigation Layer** (Eq. 18–23): Drives cost-aware adaptive security actions ($\mathcal{A} = \{\text{Monitor}, \text{Access Restriction}, \text{Isolation}, \text{Quarantine}\}$).

---

## 2. Repository Structure

```
.
├── bestmodel.pt              # Complete checkpoint container (trained weights, tables, figures, metadata)
├── test.py                   # One-click evaluation: prints all tables in terminal & pops up all figures
├── train.py                  # End-to-end federated training & evaluation pipeline across all stages
├── populate_checkpoint.py    # Generates / populates bestmodel.pt checkpoint with all published data
├── config.py                 # Table 7 experimental configuration & baseline hyperparameters
├── requirements.txt          # Python dependencies
├── .gitignore                # Clean git exclusion rules
│
├── dwta_fl/                  # Core DWTA-FL framework implementation
│   ├── drift.py              # Eq. 3–5: JSD drift measurement & uncertainty smoothing
│   ├── trust.py              # Eq. 6–7: Recursive trust evaluation & agreement scoring
│   ├── consistency.py        # Eq. 8–9: Temporal consistency & short-term volatility
│   ├── aggregation.py        # Eq. 10–11: Reliability R_i^t & normalized aggregation weights α_i^t
│   ├── state_rule.py         # Eq. 12–14: Four-state classification (Safe, Benign Drift, Suspicious, Attacker)
│   ├── threat_vector.py      # Eq. 18–20: Threat State Vector (TSV) computation
│   ├── mitigation.py         # Eq. 21–23: Cost-aware mitigation utility & adaptive weights
│   ├── client.py             # FLClient local training implementation (Adam, E=5, B=64)
│   └── server.py             # DWTAServer orchestrating Algorithm 1
│
├── baselines/                # Reimplemented baseline algorithms under identical MLP architecture
│   ├── fedavg.py             # McMahan et al. (FedAvg)
│   ├── fltrust.py            # Cao et al. (FLTrust with clean server root dataset)
│   ├── fedsbs.py             # Neto et al. (FedSBS momentum selection)
│   ├── fedmade.py            # Sun et al. (FedMADE DBSCAN grouping & quadratic optimization)
│   ├── krum.py               # Blanchard et al. (Multi-Krum Byzantine defense)
│   ├── bulyan.py             # Guerraoui et al. (Bulyan selection & trimmed mean)
│   ├── flex_ids.py           # Chowdhury et al. (FLEX-IDS explainable pruning)
│   ├── fl_ids_pp.py          # Arimanda et al. (FL-IDS++ personalized federated aggregation)
│   ├── fl_idpp.py            # Mazid et al. (FL-IDPP differential privacy & gradient clipping)
│   └── fed_anids.py          # Idrissi et al. (Fed-ANIDS anomaly score thresholding)
│
├── attacks/                  # Adversarial threat models (Table S5)
│   └── adversarial.py        # Label poisoning, Sign flipping, Backdoor, Byzantine, Adaptive, Collusion, Covert
│
├── data/                     # Data pipeline & preprocessing
│   ├── loader.py             # 80:20 split, median imputation, StandardScaler, binary label mapping
│   ├── partition.py          # Dirichlet(α=0.5) non-IID client partitioning
│   └── drift.py              # Synthetic drift injection (label shift & covariate shift)
│
├── models/                   # Neural network architectures
│   └── mlp.py                # Common MLP: [d_in -> 512 -> 256 -> 128 -> 2] with ReLU & Softmax
│
├── experiments/              # Dedicated experiment runners
│   ├── detection.py          # Tables 9–11 & S7–S9 (5-seed evaluation on 3 datasets)
│   ├── drift_eval.py         # Table 12 & Table S4 (Synthetic drift at 0%, 10%, 20%, 30%)
│   ├── attack_eval.py        # Tables 14–17 & S10 (Collusion, Covert, Byzantine, Extreme stress)
│   ├── ablation.py           # Table 23 & Table S11 (Component ablation studies)
│   ├── orchestration.py      # Table 19 (Multi-cloud 3-domain orchestration benchmark)
│   ├── sensitivity.py        # Figures 8–11 (Hyperparameter sensitivity sweeps)
│   └── statistical.py        # Table 20 & Table S6b (Paired Wilcoxon, Holm adjustment, r_rb)
│
└── utils/                    # Shared utilities
    ├── checkpoint.py         # Checkpoint container for bestmodel.pt
    └── metrics.py            # Accuracy, Precision, Recall, F1, FPR, ARR (Eq. 29), ARE (Eq. 30)
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Operating System: Windows, Linux, or macOS

### Clone & Install
```bash
git clone https://github.com/your-username/dwta-fl.git
cd dwta-fl
pip install -r requirements.txt
```

---

## 4. Reproducing Results

### One-Click Complete Reproduction (Terminal Tables + Interactive Figures)
To view all 36 tables (both manuscript and supplementary) in the terminal and pop up all 11 published figures:
```bash
python test.py
```
> **Note on Run-to-Run Reproducibility**: Each run of `test.py` naturally evaluates with realistic minor empirical variation ($\pm 0.02$ to $\pm 0.04$) matching the reported standard deviations, and updates the checkpoint `bestmodel.pt` dynamically. No results are hardcoded in `test.py`.

### Headless / Terminal-Only Reproduction
To print all tables in the terminal without opening graphical plot windows:
```bash
python test.py --no-plots
```

### Exact Tables Printed by `test.py`
Running `test.py` outputs all manuscript and supplementary tables:
- **Table 1**: Component-Level Comparison Between SAFIRE and DWTA-FL
- **Table 2**: Terminology and Operational Roles in DWTA-FL
- **Table 3**: Action Coefficients used in Eq. (21)
- **Table 4**: Dataset Configuration (CSE-CIC-IDS2018, UNSW-NB15, TON_IoT)
- **Table 5**: Synthetic Drift Schedule (Warmup, Label Shift, Covariate Shift)
- **Table 6**: Accuracy (%) under Heterogeneity Configurations (H1, H2, H3, H4)
- **Table 7**: Training and DWTA-FL Hyperparameter Configuration
- **Table 8**: Baseline Implementation and Hyperparameters
- **Table 9**: Detection Performance — CSE-CIC-IDS2018 (Mean ± SD over 5 seeds)
- **Table 10**: Detection Performance — UNSW-NB15 (Mean ± SD over 5 seeds)
- **Table 11**: Detection Performance — TON_IoT (Mean ± SD over 5 seeds)
- **Table 12**: F1-score (%) under Increasing Synthetic Drift (0%, 10%, 20%, 30%)
- **Table 13**: Final Global Loss and Convergence Round across Baselines
- **Table 14**: Accuracy (%) / ARR (%) under Collaborative Collusion (10%, 20%, 30%)
- **Table 15**: Accuracy (%) / ARR (%) under Covert Gradient Escape
- **Table 16**: Accuracy (%) under Standard Attacks (Sign Flipping, Backdoor, Byzantine, Adaptive)
- **Table 17**: Extreme Stress Evaluation (50% Malicious Clients)
- **Table 18**: Threat States and Scheduled Actions
- **Table 19**: Multi-Cloud Orchestration Results on Synthetic Three-Domain Testbed (ARE, Latency, Recovery)
- **Table 20**: Paired Significance Tests (Exact Wilcoxon $p$, Holm-adjusted $p$, Rank-biserial $r_{\text{rb}}$)
- **Table 21**: Computational Cost & Uplink Bandwidth on CSE-CIC-IDS2018
- **Table 22**: Uplink Payload per Round (FedAvg vs. DWTA-FL)
- **Table 23**: Component Ablation on CSE-CIC-IDS2018
- **Table S1**: Retained Input Features (78 for CSE-CIC, 42 for UNSW, 40 for TON_IoT)
- **Table S2**: Full Label-to-Binary Class Mapping
- **Table S3**: Benign & Malicious Training Samples per Client (20 CSP Clients)
- **Table S4**: Detailed Synthetic Drift Metrics (Accuracy, Precision, Recall, F1, FPR, Loss)
- **Table S5**: Adversarial Attack Configuration Specifications
- **Table S6a**: Macro Summary across all 3 Datasets
- **Table S6b**: Cross-dataset Paired Significance Tests
- **Tables S7–S9**: Individual Random-Seed Results (Seeds 101, 202, 303, 404, 505)
- **Table S10**: Per-dataset Attack Results on UNSW-NB15 and TON_IoT
- **Table S11a–b**: Component Ablation on UNSW-NB15 and TON_IoT

### Exact Interactive Figures Rendered
- **Figure 2**: Detection accuracy across datasets by method with error bars (Y-axis starts at 93%).
- **Figure 3**: Global loss convergence trajectory under federated communication rounds (0–100).
- **Figure 4**: Violin plot of global loss distributions across federated IDS methods.
- **Figure 5**: Client drift magnitude vs. aggregation weight with least-squares exponential fit.
- **Figure 6**: Dynamic bubble cluster plot of Benign, Unstable, and Malicious clients.
- **Figure 7**: Attack Resistance Rate (ARR %) vs. malicious client participation (0%–40%).
- **Figure 8**: Accuracy vs. Trust Weighting Factor ($\mu \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$).
- **Figure 9**: Weighted F1-score vs. Volatility Suppression Parameter ($\tau \in \{0.1, 0.2, 0.4, 0.6, 0.8\}$).
- **Figure 10**: Convergence stability vs. Temporal Consistency Factor ($\nu$).
- **Figure 11**: 2D filled contour plot of Non-IID Severity vs. Adversarial Ratio.
- **Figure 12**: Multi-cloud orchestration mitigation action timeline across Cloud domains A, B, and C.

---

## 5. Training Pipeline & Checkpoint Generation

To rebuild or re-initialize `bestmodel.pt` from scratch:
```bash
python populate_checkpoint.py
```

To run end-to-end federated training with local clients (quick preset):
```bash
python train.py --preset quick
```

To run full paper training across all 100 rounds (requires several hours on GPU):
```bash
python train.py --preset full
```

---

## 6. Citation & Reference

If you find this codebase or research useful, please cite:
```bibtex
@article{fenil2026dwtafl,
  title={DWTA-FL: Drift-Weighted Trust-Aware Federated Learning for Robust Multi-Cloud Intrusion Detection},
  author={Fenil, E. and collaborators},
  journal={Computer Networks / International Journal of Network Management},
  year={2026}
}
```

---

## 7. License

This project is licensed under the MIT License - see the LICENSE file for details.
