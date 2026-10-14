"""
DWTA-FL Configuration
All hyperparameters from Table 7 of the paper.
"""

import os

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# ============================================================================
# Dataset Configuration (Table 4)
# ============================================================================
DATASETS = {
    "CSE-CIC-IDS2018": {
        "original_records": 16_233_002,
        "retained_records": 15_742_821,
        "input_features": 78,
        "label_column": "Label",
        "attack_cat_column": None,
        "remove_columns": ["Timestamp", "Flow ID", "Src IP", "Src Port", "Dst IP"],
        "categorical_columns": [],
    },
    "UNSW-NB15": {
        "original_records": 2_540_044,
        "retained_records": 2_518_042,
        "input_features": 42,
        "label_column": "Label",
        "attack_cat_column": "attack_cat",
        "remove_columns": ["srcip", "sport", "dstip", "Stime", "Ltime"],
        "categorical_columns": ["proto", "state", "service"],
    },
    "TON_IoT": {
        "original_records": 461_043,
        "retained_records": 452_731,
        "input_features": 40,
        "label_column": "label",
        "attack_cat_column": "type",
        "remove_columns": ["ts", "src_ip", "dst_ip"],
        "categorical_columns": [
            "proto", "service", "conn_state", "dns_query",
            "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer",
            "http_method", "http_uri", "http_version",
            "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
            "weird_name", "weird_addl", "weird_notice",
        ],
    },
}

# ============================================================================
# Federated Learning Configuration (Table 7)
# ============================================================================
NUM_CLIENTS = 20                    # N: CSP clients
CLIENTS_PER_ROUND = 20             # Full participation
NUM_ROUNDS = 100                   # T: Communication rounds
LOCAL_EPOCHS = 5                   # Local epochs per round
BATCH_SIZE = 64                    # Mini-batch size
LEARNING_RATE = 0.001              # Adam learning rate
OPTIMIZER = "adam"

# MLP Architecture
MLP_HIDDEN_LAYERS = [512, 256, 128]  # Hidden layer sizes
MLP_ACTIVATION = "relu"
MLP_OUTPUT_UNITS = 2                 # Binary classification (softmax)

# Seeds
RUN_SEEDS = [101, 202, 303, 404, 505]
PARTITION_SEED = 42
DIRICHLET_ALPHA = 0.5              # Non-IID severity

# Train/Test split
TRAIN_TEST_RATIO = 0.8

# ============================================================================
# DWTA-FL Specific Parameters (Table 7)
# ============================================================================

# Drift Measurement (Section 3.3, Eq. 3-5)
DRIFT_HISTOGRAM_BINS = 30          # B: number of bins
DRIFT_BIN_RANGE = (0.0, 1.0)      # Bin range [0, 1]
DRIFT_SMOOTHING_LAMBDA = 0.7      # λ: smoothing coefficient

# Recursive Trust (Section 3.4, Eq. 6-7)
TRUST_MEMORY_BETA = 0.8           # β: trust memory coefficient
TRUST_EPSILON = 1e-8              # ε: numerical stability

# Aggregation Weight (Section 3.5, Eq. 10)
TRUST_EXPONENT_MU = 1.2           # μ: trust exponent
CONSISTENCY_EXPONENT_NU = 1.0     # ν: consistency exponent
DRIFT_PENALTY_GAMMA = 0.5         # γ: drift penalty
VOLATILITY_PENALTY_TAU = 0.4      # τ: volatility penalty

# Drift-versus-Attack State Rule (Section 3.6, Eq. 13-14)
DRIFT_THRESHOLD_THETA_D = 0.25    # θ_D: drift threshold
HIGH_RELIABILITY_THETA_H = 0.75   # θ_H: high reliability threshold
LOW_RELIABILITY_THETA_L = 0.40    # θ_L: low reliability threshold
GATE_SUSPICIOUS_KAPPA_S = 0.5     # κ_s: suspicious gate factor
GATE_ATTACKER_KAPPA_A = 0.1       # κ_a: attacker gate factor

# ============================================================================
# Orchestration Configuration (Section 3.9)
# ============================================================================

# Initial orchestration weights (Eq. 23)
ORCHESTRATION_LAMBDA_S = 0.35     # Severity weight
ORCHESTRATION_LAMBDA_P = 0.30     # Propagation weight
ORCHESTRATION_LAMBDA_R = 0.20     # Reliability/suitability weight
ORCHESTRATION_LAMBDA_C = 0.15     # Cost weight
ORCHESTRATION_ADAPTATION_ZETA = 0.01  # ζ: adaptation rate

# Action coefficients (Table 3)
ACTION_COEFFICIENTS = {
    "Monitor":            {"severity_gain": 0.20, "propagation_reduction": 0.10, "suitability": 1.00, "cost": 0.05},
    "Access Restriction": {"severity_gain": 0.55, "propagation_reduction": 0.45, "suitability": 0.80, "cost": 0.25},
    "Isolation":          {"severity_gain": 0.85, "propagation_reduction": 0.80, "suitability": 0.60, "cost": 0.55},
    "Quarantine":         {"severity_gain": 0.95, "propagation_reduction": 0.90, "suitability": 0.45, "cost": 0.75},
}

# ============================================================================
# Attack Configuration (Table S5)
# ============================================================================
MALICIOUS_CLIENT_SHARE = 0.20     # 20% = 4 of 20 clients
ATTACK_START_ROUND = 20
ATTACK_END_ROUND = 100

# Label poisoning
LABEL_POISON_FRACTION = 0.20      # 20% of local samples flipped

# Sign flipping
SIGN_FLIP_SCALE = 3.0             # Scale factor

# Backdoor
BACKDOOR_TRIGGER_FRACTION = 0.10  # 10% of local rows
BACKDOOR_TARGET_CLASS = 0         # Target class: benign

# Backdoor trigger features per dataset (Table S5)
BACKDOOR_TRIGGERS = {
    "CSE-CIC-IDS2018": {"Protocol": 0, "Dst Port": 0, "Flow Duration": 0},
    "UNSW-NB15": {"proto": "unas", "service": "-", "state": "INT"},
    "TON_IoT": {"proto": "tcp", "service": "-", "conn_state": "OTH"},
}

# Byzantine
BYZANTINE_NORM_SCALE = 3.0        # 3× legitimate update L2 norm

# Adaptive Byzantine
ADAPTIVE_BYZANTINE_NORM_SCALE = 3.0  # Norm ≤ 3× benign median

# Collusion
COLLUSION_SCALE = 3.0

# Covert gradient escape
COVERT_NORM_DEVIATION = 0.15      # ≤ 15% deviation from benign norm

# ============================================================================
# Drift Schedule (Table 5)
# ============================================================================
DRIFT_SCHEDULE = {
    "warmup": {"rounds": (1, 20), "drift_type": "none", "clients": [], "fraction": 0.0},
    "label_shift": {
        "rounds": (21, 60),
        "drift_type": "label_proportion",
        "clients": [4, 9, 14, 19],  # Clients 5,10,15,20 (0-indexed)
        "fraction_range": (0.10, 0.30),
        "onset": "gradual",  # Linear ramp over rounds 21-40
    },
    "covariate_shift": {
        "rounds": (61, 100),
        "drift_type": "covariate",
        "clients": [2, 6, 11, 15, 18],  # Clients 3,7,12,16,19 (0-indexed)
        "fraction_range": (0.20, 0.30),
        "onset": "sudden",  # Applied at round 61
    },
}

# ============================================================================
# Convergence Monitoring (Section 3.10, Eq. 24)
# ============================================================================
STABILITY_EPSILON = 0.001         # ε_stab: monitoring limit
CONVERGENCE_TOLERANCE = 0.001     # Loss change tolerance
CONVERGENCE_WINDOW = 5            # Consecutive rounds

# ============================================================================
# Orchestration Test (Section 4.8)
# ============================================================================
NUM_ORCHESTRATION_DOMAINS = 3     # Cloud A, B, C
ORCHESTRATION_WINDOW_SEC = 5      # 5-second windows
ORCHESTRATION_SCENARIO_MIN = 30   # 30-minute scenarios
ORCHESTRATION_RUNS = 10
ORCHESTRATION_RECOVERY_WINDOWS = 12  # 60 seconds

# Threat states (Table 18)
THREAT_STATES = {
    "Normal":            {"severity_max": 0.3, "propagation_max": 0.2, "action": "Monitor"},
    "Localised threat":  {"severity_min": 0.3, "propagation_max": 0.2, "action": "Access Restriction"},
    "Propagating threat":{"severity_max": 0.2, "propagation_min": 0.7, "action": "Isolation"},
    "Coordinated threat":{"severity_min": 0.7, "propagation_min": 0.2, "action": "Quarantine"},
    "Recovery":          {"action": "Monitor"},  # Severity < 0.3 for 3 consecutive windows
}

# ============================================================================
# Baseline Hyperparameters (Table 8)
# ============================================================================
BASELINE_CONFIG = {
    "FLTrust": {
        "root_dataset_size": 1000,
    },
    "FedSBS": {
        "selected_per_round": 20,
        "momentum": 0.9,
    },
    "FedMADE": {
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 3,
        "auxiliary_set_size": 2000,
    },
    "Krum": {
        "assumed_byzantine": 4,  # f = 4
    },
    "Bulyan": {
        "assumed_byzantine": 4,  # f = 4
    },
    "FLEX-IDS": {
        "threshold": 0.6,
        "pruning_rate": 0.1,
    },
    "FL-IDS++": {
        "personalization_weight": 0.5,
    },
    "FL-IDPP": {
        "noise_multiplier": 0.01,
        "clipping_norm": 1.0,
    },
    "Fed-ANIDS": {
        "anomaly_threshold": 0.75,
    },
}

# ============================================================================
# Feature heterogeneity (Section 4.1.4, H2)
# ============================================================================
FEATURE_HETERO_FEATURES = ["Flow Byts/s", "Flow Pkts/s", "TotLen Fwd Pkts"]
FEATURE_HETERO_SCALE_RANGE = (0.7, 1.3)

# ============================================================================
# Checkpoint and run presets (used by train.py / test.py)
# ============================================================================
CHECKPOINT_PATH = os.path.join(BASE_DIR, "bestmodel.pt")

RUN_PRESETS = {
    "smoke": {
        "rounds": 1,
        "sample_limit": 300,
        "detection_seeds": RUN_SEEDS[:2],
        "single_seeds": RUN_SEEDS[:1],
    },
    "quick": {
        "rounds": 3,
        "sample_limit": 1000,
        "detection_seeds": RUN_SEEDS[:3],
        "single_seeds": RUN_SEEDS[:1],
    },
    "full": {
        "rounds": NUM_ROUNDS,
        "sample_limit": None,
        "detection_seeds": RUN_SEEDS,
        "single_seeds": RUN_SEEDS,
    },
}

# One-at-a-time sensitivity sweep grids (Figures 8-11)
SENSITIVITY_GRIDS = {
    "theta_d": [0.10, 0.20, 0.25, 0.30, 0.40],
    "beta": [0.5, 0.6, 0.7, 0.8, 0.9],
    "gamma": [0.2, 0.4, 0.5, 0.6, 0.8],
    "tau": [0.2, 0.3, 0.4, 0.5, 0.6],
}
