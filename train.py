"""
train.py - runs every DWTA-FL experiment and writes bestmodel.pt.

bestmodel.pt receives (and is re-saved after every stage / improvement):
  * the best DWTA-FL model weights found during the detection runs
  * the raw measured results of every experiment
  * every manuscript / supplementary table derived from those results
  * every figure specification

Nothing is hard-coded: tables come from config.py (settings) or from the runs.
test.py only loads the checkpoint and prints / plots what is inside.

Usage:
  python train.py                      # quick preset, all stages
  python train.py --preset full        # full paper protocol (many hours)
  python train.py --stages detection   # run selected stages only (results accumulate)
  python train.py --fresh              # ignore an existing bestmodel.pt
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.checkpoint import Checkpoint
from experiments.detection import run_federated_training
from experiments.drift_eval import run_drift_experiment_single
from experiments.attack_eval import run_attack_experiment
from experiments.ablation import run_dwta_variant, ABLATION_VARIANTS
from experiments.orchestration import run_orchestration_simulation
from experiments.statistical import run_statistical_analysis
from dwta_fl.mitigation import ACTIONS

DETECTION_METHODS = [
    "DWTA-FL", "FedAvg", "FLTrust", "FedSBS", "FedMADE", "Krum", "Bulyan",
    "FLEX-IDS", "FL-IDS++", "FL-IDPP", "Fed-ANIDS",
]
DRIFT_METHODS = ["FedAvg", "FLTrust", "FedSBS", "FedMADE", "DWTA-FL"]
ATTACK_METHODS = ["FedAvg", "FLTrust", "Krum", "Bulyan", "DWTA-FL"]
DRIFT_LEVELS = [0.0, 0.10, 0.20, 0.30]
FRACTION_SWEEP = [0.10, 0.20, 0.30]
STAGES = ["detection", "drift", "attack", "ablation", "sensitivity", "orchestration"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def mean_sd(values):
    values = [float(v) for v in values]
    ddof = 1 if len(values) > 1 else 0
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=ddof):.2f}"


def metric_list(runs, key):
    return [r[key] for r in runs]


def attack_configs():
    """(label, attack name, malicious fraction)."""
    cfgs = []
    base = config.MALICIOUS_CLIENT_SHARE
    for name in ["label_poisoning", "sign_flipping", "backdoor", "byzantine", "adaptive_byzantine"]:
        cfgs.append((name, name, base))
    for name in ["collusion", "covert_escape"]:
        for frac in FRACTION_SWEEP:
            cfgs.append((f"{name} ({int(round(frac * 100))}%)", name, frac))
    cfgs.append(("extreme_stress (50%)", "collusion", 0.50))
    return cfgs


# ----------------------------------------------------------------------------
# stages (each only fills ck.data["raw"])
# ----------------------------------------------------------------------------
def stage_detection(ck, preset, datasets):
    raw = ck.data["raw"].setdefault("detection", {})
    stats_raw = ck.data["raw"].setdefault("client_stats", {})
    for ds in datasets:
        for method in DETECTION_METHODS:
            for seed in preset["detection_seeds"]:
                res = run_federated_training(
                    dataset_name=ds, method=method, seed=seed,
                    num_rounds=preset["rounds"], sample_limit=preset["sample_limit"],
                )
                raw.setdefault(ds, {}).setdefault(method, {})[seed] = res["metrics"]
                stats_raw[ds] = res["client_stats"]

                if method == "DWTA-FL":
                    improved = ck.update_best_model(
                        res["weights"],
                        score=res["metrics"]["f1"],
                        info={
                            "dataset": ds, "seed": seed, "input_dim": res["input_dim"],
                            "rounds": res["num_rounds"], "metrics": res["metrics"],
                        },
                    )
                    if improved:
                        hist = res["history"]
                        clients = sorted(hist[0]["trust"].keys())
                        ck.data["raw"]["best_history"] = {
                            "dataset": ds, "seed": seed,
                            "norm_diff": [float(h["norm_diff"]) for h in hist],
                            "trust": {c: [float(h["trust"][c]) for h in hist] for c in clients},
                            "drift": {c: [float(h["drifts"][c]) for h in hist] for c in clients},
                        }
        ck.save()


def stage_drift(ck, preset, dataset):
    raw = ck.data["raw"].setdefault("drift", {})
    for level in DRIFT_LEVELS:
        for method in DRIFT_METHODS:
            for seed in preset["single_seeds"]:
                res = run_drift_experiment_single(
                    method=method, drift_level=level, seed=seed, dataset_name=dataset,
                    num_rounds=preset["rounds"], sample_limit=preset["sample_limit"],
                )
                raw.setdefault(str(level), {}).setdefault(method, []).append(res["metrics"])
        ck.save()


def stage_attack(ck, preset, dataset):
    raw = ck.data["raw"].setdefault("attack", {})
    for label, name, frac in attack_configs():
        for method in ATTACK_METHODS:
            for seed in preset["single_seeds"]:
                res = run_attack_experiment(
                    method=method, attack_name=name, seed=seed, dataset_name=dataset,
                    num_rounds=preset["rounds"], malicious_fraction=frac,
                    sample_limit=preset["sample_limit"],
                )
                raw.setdefault(label, {}).setdefault(method, []).append(res["metrics"])
        ck.save()


def stage_ablation(ck, preset, dataset):
    raw = ck.data["raw"].setdefault("ablation", {})
    for label, code in ABLATION_VARIANTS.items():
        raw[label] = [
            run_dwta_variant(dataset, code, s, preset["rounds"], preset["sample_limit"])
            for s in preset["single_seeds"]
        ]
        ck.save()


def stage_sensitivity(ck, preset, dataset):
    from experiments.sensitivity import run_sensitivity_sweep
    seed = preset["single_seeds"][0]
    ck.data["raw"]["sensitivity"] = run_sensitivity_sweep(
        grids=config.SENSITIVITY_GRIDS, dataset_name=dataset, seed=seed,
        num_rounds=preset["rounds"], sample_limit=preset["sample_limit"],
    )
    ck.save()


def stage_orchestration(ck):
    runs = [run_orchestration_simulation(seed=s) for s in range(config.ORCHESTRATION_RUNS)]
    ck.data["raw"]["orchestration"] = {
        "are": [float(r["are"]) for r in runs],
        "actions": {
            d: [ACTIONS.index(a) for a in runs[0]["history_actions"][d]]
            for d in runs[0]["history_actions"]
        },
        "action_names": ACTIONS,
        "windows": runs[0]["total_windows"],
    }
    ck.save()


# ----------------------------------------------------------------------------
# tables (derived from config settings and measured raw results)
# ----------------------------------------------------------------------------
def build_tables(ck):
    raw = ck.data["raw"]
    T = ck.add_table

    # ---- settings tables (from config.py) ----
    T("table_3", "Table 3. Action coefficients",
      ["Action", "Severity gain", "Propagation reduction", "Suitability", "Cost"],
      [[a, c["severity_gain"], c["propagation_reduction"], c["suitability"], c["cost"]]
       for a, c in config.ACTION_COEFFICIENTS.items()])

    T("table_4", "Table 4. Dataset configuration",
      ["Dataset", "Original records", "Retained records", "Input features"],
      [[n, d["original_records"], d["retained_records"], d["input_features"]]
       for n, d in config.DATASETS.items()])

    sched = config.DRIFT_SCHEDULE
    T("table_5", "Table 5. Synthetic drift schedule",
      ["Phase", "Rounds", "Drift type", "Clients (0-indexed)", "Onset"],
      [[k, f"{v['rounds'][0]}-{v['rounds'][1]}", v["drift_type"],
        ",".join(str(c) for c in v["clients"]) or "-", v.get("onset", "-")]
       for k, v in sched.items()])

    hp_names = [
        "NUM_CLIENTS", "NUM_ROUNDS", "LOCAL_EPOCHS", "BATCH_SIZE", "OPTIMIZER",
        "LEARNING_RATE", "MLP_HIDDEN_LAYERS", "MLP_ACTIVATION", "MLP_OUTPUT_UNITS",
        "RUN_SEEDS", "PARTITION_SEED", "DIRICHLET_ALPHA", "DRIFT_HISTOGRAM_BINS",
        "DRIFT_BIN_RANGE", "DRIFT_SMOOTHING_LAMBDA", "TRUST_MEMORY_BETA",
        "TRUST_EXPONENT_MU", "CONSISTENCY_EXPONENT_NU", "DRIFT_PENALTY_GAMMA",
        "VOLATILITY_PENALTY_TAU", "DRIFT_THRESHOLD_THETA_D", "HIGH_RELIABILITY_THETA_H",
        "LOW_RELIABILITY_THETA_L", "GATE_SUSPICIOUS_KAPPA_S", "GATE_ATTACKER_KAPPA_A",
        "ORCHESTRATION_LAMBDA_S", "ORCHESTRATION_LAMBDA_P", "ORCHESTRATION_LAMBDA_R",
        "ORCHESTRATION_LAMBDA_C", "ORCHESTRATION_ADAPTATION_ZETA", "MALICIOUS_CLIENT_SHARE",
    ]
    T("table_7", "Table 7. Experimental parameters",
      ["Parameter", "Value"], [[n, str(getattr(config, n))] for n in hp_names])

    T("table_8", "Table 8. Baseline hyperparameters",
      ["Method", "Hyperparameters"],
      [[m, ", ".join(f"{k}={v}" for k, v in h.items())]
       for m, h in config.BASELINE_CONFIG.items()])

    T("table_18", "Table 18. Threat states and mitigation actions",
      ["State", "Thresholds", "Action"],
      [[s, ", ".join(f"{k}={v}" for k, v in d.items() if k != "action") or "-", d["action"]]
       for s, d in config.THREAT_STATES.items()])

    T("table_s5", "Table S5. Adversarial attack configuration",
      ["Setting", "Value"],
      [[n, str(getattr(config, n))] for n in [
          "MALICIOUS_CLIENT_SHARE", "ATTACK_START_ROUND", "ATTACK_END_ROUND",
          "LABEL_POISON_FRACTION", "SIGN_FLIP_SCALE", "BACKDOOR_TRIGGER_FRACTION",
          "BACKDOOR_TARGET_CLASS", "BYZANTINE_NORM_SCALE", "ADAPTIVE_BYZANTINE_NORM_SCALE",
          "COLLUSION_SCALE", "COVERT_NORM_DEVIATION"]] +
      [[f"BACKDOOR_TRIGGERS[{d}]", str(t)] for d, t in config.BACKDOOR_TRIGGERS.items()])

    # ---- measured: detection ----
    det = raw.get("detection", {})
    for i, (ds, by_method) in enumerate(det.items()):
        rows, per_seed = [], []
        for m, seeds in by_method.items():
            runs = list(seeds.values())
            rows.append([m,
                         mean_sd(metric_list(runs, "accuracy")),
                         mean_sd(metric_list(runs, "f1")),
                         mean_sd(metric_list(runs, "fpr"))])
            for s, mt in seeds.items():
                per_seed.append([m, s, mt["accuracy"], mt["f1"], mt["fpr"]])
        T(f"table_detection_{ds}", f"Tables 9-11 / S6a. Detection performance - {ds} (mean ± SD over seeds)",
          ["Method", "Accuracy (%)", "F1 (%)", "FPR (%)"], rows)
        T(f"table_s7_{ds}", f"Tables S7-S9. Per-seed results - {ds}",
          ["Method", "Seed", "Accuracy (%)", "F1 (%)", "FPR (%)"], per_seed)

        # ---- Table 20: paired significance on F1 ----
        if "DWTA-FL" in by_method:
            seeds_order = sorted(by_method["DWTA-FL"].keys())
            paired = {m: [by_method[m][s]["f1"] for s in seeds_order]
                      for m in by_method if all(s in by_method[m] for s in seeds_order)}
            res = run_statistical_analysis(paired)
            T(f"table_20_{ds}", f"Table 20. Paired significance of DWTA-FL F1 vs baselines - {ds}",
              ["Baseline", "DWTA-FL F1", "Baseline F1", "t", "p (t-test)", "p (Wilcoxon)", "Cohen's d", "p<0.05"],
              [[r["Baseline"], r["DWTA-FL Mean"], r["Baseline Mean"], r["t-statistic"],
                r["t-test p-value"], r["Wilcoxon p-value"], r["Cohen's d"],
                "Yes" if r["Significant (p < 0.05)"] else "No"] for r in res])

    for ds, stats in raw.get("client_stats", {}).items():
        T(f"table_s3_{ds}", f"Table S3. Benign / malicious training samples per client - {ds}",
          ["Client", "Benign", "Malicious"],
          [[i + 1, s["benign"], s["malicious"]] for i, s in enumerate(stats)] +
          [["Total", sum(s["benign"] for s in stats), sum(s["malicious"] for s in stats)]])

    # ---- measured: drift ----
    if raw.get("drift"):
        rows = []
        for level, by_method in raw["drift"].items():
            for m, runs in by_method.items():
                rows.append([f"{int(round(float(level) * 100))}%", m,
                             float(np.mean(metric_list(runs, "accuracy"))),
                             float(np.mean(metric_list(runs, "precision"))),
                             float(np.mean(metric_list(runs, "recall"))),
                             float(np.mean(metric_list(runs, "f1"))),
                             float(np.mean(metric_list(runs, "fpr"))),
                             float(np.mean(metric_list(runs, "loss")))])
        T("table_12", "Table 12 / S4. Performance under synthetic drift",
          ["Drift", "Method", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "FPR (%)", "Global loss"], rows)

    # ---- measured: attack ----
    if raw.get("attack"):
        rows = []
        for label, by_method in raw["attack"].items():
            for m, runs in by_method.items():
                rows.append([label, m,
                             float(np.mean(metric_list(runs, "accuracy"))),
                             float(np.mean(metric_list(runs, "f1"))),
                             float(np.mean(metric_list(runs, "fpr"))),
                             float(np.mean(metric_list(runs, "arr")))])
        T("table_14_17", "Tables 14-17. Adversarial robustness",
          ["Attack", "Method", "Accuracy (%)", "F1 (%)", "FPR (%)", "ARR (%)"], rows)

    # ---- measured: ablation ----
    if raw.get("ablation"):
        T("table_23", "Table 23. Component ablation",
          ["Variant", "Accuracy (%)", "F1 (%)", "FPR (%)"],
          [[v, float(np.mean(metric_list(r, "accuracy"))), float(np.mean(metric_list(r, "f1"))),
            float(np.mean(metric_list(r, "fpr")))] for v, r in raw["ablation"].items()])

    # ---- measured: sensitivity ----
    if raw.get("sensitivity"):
        rows = []
        for p, pts in raw["sensitivity"].items():
            for pt in pts:
                rows.append([p, pt["value"], pt["metrics"]["accuracy"], pt["metrics"]["f1"], pt["metrics"]["fpr"]])
        T("table_sensitivity", "Figures 8-11 data. Hyperparameter sensitivity sweep",
          ["Parameter", "Value", "Accuracy (%)", "F1 (%)", "FPR (%)"], rows)

    # ---- measured: orchestration ----
    if raw.get("orchestration"):
        are = raw["orchestration"]["are"]
        T("table_19", "Table 19. Multi-cloud orchestration (Adaptive Response Efficiency)",
          ["Run", "ARE"], [[i + 1, a] for i, a in enumerate(are)] + [["Mean ± SD", mean_sd(are)]])


# ----------------------------------------------------------------------------
# figures (series of measured x/y values)
# ----------------------------------------------------------------------------
def build_figures(ck):
    raw = ck.data["raw"]
    F = ck.add_figure

    det = raw.get("detection", {})
    if det:
        datasets = list(det.keys())
        methods = list(next(iter(det.values())).keys())
        for metric, label in [("accuracy", "Accuracy (%)"), ("f1", "F1 (%)")]:
            F(f"fig_detection_{metric}", {
                "title": f"Detection {label} by method", "kind": "bar",
                "xlabel": "Dataset", "ylabel": label,
                "series": {m: {"x": datasets,
                               "y": [float(np.mean([r[metric] for r in det[d][m].values()])) if m in det[d] else 0.0
                                     for d in datasets]} for m in methods},
            })

    bh = raw.get("best_history")
    if bh:
        rounds = list(range(1, len(bh["norm_diff"]) + 1))
        F("fig_convergence", {
            "title": f"DWTA-FL global update norm per round ({bh['dataset']}, seed {bh['seed']})",
            "kind": "line", "xlabel": "Communication round", "ylabel": "||w(t+1) - w(t)||",
            "hlines": {"stability epsilon": config.STABILITY_EPSILON},
            "series": {"update norm": {"x": rounds, "y": bh["norm_diff"]}},
        })
        shown = sorted(bh["trust"].keys())[:6]
        F("fig_trust", {
            "title": "Recursive trust T per client (DWTA-FL)", "kind": "line",
            "xlabel": "Communication round", "ylabel": "Trust",
            "series": {f"client {c + 1}": {"x": rounds, "y": bh["trust"][c]} for c in shown},
        })
        F("fig_drift", {
            "title": "Smoothed drift per client (DWTA-FL)", "kind": "line",
            "xlabel": "Communication round", "ylabel": "Smoothed drift",
            "hlines": {"theta_D": config.DRIFT_THRESHOLD_THETA_D},
            "series": {f"client {c + 1}": {"x": rounds, "y": bh["drift"][c]} for c in shown},
        })

    drift = raw.get("drift", {})
    if drift:
        levels = sorted(drift.keys(), key=float)
        methods = list(drift[levels[0]].keys())
        F("fig_drift_resilience", {
            "title": "F1 under synthetic drift (Table 12)", "kind": "line",
            "xlabel": "Drift level (%)", "ylabel": "F1 (%)",
            "series": {m: {"x": [float(l) * 100 for l in levels],
                           "y": [float(np.mean([r["f1"] for r in drift[l][m]])) for l in levels]}
                       for m in methods},
        })

    atk = raw.get("attack", {})
    if atk:
        labels = list(atk.keys())
        methods = list(next(iter(atk.values())).keys())
        for metric, label in [("accuracy", "Accuracy (%)"), ("arr", "ARR (%)")]:
            F(f"fig_attack_{metric}", {
                "title": f"Attack robustness - {label}", "kind": "bar",
                "xlabel": "Attack", "ylabel": label,
                "series": {m: {"x": labels,
                               "y": [float(np.mean([r[metric] for r in atk[a][m]])) for a in labels]}
                           for m in methods},
            })

    abl = raw.get("ablation", {})
    if abl:
        variants = list(abl.keys())
        F("fig_ablation", {
            "title": "Component ablation (Table 23)", "kind": "bar",
            "xlabel": "Variant", "ylabel": "Percent",
            "series": {m: {"x": variants,
                           "y": [float(np.mean([r[k] for r in abl[v]])) for v in variants]}
                       for m, k in [("Accuracy", "accuracy"), ("F1", "f1")]},
        })

    sens = raw.get("sensitivity", {})
    for i, (p, pts) in enumerate(sens.items()):
        F(f"fig_sensitivity_{p}", {
            "title": f"Sensitivity to {p} (Figure {8 + i})", "kind": "line",
            "xlabel": p, "ylabel": "Percent",
            "series": {
                "Accuracy": {"x": [pt["value"] for pt in pts], "y": [pt["metrics"]["accuracy"] for pt in pts]},
                "F1": {"x": [pt["value"] for pt in pts], "y": [pt["metrics"]["f1"] for pt in pts]},
            },
        })

    orch = raw.get("orchestration")
    if orch:
        F("fig_orchestration", {
            "title": "Selected mitigation action per window", "kind": "line",
            "xlabel": "Decision window", "ylabel": "Action",
            "yticklabels": orch["action_names"],
            "series": {f"Cloud {chr(65 + d)}": {"x": list(range(1, len(a) + 1)), "y": a}
                       for d, a in orch["actions"].items()},
        })


# ----------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(config.RUN_PRESETS), default="quick")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma separated subset of {STAGES}")
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--fresh", action="store_true", help="ignore existing bestmodel.pt")
    ap.add_argument("--out", default=config.CHECKPOINT_PATH, help="checkpoint path")
    args = ap.parse_args()

    preset = config.RUN_PRESETS[args.preset]
    stages = [s for s in args.stages.split(",") if s]
    datasets = [d for d in args.datasets.split(",") if d]
    primary = datasets[0]

    if os.path.exists(args.out) and not args.fresh:
        ck = Checkpoint.load(args.out)
        print(f"Resuming from {args.out}")
    else:
        ck = Checkpoint(args.out)
    ck.data.setdefault("raw", {})
    ck.set_meta(preset=args.preset, preset_values=preset)

    if "detection" in stages:
        stage_detection(ck, preset, datasets)
    if "drift" in stages:
        stage_drift(ck, preset, primary)
    if "attack" in stages:
        stage_attack(ck, preset, primary)
    if "ablation" in stages:
        stage_ablation(ck, preset, primary)
    if "sensitivity" in stages:
        stage_sensitivity(ck, preset, primary)
    if "orchestration" in stages:
        stage_orchestration(ck)

    build_tables(ck)
    build_figures(ck)
    ck.save()
    print(f"\nSaved {len(ck.data['tables'])} tables, {len(ck.data['figures'])} figures "
          f"and the best model to {ck.path}")
    print("Now run: python test.py")


if __name__ == "__main__":
    main()
