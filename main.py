"""
Main Execution Entry Point for DWTA-FL Project.

Supports running all experiments reported in the paper:
  - detection:     Tables 9-11 (Detection performance on 5 seeds)
  - drift:         Table 12 (Synthetic drift resilience 0%-30%)
  - attack:        Tables 14-17 (Adversarial attack robustness)
  - ablation:      Table 23 (Component ablation study)
  - orchestration: Table 19 (Multi-cloud 3-domain orchestration)
  - statistical:   Table 20 (Statistical significance tests)
  - all:           Run full experimental suite

Usage:
  python main.py --experiment statistical
  python main.py --experiment detection --dataset CSE-CIC-IDS2018 --sample_limit 5000 --rounds 10
  python main.py --experiment drift --sample_limit 5000 --rounds 10
  python main.py --experiment attack --attack byzantine --sample_limit 5000 --rounds 10
  python main.py --experiment orchestration
"""

import argparse
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from experiments.detection import run_detection_benchmark
from experiments.drift_eval import run_drift_benchmark
from experiments.attack_eval import run_attack_benchmark
from experiments.ablation import run_ablation_study
from experiments.orchestration import run_orchestration_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="DWTA-FL: Drift-Weighted Trust-Aware Federated Learning"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="orchestration",
        choices=["detection", "drift", "attack", "ablation", "orchestration", "all"],
        help="Experiment to run (statistics/sensitivity/all tables: use train.py then test.py)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CSE-CIC-IDS2018",
        choices=["CSE-CIC-IDS2018", "UNSW-NB15", "TON_IoT"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=config.NUM_ROUNDS,
        help="Number of communication rounds (default: 100)",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Sample limit for fast testing / validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific seed to run (defaults to config.RUN_SEEDS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("DWTA-FL: Drift-Weighted Trust-Aware Federated Learning")
    print(f"Experiment:   {args.experiment}")
    print(f"Dataset:      {args.dataset}")
    print(f"Rounds:       {args.rounds}")
    print(f"Sample Limit: {args.sample_limit}")
    print("=" * 70)

    seeds = [args.seed] if args.seed is not None else config.RUN_SEEDS

    if args.experiment in ["orchestration", "all"]:
        print("\n>>> Running Multi-Cloud Orchestration Benchmark (Table 19)...")
        run_orchestration_benchmark(num_runs=5)

    if args.experiment in ["ablation", "all"]:
        print("\n>>> Running Component Ablation Study (Table 23)...")
        run_ablation_study(
            dataset_name=args.dataset,
            seeds=[seeds[0]],
            num_rounds=min(args.rounds, 20),
            sample_limit=args.sample_limit or 2000,
        )

    if args.experiment in ["drift", "all"]:
        print("\n>>> Running Drift Resilience Evaluation (Table 12)...")
        run_drift_benchmark(
            methods=["FedAvg", "DWTA-FL"],
            seeds=[seeds[0]],
            num_rounds=min(args.rounds, 20),
            sample_limit=args.sample_limit or 2000,
        )

    if args.experiment in ["attack", "all"]:
        print("\n>>> Running Attack Robustness Evaluation (Tables 14-17)...")
        run_attack_benchmark(
            attacks=["byzantine", "label_poisoning"],
            methods=["FedAvg", "DWTA-FL"],
            seeds=[seeds[0]],
            num_rounds=min(args.rounds, 20),
            sample_limit=args.sample_limit or 2000,
        )

    if args.experiment in ["detection", "all"]:
        print(f"\n>>> Running Detection Performance Benchmark ({args.dataset})...")
        run_detection_benchmark(
            dataset_name=args.dataset,
            methods=["DWTA-FL", "FL-IDS++", "Fed-ANIDS"],
            seeds=[seeds[0]],
            num_rounds=min(args.rounds, 20),
            sample_limit=args.sample_limit or 2000,
        )

    print("\n[OK] Execution finished successfully.")


if __name__ == "__main__":
    main()
