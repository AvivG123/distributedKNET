"""Train GraphKalmanProcess runs and compare them with a DEKF baseline.

Run from the repository root, for example:

    python skills/train-compare-baseline/scripts/train_eval_compare.py --preset fast_debug --r-scales 0.25,0.5,1 --run-name smoke_compare

The script trains a separate model for every noise level and mismatch state.
It uses the existing experiment runner in-process, evaluates each best
checkpoint on fresh data, computes MSE in dB for Distributed KalmanNet and the
diffusion EKF baseline, writes a CSV summary, and saves a comparison figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.graphkalmanprocess_hparams import PRESETS
from experiments.run_graphkalmanprocess import _build_system, run_one_experiment
from experiments.run_graphkalmanprocess import parse_scalar, set_by_dotted_key
from utils.ClassicDistributedKalman import diffusion_extended_kalman_filter_parallel_edge
from utils.DistributedKalmanData import CreateGraph, GraphDataset
from utils.DistributedKalmanNet import GraphKalmanProcess


def parse_csv_scalars(value: str) -> list[Any]:
    return [parse_scalar(part.strip()) for part in value.split(",") if part.strip()]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def mse_db(pred: np.ndarray, true: np.ndarray) -> float:
    mse = float(np.mean((pred - true) ** 2))
    return 10.0 * math.log10(max(mse, 1e-12))


def build_model(cfg: dict) -> GraphKalmanProcess:
    node_num = int(cfg["graph"]["node_num"])
    _, f_model, _ = _build_system(cfg["system"], node_num=node_num)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return GraphKalmanProcess(
        f_model,
        signal_dim=int(model_cfg["signal_dim"]),
        edge_features_dim=int(model_cfg["edge_features_dim"]),
        node_kalman_dim=int(model_cfg["node_kalman_dim"]),
        edge_kalman_dim=int(model_cfg["edge_kalman_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        heads=int(model_cfg["heads"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(model_cfg["lr"]),
        r_array=float(data_cfg["r_scale"]),
        learn_edge_kalman=bool(model_cfg["learn_edge_kalman"]),
        x0_scale=float(model_cfg.get("x0_scale", data_cfg["x0_scale"])),
        consensus_layer=model_cfg.get("consensus_layer", "none"),
        position_only_loss=bool(model_cfg.get("position_only_loss", False)),
    ).to(torch.float)


def load_checkpoint(model: GraphKalmanProcess, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)


def evaluate_checkpoint_vs_dekf(
    *,
    cfg: dict,
    checkpoint_path: str,
    eval_sims: int,
    eval_time_steps: int,
    eval_seed: int,
) -> tuple[float, float]:
    seed = int(cfg["seed"])
    node_num = int(cfg["graph"]["node_num"])
    q = float(cfg["data"]["q"])
    r_scale = float(cfg["data"]["r_scale"])
    x0_scale = float(cfg["data"]["x0_scale"])
    n_expansions = int(cfg["data"]["n_expansions"])
    r_array = r_scale * np.ones(node_num)

    graph = CreateGraph(
        node_num=node_num,
        k_neighbors=int(cfg["graph"]["k_neighbors"]),
        rewrite_prob=float(cfg["graph"]["rewrite_prob"]),
        seed=seed,
    )
    f_true, f_model, h_system = _build_system(cfg["system"], node_num=node_num)
    dataset = GraphDataset(
        graph,
        f_true,
        h_system,
        q,
        r_array,
        monte_carlo_simulations=eval_sims,
        time_steps=eval_time_steps,
        n_expansions=n_expansions,
        x0=x0_scale,
        seed=eval_seed,
    )

    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    p0 = np.eye(int(cfg["model"]["signal_dim"]), dtype=np.float32)
    x0 = x0_scale * np.ones((int(cfg["model"]["signal_dim"]), 1), dtype=np.float32)
    true_all: list[np.ndarray] = []
    deep_all: list[np.ndarray] = []
    dekf_all: list[np.ndarray] = []

    with torch.no_grad():
        for graph_item in dataset:
            true_path = graph_item.y[..., 0].numpy()
            true_all.append(true_path)

            deep_pred = model(graph_item).detach().cpu().numpy()
            deep_mean = deep_pred.mean(axis=(0, 2, 4))
            deep_all.append(deep_mean)

            measurements = graph_item.x.numpy().transpose(0, 2, 1)
            adjacency = graph_item.adj_matrix.numpy()
            dekf_pred = diffusion_extended_kalman_filter_parallel_edge(
                measurements,
                f_model,
                h_system,
                r_array,
                q,
                p0,
                x0,
                adjacency,
                time_steps=eval_time_steps,
                node_num=node_num,
            )
            dekf_mean = dekf_pred.mean(axis=(1, 3))
            dekf_all.append(dekf_mean)

    true_np = np.stack(true_all, axis=0)
    deep_np = np.stack(deep_all, axis=0)
    dekf_np = np.stack(dekf_all, axis=0)
    return mse_db(dekf_np, true_np), mse_db(deep_np, true_np)


def plan_configs(args: argparse.Namespace) -> list[tuple[str, dict]]:
    base_cfg = deepcopy(PRESETS[args.preset])
    for override in args.override:
        if "=" not in override:
            raise ValueError(f"Invalid --override {override!r}; expected key=value")
        key, value = override.split("=", 1)
        set_by_dotted_key(base_cfg, key.strip(), parse_scalar(value))

    r_scales = parse_csv_scalars(args.r_scales) if args.r_scales else [base_cfg["data"]["r_scale"]]
    variants: list[tuple[str, dict]] = []
    for r_scale in r_scales:
        cfg = deepcopy(base_cfg)
        set_by_dotted_key(cfg, "data.r_scale", float(r_scale))
        variants.append((f"r={r_scale}", cfg))

    mismatch_mode = args.mismatch_mode
    if args.with_without_mismatch:
        mismatch_mode = "both"

    expanded: list[tuple[str, dict]] = []
    for name, cfg in variants:
        if mismatch_mode in {"both", "matched"}:
            cfg_nom = deepcopy(cfg)
            set_by_dotted_key(cfg_nom, "system.deg_true", 0.0)
            set_by_dotted_key(cfg_nom, "system.deg_model", 0.0)
            expanded.append((f"{name}__mismatch=off", cfg_nom))

        if mismatch_mode in {"both", "mismatched"}:
            cfg_mis = deepcopy(cfg)
            set_by_dotted_key(cfg_mis, "system.deg_true", float(args.mismatch_angle_deg))
            set_by_dotted_key(cfg_mis, "system.deg_model", 0.0)
            expanded.append((f"{name}__mismatch=on", cfg_mis))

        if mismatch_mode == "config":
            expanded.append((name, cfg))

    variants = expanded

    return variants


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted({str(row["mismatch"]) for row in rows})
    styles = {
        "matched": ("rv--", "bo--"),
        "mismatch": ("r^-.", "bs-."),
        "config": ("rv--", "bo--"),
    }
    plt.figure(figsize=(8, 6))
    for mismatch in groups:
        group_rows = sorted(
            [row for row in rows if str(row["mismatch"]) == mismatch],
            key=lambda row: float(row["err_db"]),
        )
        x_values = [float(row["err_db"]) for row in group_rows]
        dekf_values = [float(row["dekf_mse_db"]) for row in group_rows]
        deep_values = [float(row["deep_mse_db"]) for row in group_rows]
        suffix = "" if mismatch == "config" else f" ({mismatch})"
        dekf_style, deep_style = styles.get(mismatch, ("rv--", "bo--"))
        plt.plot(x_values, dekf_values, dekf_style, label=f"DEKF{suffix}")
        plt.plot(x_values, deep_values, deep_style, label=f"Distributed KalmanNet{suffix}")

    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$\frac{1}{r^2}$ [dB] ", fontsize=16)
    plt.ylabel("MSE [dB]", fontsize=16)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphKalmanProcess and plot DEKF comparison.")
    parser.add_argument("--preset", default="baseline", choices=sorted(PRESETS.keys()))
    parser.add_argument("--override", action="append", default=[], help="Override dotted key, e.g. model.hidden_dim=128")
    parser.add_argument("--r-scales", default=None, help="Comma-separated r scales, e.g. 0.25,0.5,1,2,4")
    parser.add_argument(
        "--mismatch-mode",
        choices=["both", "matched", "mismatched", "config"],
        default="both",
        help=(
            "Which mismatch states to train separately. Default 'both' trains matched "
            "and mismatched runs for every noise level. 'config' uses the preset/override "
            "system.deg_true and system.deg_model as-is."
        ),
    )
    parser.add_argument("--with-without-mismatch", action="store_true", help="Deprecated alias for --mismatch-mode both.")
    parser.add_argument("--mismatch-angle-deg", type=float, default=20.0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--root-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--eval-sims", type=int, default=128)
    parser.add_argument("--eval-time-steps", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--figure-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print planned configs without training.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    run_prefix = safe_name(args.run_name or f"{args.preset}_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    planned = plan_configs(args)

    if args.dry_run:
        print(f"Planned runs ({len(planned)}):")
        for name, cfg in planned:
            print(f"  - {run_prefix}__{safe_name(name)}")
            print(json.dumps(cfg, indent=2, sort_keys=True))
        return

    rows: list[dict[str, Any]] = []
    for idx, (name, cfg) in enumerate(planned):
        run_name = f"{run_prefix}__{idx:03d}__{safe_name(name)}"
        result = run_one_experiment(cfg, run_name=run_name, root_dir=root_dir)
        if not result.best_checkpoint:
            raise RuntimeError(f"Run {run_name} did not produce a best checkpoint.")

        eval_time_steps = int(args.eval_time_steps or cfg["data"]["time_steps"])
        dekf_mse_db, deep_mse_db = evaluate_checkpoint_vs_dekf(
            cfg=cfg,
            checkpoint_path=result.best_checkpoint,
            eval_sims=int(args.eval_sims),
            eval_time_steps=eval_time_steps,
            eval_seed=int(args.eval_seed),
        )
        r_scale = float(cfg["data"]["r_scale"])
        mismatch = "config"
        if args.with_without_mismatch or args.mismatch_mode != "config":
            mismatch = "mismatch" if float(cfg["system"]["deg_true"]) != float(cfg["system"]["deg_model"]) else "matched"

        rows.append(
            {
                "run_name": run_name,
                "variant": name,
                "mismatch": mismatch,
                "r_scale": r_scale,
                "err_db": 10.0 * math.log10(1.0 / (r_scale**2)),
                "best_val_loss": result.best_val,
                "best_checkpoint": result.best_checkpoint,
                "log_dir": result.log_dir,
                "eval_sims": int(args.eval_sims),
                "eval_time_steps": eval_time_steps,
                "dekf_mse_db": dekf_mse_db,
                "deep_mse_db": deep_mse_db,
            }
        )

    summary_path = Path(args.summary_path) if args.summary_path else root_dir / "experiments" / "results" / f"{run_prefix}_comparison.csv"
    figure_path = Path(args.figure_path) if args.figure_path else root_dir / "figures" / f"{run_prefix}_comparison.png"
    write_csv(summary_path, rows)
    plot_results(figure_path, rows, title=f"{args.preset} DEKF vs Distributed KalmanNet")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
