from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from experiments.graphkalmanprocess_hparams import LOCALIZATION_BASELINE, PRESETS, SWEEPS
from utils.reproducibility import seed_everything
from utils.DistributedKalmanData import (
    CreateGraph,
    FSystem,
    FSystemLinear,
    GraphDataset,
    HSystem,
    HSystemLinear,
)
from utils.DistributedKalmanNet import GraphKalmanProcess, loss_function
from utils.LocalizationScenario import (
    dkn_model_path,
    gnn_rnn_model_path,
    normalize_localization_config,
    normalize_localization_train_models,
    run_localization_experiment,
)


def parse_scalar(value: str) -> Any:
    lower = value.strip().lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            f = float(value)
            if math.isfinite(f):
                return f
        i = int(value)
        return i
    except ValueError:
        return value


def set_by_dotted_key(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def flatten_dict(cfg: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


@dataclass(frozen=True)
class RunResult:
    run_name: str
    best_val: float | None
    best_checkpoint: str | None
    log_dir: str
    eval_loss: float | None = None


def _build_system(system_cfg: dict, node_num: int):
    kind = system_cfg["kind"]
    if kind not in {"linear", "nonlinear"}:
        raise ValueError(f"system.kind must be 'linear' or 'nonlinear', got {kind!r}")

    deg_true = np.deg2rad(float(system_cfg["deg_true"]))
    deg_model = np.deg2rad(float(system_cfg["deg_model"]))

    if kind == "linear":
        h_system = HSystemLinear(node_num=node_num)
        f_true = FSystemLinear(deg=deg_true)
        f_model = FSystemLinear(deg=deg_model)
    else:
        h_system = HSystem(node_num=node_num, alpha=float(system_cfg.get("h_alpha", 0.0)))
        f_true = FSystem(deg=deg_true)
        f_model = FSystem(deg=deg_model)

    return f_true, f_model, h_system


def _build_graph_dataloaders(
    *,
    graph,
    f_true,
    h_system,
    q: float,
    r_array,
    train_sims: int,
    val_sims: int,
    time_steps: int,
    n_expansions: int,
    x0_scale: float,
    seed: int,
    batch_size: int,
):
    train_ds = GraphDataset(
        graph,
        f_true,
        h_system,
        q,
        r_array,
        monte_carlo_simulations=train_sims,
        time_steps=time_steps,
        n_expansions=n_expansions,
        x0=x0_scale,
        seed=seed,
    )
    val_ds = GraphDataset(
        graph,
        f_true,
        h_system,
        q,
        r_array,
        monte_carlo_simulations=val_sims,
        time_steps=time_steps,
        n_expansions=n_expansions,
        x0=x0_scale,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader


def _curriculum_schedule(curriculum_cfg: dict, base_time_steps: int) -> list[int]:
    if not bool(curriculum_cfg.get("enabled", False)):
        return [base_time_steps]

    start = int(curriculum_cfg.get("start_time_steps", 10))
    step = int(curriculum_cfg.get("step_time_steps", 10))
    maximum = int(curriculum_cfg.get("max_time_steps", base_time_steps))
    if step <= 0:
        raise ValueError("curriculum.step_time_steps must be positive")
    if maximum < start:
        raise ValueError("curriculum.max_time_steps must be >= curriculum.start_time_steps")

    schedule = list(range(start, maximum + 1, step))
    if base_time_steps not in schedule:
        schedule.append(base_time_steps)
    return sorted(set(schedule))


def run_one_experiment(cfg: dict, *, run_name: str, root_dir: Path) -> RunResult:
    print(f"\n=== Run: {run_name} ===")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    seed_everything(int(cfg["seed"]))
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    node_num = int(cfg["graph"]["node_num"])
    time_steps = int(cfg["data"]["time_steps"])
    batch_size = int(cfg["data"]["batch_size"])
    q = float(cfg["data"]["q"])
    r_scale = float(cfg["data"]["r_scale"])
    x0_scale = float(cfg["data"]["x0_scale"])
    n_expansions = int(cfg["data"]["n_expansions"])

    graph = CreateGraph(
        node_num=node_num,
        k_neighbors=int(cfg["graph"]["k_neighbors"]),
        rewrite_prob=float(cfg["graph"]["rewrite_prob"]),
        seed=int(cfg["seed"]),
    )

    r_array = r_scale * np.ones(node_num)
    f_true, f_model, h_system = _build_system(cfg["system"], node_num=node_num)

    model_cfg = cfg["model"]
    model = GraphKalmanProcess(
        f_model,
        signal_dim=int(model_cfg["signal_dim"]),
        edge_features_dim=int(model_cfg["edge_features_dim"]),
        node_kalman_dim=int(model_cfg["node_kalman_dim"]),
        edge_kalman_dim=int(model_cfg["edge_kalman_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        heads=int(model_cfg["heads"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(model_cfg["lr"]),
        r_array=r_scale,
        learn_edge_kalman=bool(model_cfg["learn_edge_kalman"]),
        x0_scale=float(model_cfg.get("x0_scale", x0_scale)),
        consensus_layer=model_cfg.get("consensus_layer", "none"),
    ).to(torch.float)

    curriculum_cfg = cfg.get("curriculum", {})
    schedule = _curriculum_schedule(curriculum_cfg, base_time_steps=time_steps)
    epochs_per_stage = int(curriculum_cfg.get("epochs_per_stage", cfg["trainer"]["max_epochs"]))
    stage_max_epochs = epochs_per_stage if bool(curriculum_cfg.get("enabled", False)) else int(cfg["trainer"]["max_epochs"])

    log_root = root_dir / "lightning_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    monitor_key = "val_loss"
    best_val: float | None = None
    best_path: str | None = None
    last_log_dir: str | None = None

    for stage_idx, stage_time_steps in enumerate(schedule):
        train_loader, val_loader = _build_graph_dataloaders(
            graph=graph,
            f_true=f_true,
            h_system=h_system,
            q=q,
            r_array=r_array,
            train_sims=int(cfg["data"]["train_sims"]),
            val_sims=int(cfg["data"]["val_sims"]),
            time_steps=stage_time_steps,
            n_expansions=n_expansions,
            x0_scale=x0_scale,
            seed=int(cfg["seed"]),
            batch_size=batch_size,
        )

        stage_run_name = run_name if len(schedule) == 1 else f"{run_name}__ts{stage_time_steps}"
        logger = pl.loggers.CSVLogger(save_dir=str(log_root), name="graphkalmanprocess", version=stage_run_name)
        Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor=monitor_key,
            patience=int(cfg["trainer"]["early_stop_patience"]),
            verbose=True,
            mode="min",
            min_delta=float(cfg["trainer"]["early_stop_min_delta"]),
        )
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            filename="best-{epoch}",
        )
        trainer = pl.Trainer(
            max_epochs=stage_max_epochs,
            accelerator="auto",
            log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
            callbacks=[early_stopping, checkpoint],
            gradient_clip_val=float(cfg["trainer"].get("gradient_clip_val", 0.0)),
            logger=logger,
            default_root_dir=str(root_dir),
        )
        trainer.fit(model, train_loader, val_loader)
        stage_best_val = checkpoint.best_model_score.item() if checkpoint.best_model_score is not None else None
        stage_best_path = checkpoint.best_model_path or None
        if stage_best_val is not None and (best_val is None or stage_best_val < best_val):
            best_val = stage_best_val
            best_path = stage_best_path
        last_log_dir = str(Path(logger.log_dir))

    eval_loss = None
    eval_cfg = cfg.get("eval")
    if eval_cfg and bool(eval_cfg.get("enabled", False)):
        if best_path:
            ckpt = torch.load(best_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)

        eval_loss = evaluate_on_graph(
            model=model,
            cfg={**cfg, "data": {**cfg["data"], "time_steps": schedule[-1]}},
            eval_cfg=eval_cfg,
        )

    return RunResult(
        run_name=run_name,
        best_val=best_val,
        best_checkpoint=best_path,
        log_dir=last_log_dir or str(log_root),
        eval_loss=eval_loss,
    )


def _iter_grid_runs(base_cfg: dict, grid: dict[str, list[Any]]):
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    for values in itertools.product(*value_lists):
        cfg = deepcopy(base_cfg)
        name_parts = []
        for k, v in zip(keys, values, strict=True):
            set_by_dotted_key(cfg, k, v)
            safe_v = str(v).replace("/", "_")
            name_parts.append(f"{k.split('.')[-1]}={safe_v}")
        yield cfg, "__".join(name_parts)


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_grid_kv(items: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --grid item {item!r}; expected key=v1,v2,...")
        key, values = item.split("=", 1)
        values_list = [parse_scalar(v) for v in values.split(",") if v != ""]
        grid[key.strip()] = values_list
    return grid


def _parse_csv_scalars(values: str | None) -> list[Any] | None:
    if values is None:
        return None
    values = values.strip()
    if not values:
        return None
    return [parse_scalar(v.strip()) for v in values.split(",") if v.strip() != ""]


def _expand_noise_and_mismatch(planned: list[tuple[dict, str]], *, args: argparse.Namespace) -> list[tuple[dict, str]]:
    r_scales = _parse_csv_scalars(args.r_scales)
    q_values = _parse_csv_scalars(args.q_values)

    out: list[tuple[dict, str]] = []
    for base_cfg, base_name in planned:
        variants: list[tuple[dict, str]] = [(base_cfg, base_name)]

        if r_scales is not None:
            next_variants: list[tuple[dict, str]] = []
            for cfg, name in variants:
                for r in r_scales:
                    cfg2 = deepcopy(cfg)
                    set_by_dotted_key(cfg2, "data.r_scale", float(r))
                    next_variants.append((cfg2, f"{name}__r={r}"))
            variants = next_variants

        if q_values is not None:
            next_variants = []
            for cfg, name in variants:
                for q in q_values:
                    cfg2 = deepcopy(cfg)
                    set_by_dotted_key(cfg2, "data.q", float(q))
                    next_variants.append((cfg2, f"{name}__q={q}"))
            variants = next_variants

        if args.with_without_mismatch:
            next_variants = []
            for cfg, name in variants:
                cfg_nom = deepcopy(cfg)
                set_by_dotted_key(cfg_nom, "system.deg_true", 0.0)
                set_by_dotted_key(cfg_nom, "system.deg_model", 0.0)
                next_variants.append((cfg_nom, f"{name}__mismatch=off"))

                cfg_mis = deepcopy(cfg)
                set_by_dotted_key(cfg_mis, "system.deg_true", float(args.mismatch_angle_deg))
                set_by_dotted_key(cfg_mis, "system.deg_model", 0.0)
                next_variants.append((cfg_mis, f"{name}__mismatch=on"))
            variants = next_variants

        out.extend(variants)

    return out


def evaluate_on_graph(*, model: GraphKalmanProcess, cfg: dict, eval_cfg: dict) -> float:
    model.eval()

    node_num = int(cfg["graph"]["node_num"])
    q = float(cfg["data"]["q"])
    r_scale = float(cfg["data"]["r_scale"])
    x0_scale = float(cfg["data"]["x0_scale"])
    n_expansions = int(cfg["data"]["n_expansions"])

    graph = CreateGraph(
        node_num=node_num,
        k_neighbors=int(cfg["graph"]["k_neighbors"]),
        rewrite_prob=float(cfg["graph"]["rewrite_prob"]),
        seed=int(cfg["seed"]),
    )
    r_array = r_scale * np.ones(node_num)
    f_true, _, h_system = _build_system(cfg["system"], node_num=node_num)

    eval_ds = GraphDataset(
        graph,
        f_true,
        h_system,
        q,
        r_array,
        monte_carlo_simulations=int(eval_cfg.get("sims", cfg["data"]["val_sims"])),
        time_steps=int(eval_cfg.get("time_steps", cfg["data"]["time_steps"])),
        n_expansions=n_expansions,
        x0=x0_scale,
        seed=int(cfg["seed"]),
    )
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        batch_size=int(eval_cfg.get("batch_size", cfg["data"]["batch_size"])),
    )

    device = model.device
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            x_true = batch.y.reshape(batch.num_graphs, -1, model.signal_dim, 1)
            x_pred = model(batch).to(device=x_true.device)
            loss = loss_function(x_pred, x_true)
            total += float(loss.item()) * int(batch.num_graphs)
            count += int(batch.num_graphs)
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/sweep GraphKalmanProcess with simple hyperparam grids.")
    parser.add_argument("--localization", action="store_true", help="Run the localization DKN/GNN-RNN workflow.")
    parser.add_argument("--preset", default="baseline", choices=sorted(PRESETS.keys()))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override dotted key, e.g. model.hidden_dim=128 (repeatable).",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Grid item key=v1,v2,... (repeatable) for cartesian sweep.",
    )
    parser.add_argument("--sweep", choices=sorted(SWEEPS.keys()), help="Run a named sweep from graphkalmanprocess_hparams.py.")
    parser.add_argument("--run-name", default=None, help="Single-run name; defaults to timestamp.")
    parser.add_argument("--root-dir", default=".", help="Project root directory (default: current).")
    parser.add_argument("--description", default=None, help="Experiment description for localization runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    parser.add_argument("--r-scales", default=None, help="Comma-separated sweep over data.r_scale (e.g. 0.5,1,2).")
    parser.add_argument("--q-values", default=None, help="Comma-separated sweep over data.q (e.g. 0.1,1,10).")
    parser.add_argument(
        "--with-without-mismatch",
        action="store_true",
        help="Duplicate each planned run with mismatch off/on (system.deg_true=0 vs mismatch-angle; system.deg_model=0).",
    )
    parser.add_argument("--mismatch-angle-deg", type=float, default=20.0, help="Mismatch angle in degrees for mismatch-on runs.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the best checkpoint on a fresh graph dataset after training.")
    parser.add_argument("--eval-sims", type=int, default=None, help="Evaluation monte-carlo simulations (default: data.val_sims).")
    parser.add_argument("--eval-time-steps", type=int, default=None, help="Evaluation time steps (default: data.time_steps).")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size (default: data.batch_size).")
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the final config JSON (single-run) and exit.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()

    if args.localization:
        config_val = deepcopy(LOCALIZATION_BASELINE)
        for ov in args.override:
            if "=" not in ov:
                raise ValueError(f"Invalid --override {ov!r}; expected key=value")
            k, v = ov.split("=", 1)
            set_by_dotted_key(config_val, k.strip(), parse_scalar(v))
        config_val = normalize_localization_config(config_val)

        if args.dump_config:
            print(json.dumps(config_val, indent=2, sort_keys=True))
            return

        if args.dry_run:
            dt_values = config_val.get("dt_mismatch_values", [1.0]) if config_val.get("use_dt_mismatch", False) else [1.0]
            print("Planned localization runs:")
            for r_noise in config_val["measurement_noise_values"]:
                if "gnn_rnn" in normalize_localization_train_models(config_val):
                    path = gnn_rnn_model_path(Path("<experiment>"), r_noise, config=config_val, for_save=True)
                    print(f"  - {path.parent.name}/{path.name}")
                if "dkn" in normalize_localization_train_models(config_val):
                    for dt_mismatch in dt_values:
                        path = dkn_model_path(
                            Path("<experiment>"),
                            r_noise,
                            use_dt_mismatch=config_val.get("use_dt_mismatch", False),
                            dt_ratio=dt_mismatch,
                            config=config_val,
                            for_save=True,
                        )
                        print(f"  - {path.parent.name}/{path.name}")
            return

        run_localization_experiment(config_val, root_dir=root_dir, description=args.description)
        return

    base_cfg = deepcopy(PRESETS[args.preset])
    for ov in args.override:
        if "=" not in ov:
            raise ValueError(f"Invalid --override {ov!r}; expected key=value")
        k, v = ov.split("=", 1)
        set_by_dotted_key(base_cfg, k.strip(), parse_scalar(v))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dump_config:
        print(json.dumps(base_cfg, indent=2, sort_keys=True))
        return

    if args.eval:
        base_cfg.setdefault("eval", {})
        base_cfg["eval"]["enabled"] = True
        if args.eval_sims is not None:
            base_cfg["eval"]["sims"] = int(args.eval_sims)
        if args.eval_time_steps is not None:
            base_cfg["eval"]["time_steps"] = int(args.eval_time_steps)
        if args.eval_batch_size is not None:
            base_cfg["eval"]["batch_size"] = int(args.eval_batch_size)

    if args.sweep:
        grid = SWEEPS[args.sweep]
        run_prefix = f"{args.preset}__{args.sweep}__{timestamp}"
        planned = list(_iter_grid_runs(base_cfg, grid))
    elif args.grid:
        grid = _parse_grid_kv(args.grid)
        run_prefix = f"{args.preset}__grid__{timestamp}"
        planned = list(_iter_grid_runs(base_cfg, grid))
    else:
        run_name = args.run_name or f"{args.preset}__{timestamp}"
        planned = [(base_cfg, run_name)]
        run_prefix = run_name

    planned = _expand_noise_and_mismatch(planned, args=args)

    if args.dry_run:
        print(f"Planned runs ({len(planned)}):")
        for _, name in planned:
            print(f"  - {name}")
        return

    rows: list[dict[str, Any]] = []
    for i, (cfg, suffix) in enumerate(planned):
        run_name = suffix if args.sweep or args.grid else planned[0][1]
        if args.sweep or args.grid:
            run_name = f"{run_prefix}__{i:03d}__{suffix}"

        result = run_one_experiment(cfg, run_name=run_name, root_dir=root_dir)
        flat = flatten_dict(cfg)
        flat.update(
            {
                "run_name": result.run_name,
                "best_val_loss": result.best_val,
                "best_checkpoint": result.best_checkpoint,
                "log_dir": result.log_dir,
                "eval_loss": result.eval_loss,
            }
        )
        rows.append(flat)

    results_path = root_dir / "experiments" / "results" / f"{run_prefix}.csv"
    _write_results_csv(results_path, rows)
    print(f"Wrote results: {results_path}")


if __name__ == "__main__":
    main()
