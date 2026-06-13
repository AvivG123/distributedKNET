"""
Config-Driven DKN / GNN-RNN Training

Trains selected neural models on the distance/angle constant-velocity scenario
using a single config_val dictionary loaded from JSON.  Saves model weights,
trajectory previews, learning curves, and a matching JSON config file fornum
later loading.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "utils").exists() else SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+/Blackwell

from utils.DistributedKalmanData import GraphDataset
from utils.DistributedKalmanNet import GraphKalmanProcess
from utils.BaselineModels import GnnRnnLightning
from utils.ConstantVelocityScenario import (
    ConstantVelocityModel,
    DistanceAngleObservation,
    create_distance_based_graph,
    generate_node_positions,
    get_trainer_accelerator,
    plot_graph,
    load_config,
    next_experiment_dir,
    plot_generated_trajectories,
    save_experiment_config,
    seed_everything,
)


def normalize_train_models(config_val):
    aliases = {
        "dkn": "dkn",
        "gnn_rnn": "gnn_rnn",
        "gnn-rnn": "gnn_rnn",
        "gnnrnn": "gnn_rnn",
    }
    raw_models = config_val.get("train_models", ["dkn"])
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if any(str(model).lower() == "all" for model in raw_models):
        return {"dkn", "gnn_rnn"}

    train_models = set()
    for model in raw_models:
        key = str(model).lower()
        if key not in aliases:
            raise ValueError(f"Unknown train model '{model}'. Expected one of: {sorted(aliases)} or 'all'.")
        train_models.add(aliases[key])
    if "dkn" in train_models:
        train_models.add("gnn_rnn")
    return train_models


def build_trainer(config_val, trainer_accelerator):
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss:_epoch",
        patience=5,
        verbose=True,
        mode="min",
        min_delta=0.001,
    )
    return pl.Trainer(
        max_epochs=config_val["max_epochs"],
        accelerator=trainer_accelerator,
        devices=1,
        **({"precision": "bf16-mixed"} if trainer_accelerator == "gpu" else {}),
        logger=False,
        callbacks=[early_stopping],
        gradient_clip_val=1,
    )


def plot_prediction_sample(graph, prediction, node_positions, title, label, save_path=None):
    """Plot one validation trajectory against a neural model prediction."""
    x_true = graph.y[:, 0, 0].cpu().numpy()
    y_true = graph.y[:, 2, 0].cpu().numpy()
    x_pred = prediction[:, 0]
    y_pred = prediction[:, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_true, y_true, "b-", linewidth=2, label="True trajectory")
    ax.plot(x_pred, y_pred, "m--", linewidth=2, label=f"{label} prediction")
    ax.scatter(node_positions[:, 0], node_positions[:, 1], c="black", marker="s", s=40, label="Nodes")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved prediction plot: {save_path}")
    plt.close(fig)


def main():
    # ── Configuration ─────────────────────────────────────────────────────
    config_path = REPO_ROOT / "configs" / "const_vel_scenario_configured.json"
    config_val = load_config(config_path)
    seed_everything(config_val["seed"])

    num_nodes = config_val["num_nodes"]
    state_dimension = config_val["state_dimension"]
    time_steps = config_val["train_time_steps"]
    time_delta = config_val["time_delta"]
    process_noise_std = config_val["process_noise_std"]
    measurement_noise_values = config_val["measurement_noise_values"]
    x0 = np.array(config_val["x0"], dtype=float).reshape(state_dimension, 1)
    if "node_positions" in config_val and config_val["node_positions"] is not None:
        node_positions = np.array(config_val["node_positions"], dtype=float)
    else:
        node_positions = generate_node_positions(num_nodes, seed=config_val["graph_seed"])
    trainer_accelerator = get_trainer_accelerator()
    use_dt_mismatch = config_val.get("use_dt_mismatch", False)
    train_models = normalize_train_models(config_val)
    save_root = REPO_ROOT / config_val["save_root"]
    experiment_dir = next_experiment_dir(save_root)
    description = input("Experiment description: ").strip()
    config_val["description"] = description

    print(f"Loaded config from: {config_path}")
    print(pd.Series(config_val))
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Trainer accelerator: {trainer_accelerator}")
    print(f"Training models: {', '.join(sorted(train_models))}")
    print(f"Initial state vector x0:\n{x0}")

    # ── Build scenario objects ────────────────────────────────────────────
    # f_system_data always uses the true time_delta for measurement generation
    f_system_data = ConstantVelocityModel(time_delta)
    h_system = DistanceAngleObservation(node_positions)
    adjacency_matrix = create_distance_based_graph(
        node_positions,
        k_neighbors=config_val["k_neighbors"],
        seed=config_val["graph_seed"],
    )
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(int)

    dt_mismatch_values = config_val.get("dt_mismatch_values", [1.0]) if use_dt_mismatch else [1.0]

    model_dir = experiment_dir / "DKN"
    gnn_rnn_model_dir = experiment_dir / "GNN_RNN"
    plot_dir = experiment_dir / "plots"
    model_paths = [plot_dir]
    if "dkn" in train_models:
        model_paths.append(model_dir)
    if "gnn_rnn" in train_models:
        model_paths.append(gnn_rnn_model_dir)
    for path in model_paths:
        path.mkdir(parents=True, exist_ok=True)

    save_experiment_config(
        experiment_dir,
        full_config=config_val,
        node_positions=node_positions,
    )
    plot_graph(adjacency_matrix, node_positions, save_path=plot_dir / "sensor_graph.png")

    print("Training node positions:")
    print(node_positions)
    if use_dt_mismatch:
        print(f"dt mismatch enabled — model dt multipliers: {dt_mismatch_values}")

    # ── Train one model per measurement-noise level (× dt mismatch for DKN) ──
    for r_noise in measurement_noise_values:
        r_array = r_noise * np.ones(num_nodes)

        # Data generation always uses true time_delta
        train_dataset = GraphDataset(
            adjacency_matrix,
            f_system_data,
            h_system,
            process_noise_std,
            r_array,
            monte_carlo_simulations=config_val["train_sims"],
            time_steps=time_steps,
            n_expansions=1,
            x0=x0,
            state_dim=state_dimension,
        )
        val_dataset = GraphDataset(
            adjacency_matrix,
            f_system_data,
            h_system,
            process_noise_std,
            r_array,
            monte_carlo_simulations=config_val["val_sims"],
            time_steps=time_steps,
            n_expansions=1,
            x0=x0,
            state_dim=state_dimension,
        )

        # ── Preview sample trajectories (once per noise level) ───────────
        n_preview = config_val.get("preview_trajectories", 4)
        sample_trajectories = [
            train_dataset[idx].y.cpu()
            for idx in range(min(n_preview, len(train_dataset)))
        ]
        plot_generated_trajectories(
            node_positions,
            node_types,
            sample_trajectories,
            max_trajectories=n_preview,
            title_prefix=f"Training trajectories (r={r_noise})",
            save_path=plot_dir / f"r={r_noise}_trajectories.png",
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config_val["batch_size"],
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config_val["batch_size"],
            num_workers=0,
            pin_memory=False,
        )

        if "gnn_rnn" in train_models:
            run_name = f"r={r_noise}"
            print(f"\nTraining GNN-RNN baseline: r={r_noise}")

            gnn_rnn_process = GnnRnnLightning(
                input_dim=1,
                type_num=2,
                hidden_dim=config_val.get("gnn_rnn_hidden_dim", config_val["hidden_dim"]),
                output_dim=state_dimension,
                lr=config_val.get("gnn_rnn_learning_rate", config_val["learning_rate"]),
            ).to(torch.float32)

            trainer = build_trainer(config_val, trainer_accelerator)
            trainer.fit(gnn_rnn_process, train_loader, val_loader)

            model_path = gnn_rnn_model_dir / f"{run_name}.pth"
            torch.save(gnn_rnn_process.state_dict(), model_path)
            print(f"Saved model: {model_path}")

            sample_device = next(gnn_rnn_process.parameters()).device
            sample_graph = val_dataset[0].to(sample_device)
            sample_node_types = h_system.node_classification.to(device=sample_device, dtype=torch.long)
            with torch.no_grad():
                sample_prediction = (
                    gnn_rnn_process(
                        sample_graph.x,
                        sample_node_types,
                        sample_graph.edge_index,
                    )
                    .cpu()
                    .numpy()
                    .mean(axis=0)
                )
            plot_prediction_sample(
                sample_graph,
                sample_prediction,
                node_positions,
                title=f"GNN-RNN validation example (r={r_noise})",
                label="GNN-RNN",
                save_path=plot_dir / f"gnn_rnn_{run_name}_prediction.png",
            )

        if "dkn" not in train_models:
            continue

        for dt_mismatch in dt_mismatch_values:
            run_name = f"r={r_noise}_dtx{dt_mismatch}" if use_dt_mismatch else f"r={r_noise}"
            print(f"\nTraining DKN model: r={r_noise}" + (f", dt_mismatch={dt_mismatch}" if use_dt_mismatch else ""))

            # Model uses mismatched dt; with no mismatch this is just time_delta
            f_system_model = ConstantVelocityModel(time_delta * dt_mismatch)

            kalman_process = GraphKalmanProcess(
                f_system_model,
                signal_dim=state_dimension,
                edge_features_dim=1,
                node_kalman_dim=state_dimension ** 2,
                edge_kalman_dim=2,
                hidden_dim=config_val["hidden_dim"],
                lr=config_val["learning_rate"],
                r_array=r_noise,
                learn_edge_kalman=config_val["learn_edge_kalman"],
                x0_scale=x0,
            ).to(torch.float32)

            trainer = build_trainer(config_val, trainer_accelerator)
            trainer.fit(kalman_process, train_loader, val_loader)

            # ── Save model ────────────────────────────────────────────────
            model_path = model_dir / f"{run_name}.pth"
            torch.save(kalman_process.state_dict(), model_path)
            print(f"Saved model: {model_path}")

            # ── Post-training plots ───────────────────────────────────────
            sample_device = next(kalman_process.parameters()).device
            sample_graph = val_dataset[0].to(sample_device)
            with torch.no_grad():
                sample_prediction = kalman_process(sample_graph).squeeze().cpu().numpy().mean(axis=1)
            title = f"DKN validation example (r={r_noise}" + (f", dt×{dt_mismatch})" if use_dt_mismatch else ")")
            plot_prediction_sample(
                sample_graph,
                sample_prediction,
                node_positions,
                title=title,
                label="DKN",
                save_path=plot_dir / f"{run_name}_prediction.png",
            )


if __name__ == "__main__":
    main()
