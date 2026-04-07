"""
Config-Driven DKN Training

Trains GraphKalmanProcess on the distance/angle constant-velocity scenario
using a single config_val dictionary loaded from JSON.  Previews generated
trajectories for each run and saves both model weights and a matching JSON
config file for later loading.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "utils").exists() else SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float32)

from utils.DistributedKalmanData import GraphDataset
from utils.DistributedKalmanNet import GraphKalmanProcess
from utils.ConstantVelocityScenario import (
    ConstantVelocityModel,
    DistanceAngleObservation,
    create_distance_based_graph,
    get_trainer_accelerator,
    load_config,
    plot_generated_trajectories,
    plot_learning_curve,
    save_config,
    seed_everything,
)


def plot_dkn_prediction_sample(graph, prediction, node_positions, title):
    """Plot one validation trajectory against the DKN prediction."""
    x_true = graph.y[:, 0, 0].cpu().numpy()
    y_true = graph.y[:, 2, 0].cpu().numpy()
    x_pred = prediction[:, 0]
    y_pred = prediction[:, 2]

    plt.figure(figsize=(7, 7))
    plt.plot(x_true, y_true, "b-", linewidth=2, label="True trajectory")
    plt.plot(x_pred, y_pred, "m--", linewidth=2, label="DKN prediction")
    plt.scatter(
        node_positions[:, 0], node_positions[:, 1],
        c="black", marker="s", s=40, label="Nodes",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()


def main():
    # ── Configuration ─────────────────────────────────────────────────────
    config_path = REPO_ROOT / "configs" / "const_vel_scenario_configured.json"
    config_val = load_config(config_path)
    seed_everything(config_val["seed"])

    num_nodes = config_val["num_nodes"]
    state_dimension = config_val["state_dimension"]
    time_steps = config_val["time_steps"]
    time_delta = config_val["time_delta"]
    process_noise_std = config_val["process_noise_std"]
    measurement_noise_values = config_val["measurement_noise_values"]
    x0 = np.array(config_val["x0"], dtype=float).reshape(state_dimension, 1)
    node_positions = np.array(config_val["node_positions"], dtype=float)
    trainer_accelerator = get_trainer_accelerator()
    save_root = REPO_ROOT / "models/itay"
    log_root = REPO_ROOT / "models/itay/logs"

    print(f"Loaded config from: {config_path}")
    print(pd.Series(config_val))
    print(f"\nTrainer accelerator: {trainer_accelerator}")
    print(f"Initial state vector x0:\n{x0}")

    # ── Build scenario objects ────────────────────────────────────────────
    f_system = ConstantVelocityModel(time_delta)
    h_system = DistanceAngleObservation(node_positions)
    adjacency_matrix = create_distance_based_graph(
        node_positions,
        k_neighbors=config_val["k_neighbors"],
        seed=config_val["graph_seed"],
    )
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(int)

    model_dir = save_root / "DKN"
    plot_dir = save_root / "plots"
    config_dir = save_root / "configs"
    for path in [model_dir, plot_dir, config_dir, log_root]:
        path.mkdir(parents=True, exist_ok=True)

    print("Training node positions:")
    print(node_positions)

    # ── Train one model per measurement-noise level ───────────────────────
    for r_noise in measurement_noise_values:
        print(f"\nTraining model for measurement noise r = {r_noise}")
        r_array = r_noise * np.ones(num_nodes)

        train_dataset = GraphDataset(
            adjacency_matrix,
            f_system,
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
            f_system,
            h_system,
            process_noise_std,
            r_array,
            monte_carlo_simulations=config_val["val_sims"],
            time_steps=time_steps,
            n_expansions=1,
            x0=x0,
            state_dim=state_dimension,
        )

        sample_trajectories = [
            train_dataset[idx].y.cpu()
            for idx in range(min(config_val["preview_trajectories"], len(train_dataset)))
        ]
        plot_generated_trajectories(
            node_positions,
            node_types,
            sample_trajectories,
            max_trajectories=config_val["preview_trajectories"],
            title_prefix=f"Training trajectories (r={r_noise})",
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

        kalman_process = GraphKalmanProcess(
            f_system,
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

        logger = CSVLogger(
            str(log_root),
            name=f"const_vel_scenario_r={r_noise}",
            version="0",
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss:_epoch",
            patience=5,
            verbose=True,
            mode="min",
            min_delta=0.001,
        )
        trainer = pl.Trainer(
            max_epochs=config_val["max_epochs"],
            accelerator=trainer_accelerator,
            devices=1,
            logger=logger,
            log_every_n_steps=5,
            callbacks=[early_stopping],
            gradient_clip_val=1,
        )
        trainer.fit(kalman_process, train_loader, val_loader)

        # ── Save model + config ──────────────────────────────────────────
        model_path = model_dir / f"const_vel_scenario_r={r_noise}.pth"
        run_config_path = config_dir / f"const_vel_scenario_r={r_noise}.json"
        torch.save(kalman_process.state_dict(), model_path)

        run_config = dict(config_val)
        run_config["measurement_noise"] = float(r_noise)
        run_config["graph_type"] = "distance_based_graph"
        run_config["model_path"] = str(model_path)
        run_config["config_path"] = str(run_config_path)
        save_config(run_config, run_config_path)

        plot_learning_curve(logger.log_dir, r_noise, save_dir=str(plot_dir))

        # ── Quick visual check on one validation sample ──────────────────
        sample_graph = val_dataset[0]
        with torch.no_grad():
            sample_prediction = (
                kalman_process(sample_graph).squeeze().cpu().numpy().mean(axis=1)
            )
        plot_dkn_prediction_sample(
            sample_graph, sample_prediction, node_positions,
            title=f"DKN validation example (r={r_noise})",
        )

        print(f"Saved model:  {model_path}")
        print(f"Saved config: {run_config_path}")


if __name__ == "__main__":
    main()
