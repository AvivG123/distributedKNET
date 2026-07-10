# Nonlinear Notebook Comparison Reference

This reference captures the comparison logic from the end of
`notebooks/non_linear_simulation.ipynb`.

## Imports and Core Objects

The notebook uses:

- `diffusion_extended_kalman_filter_parallel_edge` from
  `utils.ClassicDistributedKalman`
- `HSystem`, `HSystemLinear`, `FSystem`, `FSystemLinear`, `GraphDataset`, and
  `CreateGraph` from `utils.DistributedKalmanData`
- `GraphKalmanProcess` from `utils.DistributedKalmanNet`
- `GnnRnnLightning` from `utils.BaselineModels`

It creates `FIGURES_DIR = PROJECT_ROOT / "figures"` and saves plots there.

## Helper Functions

Classical DEKF evaluation:

```python
def run_experiment_classical_diffusion_kalman(dataset, f_sys, h_sys, r_array, q, P0, x0, time_steps, node_num):
    real_x_list = []
    predicted_x_list = []
    for graph_i in dataset:
        real_x = graph_i.y[..., 0].numpy().T
        real_x_list.append(real_x)
        y_meas = graph_i.x.numpy().transpose(0, 2, 1)
        adj_matrix_graph = graph_i.adj_matrix.numpy()
        predicted_x = diffusion_extended_kalman_filter_parallel_edge(
            y_meas, f_sys, h_sys, r_array, q, P0, x0, adj_matrix_graph,
            time_steps=time_steps, node_num=node_num
        )
        predicted_x_list.append(predicted_x)
    return np.stack(real_x_list, axis=0), np.stack(predicted_x_list, axis=0)
```

Mean distance:

```python
def mean_distance_function(real_x, predicted_x):
    return np.linalg.norm(
        predicted_x - real_x.transpose(0, 2, 1)[:, :, None, ..., None],
        axis=(-1, -2),
    ).mean(axis=(0, 2))
```

Consensus distance:

```python
def consensus_distance_function(predicted_x):
    return np.linalg.norm(
        predicted_x - predicted_x.mean(axis=1, keepdims=True),
        axis=(-1, -2),
    ).mean(axis=(0, 2))
```

## Notebook Defaults

The nonlinear comparison used:

- `NODE_NUM = 50`
- `TIME_STEPS = 10` during training setup in the visible notebook cells
- final plot titles mention `T=20`
- `q = 1`
- `R_SCALE = 1`
- `x0 = [[1], [1]]`
- `P0 = eye(2)`
- `mismatch_angle = 20 degrees`
- graph from `CreateGraph(node_num, 5)`
- nonlinear systems from `FSystem` and `HSystem`

When matching a user-requested setup, prefer the user's setup over these
defaults. Use these values only when the user asks to reproduce the notebook.

## Noise Axis

The final comparison uses:

```python
noise_scales = np.array([0.25, 0.5, 1, 2, 4])
err_db = 10 * np.log10(1 / noise_scales**2)
```

Use `err_db` as the x-axis for dB plots.

## Final Nonlinear Notebook Arrays

The notebook's saved nonlinear figure uses these arrays:

```python
no_mismatch_classic_err_db = [-7.29, -5.548, -3.097, -0.754, 1.276]
no_mismatch_deep_err_db = [-6.924, -5.332, -3.817, -1.33, 0.216]
mismatch_classic_err_db = [-2.147, -1.366, 0.469, 3.411, 6.66]
mismatch_deep_err_db = [-3.439, -2.655, -1.754, 0.632, 3.594]
gnn_rnn_err_db = [2.9, 3.23, 3.501, 4.00, 4.917]
```

Plot style:

```python
plt.figure(figsize=(8, 6))
plt.plot(err_db, no_mismatch_classic_err_db, "rv--", label="DEKF")
plt.plot(err_db, no_mismatch_deep_err_db, "bo--", label="Distributed KalmanNet")
plt.plot(err_db, mismatch_classic_err_db, "r^-.", label=r"DEKF (mismatched)")
plt.plot(err_db, mismatch_deep_err_db, "bs-.", label=r"Distributed KalmanNet (mismatched)")
plt.plot(err_db, gnn_rnn_err_db, "g*--", label="GNN-RNN")
plt.legend()
plt.grid(True)
plt.xlabel(r"$\frac{1}{r^2}$ [dB] ", fontsize=16)
plt.ylabel("MSE [dB]", fontsize=16)
plt.savefig(FIGURES_DIR / "nonlinear_50n_20t.png", dpi=300)
```

## Final Linear Notebook Arrays

The notebook's saved linear figure uses:

```python
linear_no_mismatch_classic_err_db = [-4.922, -3.496, -1.631, 0.258, 1.987]
linear_no_mismatch_deep_err_db = [-4.4, -2.926, -2.22, -0.45, 1.339]
linear_mismatch_classic_err_db = [-0.473, 0.458, 2.362, 4.826, 6.771]
linear_mismatch_deep_err_db = [-3.413, -2.668, -1.707, 0.543, 2.199]
linear_gnn_rnn_err_db = [3.807, 3.819, 3.95, 4.645, 5.623]
```

Plot style:

```python
plt.figure(figsize=(8, 6))
plt.plot(err_db, linear_no_mismatch_classic_err_db, "rv--", label="DEKF")
plt.plot(err_db, linear_no_mismatch_deep_err_db, "bo--", label="Distributed KalmanNet")
plt.plot(err_db, linear_mismatch_classic_err_db, "r^-.", label=r"DEKF rotated (mismatched)")
plt.plot(err_db, linear_mismatch_deep_err_db, "bs-.", label=r"Distributed KalmanNet (mismatched)")
plt.plot(err_db, linear_gnn_rnn_err_db, "g*--", label="GNN-RNN")
plt.legend()
plt.grid(True)
plt.xlabel(r"$\frac{1}{r^2}$ [dB] ", fontsize=16)
plt.ylabel("MSE [dB]", fontsize=16)
plt.savefig(FIGURES_DIR / "linear_50n_20t.png", dpi=300)
```

## Recompute Instead of Hardcoding

For new training runs, recompute the learned-model and DEKF metrics rather than
using the notebook arrays. The arrays above are only for exact reproduction or
for checking whether a new graph visually matches the notebook style.
