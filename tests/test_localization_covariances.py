import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.localization_experiment import (
    build_dkn_model,
    derive_localization_noise,
    measurement_noise_for_nodes,
)
from utils.DistributedKalmanData import GraphDataset, _stream_seed, generate_data_points
from utils.DistributedKalmanNet import loss_function
from utils.LocalizationScenario import (
    ConstantVelocityModel,
    DistanceAngleObservation,
    generate_measurements,
)


def _dataset_shell(simulations=20_000, state_dim=2, seed=7):
    dataset = GraphDataset.__new__(GraphDataset)
    dataset.monte_carlo_simulations = simulations
    dataset.state_dim = state_dim
    dataset.seed = seed
    return dataset


def test_initial_state_covariance_accepts_scalar():
    mean = np.array([1.0, -2.0])
    samples = _dataset_shell()._build_initial_state_batch(mean, 2.0)[..., 0]

    assert np.allclose(samples.mean(axis=0), mean, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), np.eye(2) * 2.0, atol=0.08)


def test_initial_state_none_preserves_unit_covariance_and_batched_x0():
    dataset = _dataset_shell(simulations=4)
    samples = dataset._build_initial_state_batch(np.zeros(2), None)
    expected = np.random.RandomState(7).randn(4, 2, 1).astype(np.float32)
    assert np.array_equal(samples, expected)

    explicit = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
    assert np.array_equal(dataset._build_initial_state_batch(explicit, 9.0), explicit)


def test_rng_streams_do_not_overlap_for_adjacent_dataset_seeds():
    train_streams = {_stream_seed(42, stream) for stream in range(3)}
    validation_streams = {_stream_seed(43, stream) for stream in range(3)}

    assert train_streams.isdisjoint(validation_streams)


def test_velocity_only_process_covariance_leaves_positions_noise_free():
    x0 = np.zeros((128, 4, 1), dtype=np.float32)
    covariance = np.diag([0.0, 4.0, 0.0, 9.0])
    samples = generate_data_points(lambda state: state, covariance, x0, 1, seed=3)
    assert np.count_nonzero(samples[:, [0, 2], :, :]) == 0
    assert np.std(samples[:, 1, 0, 0]) > 1.0
    assert np.std(samples[:, 3, 0, 0]) > 2.0


def test_noise_derivation_and_sensor_specific_measurement_std():
    rho = (1.0 / math.radians(2.0)) ** 2
    noise = derive_localization_noise(mu=10, rho=rho, r_scale=2.0)
    assert np.isclose(noise["q"], 2.0 * math.sqrt(10))
    assert np.isclose(noise["sigma_r"], 2.0)
    assert np.isclose(noise["sigma_theta"], 2.0 * math.radians(2.0))
    assert np.allclose(
        noise["q_matrix"],
        np.diag([0.0, 40.0, 0.0, 40.0]),
    )

    observation = DistanceAngleObservation(np.zeros((4, 2)))
    per_node = measurement_noise_for_nodes(
        observation, noise["sigma_r"], noise["sigma_theta"]
    )
    assert np.allclose(
        per_node,
        [noise["sigma_theta"], noise["sigma_r"]] * 2,
    )


def test_localization_dkn_uses_noise_derived_from_r_scale():
    observation = DistanceAngleObservation(np.zeros((2, 2)))
    noise = derive_localization_noise(mu=4.0, rho=16.0, r_scale=2.0)
    r_array = measurement_noise_for_nodes(
        observation, noise["sigma_r"], noise["sigma_theta"]
    )
    model = build_dkn_model(
        {
            "state_dimension": 4,
            "hidden_dim": 8,
            "learning_rate": 1e-3,
            "learn_edge_kalman": True,
        },
        ConstantVelocityModel(0.1),
        r_array,
        np.zeros((4, 1)),
    )

    assert np.allclose(model.r_array, [0.5, 2.0])
    assert np.allclose(model.gkf.edge_kalman.r_inv.numpy(), [4.0, 0.25])


def test_generate_measurements_accepts_per_node_standard_deviations(monkeypatch):
    observation = DistanceAngleObservation(np.array([[0.0, 0.0], [0.0, 0.0]]))
    trajectory = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape))
    clean = np.array([math.pi / 4, math.sqrt(2.0)])
    measurements = generate_measurements(observation, trajectory, [0.1, 2.0])
    assert np.allclose(measurements[:, 0, 0], clean + [0.1, 2.0])


def test_doa_convention_and_jacobian_match_analytic_values():
    observation = DistanceAngleObservation(np.array([[0.0, 0.0], [0.0, 0.0]]))
    state = np.array([[2.0], [0.0], [1.0], [0.0]])
    assert np.isclose(observation.func(state)[0, 0], math.atan2(2.0, 1.0))
    jacobian = observation.jacobian(state, n_expansions=1)[0, 0, 0, :, 0]
    assert np.allclose(jacobian, [1.0 / 5.0, 0.0, -2.0 / 5.0, 0.0])

    tensor_state = torch.tensor(state, dtype=torch.float32)
    tensor_jacobian = observation.jacobian(tensor_state, n_expansions=1)
    assert np.allclose(tensor_jacobian[0, 0, 0, :, 0].numpy(), jacobian)


def test_position_only_loss_ignores_velocity_errors():
    truth = torch.zeros(1, 1, 4, 1)
    prediction = torch.zeros(1, 1, 2, 4, 1)
    prediction[..., 1, 0] = 10
    prediction[..., 3, 0] = -10
    assert loss_function(prediction, truth, position_indices=[0, 2]).item() == 0
    assert loss_function(prediction, truth).item() > 0


def test_constant_velocity_moves_velocity_uncertainty_into_position():
    model = ConstantVelocityModel(0.5)
    state = np.array([[0.0], [2.0], [0.0], [-4.0]])
    assert np.allclose(model(state)[:, 0], [1.0, 2.0, -2.0, -4.0])
