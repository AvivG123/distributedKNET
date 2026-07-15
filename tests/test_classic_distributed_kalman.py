import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ClassicDistributedKalman import (
    centralized_extended_kalman_filter,
    diffusion_extended_kalman_filter,
    diffusion_extended_kalman_filter_parallel_edge,
    local_extended_kalman_filter,
)
from utils.LocalizationScenario import (
    ConstantVelocityModel,
    DistanceAngleObservation,
    generate_measurements,
)


FILTERS = (
    centralized_extended_kalman_filter,
    diffusion_extended_kalman_filter,
    diffusion_extended_kalman_filter_parallel_edge,
    local_extended_kalman_filter,
)


def _filter_inputs():
    node_positions = np.array([[0.0, 0.0], [3.0, 2.0]])
    observation = DistanceAngleObservation(node_positions)
    motion = ConstantVelocityModel(0.1)
    trajectory = torch.tensor(
        [
            [[1.0], [0.2], [1.5], [-0.1]],
            [[1.02], [0.2], [1.49], [-0.1]],
        ]
    )
    measurements = generate_measurements(observation, trajectory, 0.0)
    common = {
        "measurements": measurements,
        "f_system": motion,
        "h_system": observation,
        "r_array": np.array([0.1, 0.2]),
        "q": 0.3,
        "p0": np.eye(4),
        "x0": np.array([[1.0], [0.2], [1.5], [-0.1]]),
        "time_steps": 2,
        "node_num": 2,
    }
    return common


@pytest.mark.parametrize("filter_function", FILTERS)
def test_all_classical_filters_accept_full_process_covariance(filter_function):
    kwargs = _filter_inputs()
    if "j_matrix" in inspect.signature(filter_function).parameters:
        kwargs["j_matrix"] = np.ones((2, 2))
    kwargs["q_matrix"] = np.diag([0.0, 0.09, 0.0, 0.09])
    estimate = filter_function(**kwargs)
    assert estimate.shape[0] == 2
    assert np.isfinite(estimate).all()


@pytest.mark.parametrize("filter_function", FILTERS)
def test_omitted_q_matrix_preserves_isotropic_q_behavior(filter_function):
    kwargs = _filter_inputs()
    if "j_matrix" in inspect.signature(filter_function).parameters:
        kwargs["j_matrix"] = np.ones((2, 2))
    legacy = filter_function(**kwargs)
    explicit = filter_function(
        **kwargs,
        q_matrix=np.eye(4) * kwargs["q"] ** 2,
    )
    assert np.allclose(legacy, explicit)


@pytest.mark.parametrize("filter_function", FILTERS)
def test_all_classical_filters_accept_zero_initial_covariance(filter_function):
    kwargs = _filter_inputs()
    kwargs["p0"] = np.zeros((4, 4))
    kwargs["q_matrix"] = np.diag([0.0, 0.09, 0.0, 0.09])
    if "j_matrix" in inspect.signature(filter_function).parameters:
        kwargs["j_matrix"] = np.ones((2, 2))

    estimate = filter_function(**kwargs)

    assert estimate.shape[0] == 2
    assert np.isfinite(estimate).all()


@pytest.mark.parametrize("filter_function", FILTERS)
def test_first_measurement_uses_propagated_initial_prior(filter_function):
    kwargs = _filter_inputs()
    expected = kwargs["f_system"](kwargs["x0"])
    trajectory = torch.as_tensor(
        np.stack([expected, kwargs["f_system"](expected)]), dtype=torch.float32
    )
    kwargs["measurements"] = generate_measurements(
        kwargs["h_system"], trajectory, 0.0
    )
    kwargs["p0"] = np.zeros((4, 4))
    kwargs["q_matrix"] = np.eye(4)
    if "j_matrix" in inspect.signature(filter_function).parameters:
        kwargs["j_matrix"] = np.ones((2, 2))

    estimate = filter_function(**kwargs)
    first_estimate = estimate[0] if estimate.ndim == 3 else estimate[0, 0]

    assert np.allclose(first_estimate, expected)


def test_q_matrix_is_trailing_for_positional_call_compatibility():
    for filter_function in FILTERS:
        assert list(inspect.signature(filter_function).parameters)[-1] == "q_matrix"
