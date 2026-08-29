import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.graphkalmanprocess_hparams import LOCALIZATION_BASELINE
from experiments.localization_experiment import (
    normalize_localization_config,
    select_curriculum_checkpoint,
)


def _config_with(system_updates=None, data_updates=None):
    config = deepcopy(LOCALIZATION_BASELINE)
    config["system"].update(system_updates or {})
    config["data"].update(data_updates or {})
    return config


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([1.0, 5.0], [1.0, 5.0]),
        (5, [5.0]),
        (5.0, [5.0]),
        ("5", [5.0]),
        ("1,5", [1.0, 5.0]),
        ("1, 5, 10", [1.0, 5.0, 10.0]),
        ("1,", [1.0]),
    ],
)
def test_dt_mismatch_values_accepts_cli_scalars_and_lists(raw, expected):
    """--override parses values into scalars, so the list form must be coerced."""
    config = normalize_localization_config(_config_with({"dt_mismatch_values": raw}))
    assert config["dt_mismatch_values"] == expected


@pytest.mark.parametrize("raw", ["", "fast", [], "1,x"])
def test_dt_mismatch_values_rejects_non_numeric(raw):
    with pytest.raises(ValueError, match="dt_mismatch_values"):
        normalize_localization_config(_config_with({"dt_mismatch_values": raw}))


def test_r_scale_accepts_the_same_forms():
    config = normalize_localization_config(_config_with(data_updates={"r_scale": "0.25,1"}))
    assert config["r_scale"] == [0.25, 1.0]
    config = normalize_localization_config(_config_with(data_updates={"r_scale": 1.0}))
    assert config["r_scale"] == [1.0]


def test_preset_matches_the_reference_experiment_run_log():
    """Pinned to models/_localization/experiment_2/run.log.

    New sweeps have to share that run's noise model and stopping policy to stay
    comparable with the checkpoints already in that directory; rho in particular
    sets sigma_theta and silently changes the measurement model if it drifts.
    """
    config = normalize_localization_config(deepcopy(LOCALIZATION_BASELINE))
    assert config["rho"] == 32.82806350011744
    assert config["mu"] == 1.0
    assert config["r_scale"] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert config["dt_mismatch_values"] == [1.0, 2.0, 5.0]
    assert config["learning_rate"] == 1e-4
    assert config["max_epochs"] == 100
    assert config["early_stop_patience"] == 5
    assert config["early_stop_min_delta"] == 0.001
    assert config["num_nodes"] == 50
    assert config["k_neighbors"] == 5
    assert config["time_steps"] == 20
    assert LOCALIZATION_BASELINE["curriculum"]["epochs_per_stage"] == 20


def test_select_curriculum_checkpoint_prefers_the_final_stage():
    short_horizon = Path("stage.ts10.ckpt")
    full_horizon = Path("stage.ts20.ckpt")
    # The short stage posts the lower loss purely because it accumulates less
    # error, so the argmin would pick it; the final stage must win anyway.
    assert (
        select_curriculum_checkpoint([(short_horizon, 0.1), (full_horizon, 0.5)])
        == full_horizon
    )


def test_select_curriculum_checkpoint_falls_back_to_an_earlier_stage():
    short_horizon = Path("stage.ts10.ckpt")
    assert (
        select_curriculum_checkpoint([(short_horizon, 0.5), (None, None)])
        == short_horizon
    )


def test_select_curriculum_checkpoint_returns_none_without_any_checkpoint():
    assert select_curriculum_checkpoint([(None, None)]) is None
    assert select_curriculum_checkpoint([]) is None


def test_run_log_round_trip_preserves_noise_model_and_geometry(tmp_path):
    """Reproducing a run must reuse rho and the node positions, not the preset."""
    from experiments.localization_experiment import (
        load_experiment_config,
        save_experiment_config,
    )

    original = normalize_localization_config(deepcopy(LOCALIZATION_BASELINE))
    original["rho"] = 32.82806350011744
    positions = [[float(i), float(2 * i)] for i in range(original["num_nodes"])]
    save_experiment_config(tmp_path, original, positions)

    restored = load_experiment_config(tmp_path)
    assert restored["rho"] == 32.82806350011744
    assert restored["node_positions"] == positions
    # Overrides on a restored config use the flat run.log key names.
    restored["r_scale"] = 0.25
    restored["dt_mismatch_values"] = 2
    renormalized = normalize_localization_config(restored)
    assert renormalized["r_scale"] == [0.25]
    assert renormalized["dt_mismatch_values"] == [2.0]
    assert renormalized["rho"] == 32.82806350011744
    assert renormalized["node_positions"] == positions


def test_connectivity_and_noise_overrides_survive_normalization():
    config = normalize_localization_config(
        _config_with(
            {"dt_mismatch_values": 1.0, "use_dt_mismatch": False},
            {"r_scale": 1.0},
        )
    )
    assert config["num_nodes"] == 50
    assert config["k_neighbors"] == 5
    assert config["r_scale"] == [1.0]
    assert config["use_dt_mismatch"] is False
