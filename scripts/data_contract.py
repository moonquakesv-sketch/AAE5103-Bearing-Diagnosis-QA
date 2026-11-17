"""Shared sampling/configuration checks for validation and the future D02 pipeline."""
import math


class ContractError(ValueError):
    """An input violates the documented data contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def integer(value, name, minimum=1):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{name}: expected an integer >= {minimum}, got {value!r}")
    return value


def resample_parameters(cfg, original_fs_hz, kind):
    """Return the exact SciPy parameters used by checks and signal processing."""
    require(kind in ("normal", "fault"), f"Unknown sampling kind: {kind}")
    sampling = cfg["sampling"]
    original = integer(original_fs_hz, "original_fs_hz")
    target = integer(sampling["target_fs_hz"], "target_fs_hz")
    declared = integer(sampling[f"{kind}_original_fs_hz"], f"{kind}_original_fs_hz")
    require(original == declared, f"{kind}: original_fs_hz {original} != declared {declared}")
    options = sampling["resample"]
    up = integer(options[f"{kind}_up"], f"{kind}_up")
    down = integer(options[f"{kind}_down"], f"{kind}_down")
    require(original * up == target * down,
            f"{kind}: resampling ratio {up}/{down} does not map {original} Hz to {target} Hz")
    window = options["window"]
    return {"up": up, "down": down,
            "window": tuple(window) if isinstance(window, list) else window,
            "padtype": options["padtype"]}


def resample_signal(values, cfg, original_fs_hz, kind):
    """Full-record resampling; callers perform slicing after this function."""
    from scipy.signal import resample_poly
    return resample_poly(values, **resample_parameters(cfg, original_fs_hz, kind))


def processed_length(n_samples, parameters):
    """Integer ceiling without truncating a sampling-rate ratio."""
    return (n_samples * parameters["up"] + parameters["down"] - 1) // parameters["down"]


def validate_config(cfg):
    require(cfg["schema_version"] == 1, "Unsupported configuration schema")
    labels = cfg["labels"]
    require(len(labels) == 4 and set(labels) == {
        "normal", "inner_race", "outer_race", "rolling_element"}, "Invalid label set")
    sampling = cfg["sampling"]
    require(sampling["resample"]["implementation"] == "scipy.signal.resample_poly",
            "Unsupported resampling implementation")
    require(sampling["resample"]["scope"] == "full_selected_record_before_slicing",
            "Resampling must precede slicing")
    for kind in ("normal", "fault"):
        resample_parameters(cfg, sampling[f"{kind}_original_fs_hz"], kind)
    edge = integer(sampling["discard_edge_samples_each_side"], "edge discard", 0)
    segment = cfg["segmentation"]
    require(integer(segment["first_start_sample"], "first_start_sample", 0) == edge,
            "first_start_sample must equal edge discard")
    length = integer(segment["window_samples"], "window_samples")
    require(integer(segment["stride_samples"], "stride_samples") == length,
            "This protocol requires contiguous non-overlapping windows (stride == window)")
    require(segment["index_space"] == "full_resampled_signal_before_edge_discard"
            and segment["index_base"] == 0 and segment["end_exclusive"] is True,
            "Unsupported window index convention")
    require(segment["drop_incomplete_tail"] is True, "Incomplete tails must be discarded")
    require(cfg["split"]["random_window_split"] is False, "Random window splits are forbidden")
    require([cfg["split"][f"{s}_load_hp"] for s in ("train", "validation", "test")]
            == [[0, 1], [2], [3]], "load_holdout_v1 must preserve the 0/1, 2, 3 HP split")
    cwt = cfg["cwt"]
    require(math.isclose(cwt["sampling_period_seconds"], 1 / sampling["target_fs_hz"],
                         rel_tol=1e-12, abs_tol=0), "CWT sampling_period_seconds != 1/target_fs_hz")
    require(0 < cwt["frequency_min_hz"] < cwt["frequency_max_hz"] < sampling["target_fs_hz"] / 2,
            "CWT frequency limits must lie below Nyquist")
    integer(cwt["n_frequencies"], "n_frequencies", 2)
    integer(cwt["precision"], "CWT precision", 12)
    integer(cwt["pad_samples_each_side"], "CWT padding", 0)
    require(cwt["frequency_grid"] == "geomspace" and cwt["frequency_order"] == "ascending",
            "Unsupported CWT frequency grid")
    require(cfg["image"]["color_limits"]["fit_split"] == "train",
            "Color limits must be fitted on training data only")
