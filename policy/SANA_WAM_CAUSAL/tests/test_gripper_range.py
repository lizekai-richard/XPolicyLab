"""The deploy key ``gripper_range``: the model-domain range of the gripper closedness.

Sana 41cf94114 (2026-10-31) builds the normalization artifacts with the grippers unnormalized (mask off, q01 0 / q99 1
recorded, center 0, scale 1, ``gripper_normalization: identity``): the model sees closedness in [0, 1]. The artifacts
of 2026-09-20 up to it (the causal recipes' 1fe3b7e7, fixture here) map it by q01 0 / q99 1 (center 0.5, scale 0.5,
mask on): the model sees [-1, 1]. ``auto`` serves the artifact's own map; an explicit range rewrites the four gripper
slots and leaves every other slot alone. The [0, 1] artifact is derived from the 1fe3b7e7 fixture by 41cf94114's rule
(its commit: the new artifacts differ from their predecessors only at slots 16 / 45 and the stamps).
"""

from __future__ import annotations

import inspect
import json
import os

import numpy as np
import pytest
import torch

import _paths

from sana_wam_min.robot80 import (
    GRIPPER_RANGE_AUTO,
    GRIPPER_RANGE_SIGNED,
    GRIPPER_RANGE_UNIT,
    apply_gripper_range,
    denormalize_action,
    gripper_range_of,
    load_normalization,
    normalize_action,
    normalize_state,
    normalized_gripper_bounds,
    resolve_gripper_range,
    with_gripper_range,
)

SIGNED_ARTIFACT = os.path.join(_paths.FIXTURES, "normalization_causal_f33_1fe3b7e7.json")
GRIPPERS = (16, 45)
FIELDS = ("q01_80", "q99_80", "center80", "scale80", "normalization_mask80")


@pytest.fixture(scope="module")
def signed_norm():
    return load_normalization(SIGNED_ARTIFACT)


@pytest.fixture(scope="module")
def unit_norm(tmp_path_factory):
    """The 1fe3b7e7 artifact rebuilt by Sana 41cf94114's rule: gripper slots identity, endpoints 0 / 1 recorded."""

    with open(SIGNED_ARTIFACT) as handle:
        artifact = json.load(handle)
    for kind in ("state", "action"):
        block = artifact[kind]
        for slot in GRIPPERS:
            block["q01_80"][slot], block["q99_80"][slot] = 0.0, 1.0
            block["center80"][slot], block["scale80"][slot] = 0.0, 1.0
            block["normalization_mask80"][slot] = False
    artifact["gripper_normalization"] = "identity"
    path = tmp_path_factory.mktemp("norm") / "normalization_unit_grippers.json"
    path.write_text(json.dumps(artifact))
    return load_normalization(path)


def _arrays(norm):
    return {f"{kind}_{field}": np.asarray(getattr(norm, f"{kind}_{field}")) for kind in ("state", "action") for field in FIELDS}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, GRIPPER_RANGE_AUTO),
        ("auto", GRIPPER_RANGE_AUTO),
        ("", GRIPPER_RANGE_AUTO),
        ([-1, 1], GRIPPER_RANGE_SIGNED),
        ((-1.0, 1.0), GRIPPER_RANGE_SIGNED),
        ("[-1, 1]", GRIPPER_RANGE_SIGNED),
        ("-1,1", GRIPPER_RANGE_SIGNED),
        ([0, 1], GRIPPER_RANGE_UNIT),
        ("[0,1]", GRIPPER_RANGE_UNIT),
        ("[0.0, 1.0]", GRIPPER_RANGE_UNIT),
    ],
)
def test_resolve(value, expected):
    assert resolve_gripper_range(value) == expected


@pytest.mark.parametrize("value", ["[0, 2]", [1, 0], "signed", [-1, 0, 1], "[a, b]"])
def test_resolve_refuses(value):
    with pytest.raises(ValueError, match="gripper_range"):
        resolve_gripper_range(value)


def test_artifact_ranges(signed_norm, unit_norm):
    assert signed_norm.gripper_normalization == "statistics"
    assert gripper_range_of(signed_norm) == GRIPPER_RANGE_SIGNED
    assert normalized_gripper_bounds(signed_norm) == ((-1.0, 1.0), (-1.0, 1.0))
    assert gripper_range_of(unit_norm) == GRIPPER_RANGE_UNIT
    assert normalized_gripper_bounds(unit_norm) == ((0.0, 1.0), (0.0, 1.0))


def test_auto_serves_the_artifact(signed_norm, unit_norm):
    for norm, label in ((signed_norm, "[-1, 1](artifact)"), (unit_norm, "[0, 1](artifact)")):
        served, served_label = apply_gripper_range(norm, None)
        assert served is norm and served_label == label
        served, served_label = apply_gripper_range(norm, "auto")
        assert served is norm and served_label == label


def test_explicit_range_equal_to_the_artifact_changes_nothing(signed_norm, unit_norm):
    served, label = apply_gripper_range(signed_norm, [-1, 1])
    assert served is signed_norm and label == "[-1, 1](deploy = artifact)"
    served, label = apply_gripper_range(unit_norm, "[0, 1]")
    assert served is unit_norm and label == "[0, 1](deploy = artifact)"


def test_forced_maps_equal_the_other_artifact(signed_norm, unit_norm):
    """Forcing [-1, 1] on the 41cf94114 artifact gives the 1fe3b7e7 one slot for slot, and the reverse."""

    for source, target, requested, label in (
        (unit_norm, signed_norm, [-1, 1], "[-1, 1](deploy OVERRIDES artifact [0, 1])"),
        (signed_norm, unit_norm, "[0, 1]", "[0, 1](deploy OVERRIDES artifact [-1, 1])"),
    ):
        served, served_label = apply_gripper_range(source, requested)
        assert served_label == label
        assert gripper_range_of(served) == gripper_range_of(target)
        expected, got = _arrays(target), _arrays(served)
        for name in expected:
            np.testing.assert_array_equal(got[name], expected[name], err_msg=name)
        assert served.sha256 == source.sha256 and served.gripper_normalization == source.gripper_normalization
        # the source is untouched
        assert gripper_range_of(source) != gripper_range_of(target)


def test_values_through_the_served_maps(signed_norm, unit_norm):
    rows = torch.zeros(3, 80)
    mask = torch.zeros(3, 80, dtype=torch.bool)
    mask[:, 7:17] = True
    mask[:, 36:46] = True
    rows[:, 7:10] = torch.tensor([0.3, -0.1, 0.2])
    rows[:, 36:39] = torch.tensor([0.25, 0.1, 0.22])
    rows[:, 10:16] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rows[:, 39:45] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    closedness = torch.tensor([0.0, 0.25, 1.0])
    rows[:, 16], rows[:, 45] = closedness, closedness
    forced_unit = with_gripper_range(signed_norm, GRIPPER_RANGE_UNIT)
    for norm, expected in (
        (signed_norm, 2.0 * closedness - 1.0),
        (unit_norm, closedness),
        (forced_unit, closedness),
        (with_gripper_range(unit_norm, GRIPPER_RANGE_SIGNED), 2.0 * closedness - 1.0),
    ):
        action = normalize_action(rows, mask, norm)
        state = normalize_state(rows[0], mask[0], norm)
        for slot in GRIPPERS:
            torch.testing.assert_close(action[:, slot], expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(state[slot], expected[0], rtol=0, atol=1e-7)
        torch.testing.assert_close(denormalize_action(action, mask, norm), rows, rtol=0, atol=1e-6)
    # every non-gripper slot is the artifact's map either way
    other = [slot for slot in range(80) if slot not in GRIPPERS]
    torch.testing.assert_close(
        normalize_action(rows, mask, forced_unit)[:, other], normalize_action(rows, mask, signed_norm)[:, other], rtol=0, atol=0
    )


def test_runtime_and_deploy_carry_the_key():
    import yaml
    from sana_wam_causal.runtime import CausalRuntime

    assert "gripper_range" in inspect.signature(CausalRuntime.from_paths).parameters
    with open(os.path.join(_paths.ADAPTER_DIR, "deploy.yml")) as handle:
        deploy = yaml.safe_load(handle)
    assert resolve_gripper_range(deploy["gripper_range"]) == GRIPPER_RANGE_AUTO
