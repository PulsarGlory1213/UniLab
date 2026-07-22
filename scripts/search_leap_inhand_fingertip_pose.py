"""Retarget the Allegro in-hand home grasp to the LEAP Hand.

The script reads the authored Allegro ``home`` keyframe and extracts its
rotation-invariant grasp geometry:

* fingertip directions around the ball;
* pairwise fingertip spacing;
* fingertip-to-ball surface gaps;
* ball clearance from the palm.

It then searches LEAP joint angles and a nearby ball position that reproduce
that geometry. The contact surface is not prescribed: a finger may naturally
support the ball with its pad, tip, or side.

Finally, the best geometric candidates are simulated with MuJoCo. The output
is the physically best settled pose, not merely the lowest kinematic score.
The three main distal joints are also kept in a moderate Allegro-like range,
so stability cannot be achieved by curling them to approximately 2 radians.
The thumb is constrained to a natural LEAP opposition family instead of using
extreme joint twist. Physical validation runs for a real-time duration and
requires sustained support, rather than accepting a single stable-looking
final frame.
"""

from __future__ import annotations

import argparse
import heapq
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from unilab.base.backend.mujoco.xml import materialize_scene_fragments


ROOT = Path(__file__).resolve().parents[1]
LEAP_DIR = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand"
ALLEGRO_DIR = ROOT / "src" / "unilab" / "assets" / "robots" / "allegro_hand"

LEAP_TIP_GEOMS = (
    "fingertip_collision",
    "fingertip_2_collision",
    "fingertip_3_collision",
    "thumb_fingertip_collision",
)
ALLEGRO_TIP_GEOMS = (
    "ff_tip_col",
    "mf_tip_col",
    "rf_tip_col",
    "th_tip_col",
)
LEAP_CONTACT_SENSORS = (
    "leap_ff_contact",
    "leap_mf_contact",
    "leap_rf_contact",
    "leap_th_contact",
    "leap_rotation_palm_contact",
)

# Semantic joint correspondence. LEAP qpos ordering is not simply 0,1,2,3:
# the XML body traversal places joint "1" before joint "0".
JOINT_MAP = (
    (("ffj0", "ffj1", "ffj2", "ffj3"), ("0", "1", "2", "3")),
    (("mfj0", "mfj1", "mfj2", "mfj3"), ("4", "5", "6", "7")),
    (("rfj0", "rfj1", "rfj2", "rfj3"), ("8", "9", "10", "11")),
    (("thj0", "thj1", "thj2", "thj3"), ("12", "13", "14", "15")),
)


@dataclass(frozen=True)
class AllegroReference:
    direction_gram: np.ndarray
    pairwise_distance_by_radius: np.ndarray
    sorted_surface_gaps_by_radius: np.ndarray
    palm_gap_by_radius: float
    palm_ball_ratio: float
    normalized_joint_pose: dict[str, float]


@dataclass
class CandidateMetrics:
    score: float
    tip_gaps: np.ndarray
    layout_rmse: float
    pairwise_rmse: float
    palm_gap: float
    palm_ball_ratio: float


@dataclass
class SettledResult:
    total_score: float
    kinematic_score: float
    qpos: np.ndarray
    ball_pos: np.ndarray
    ball_quat: np.ndarray
    contacts: np.ndarray
    tip_gaps: np.ndarray
    palm_gap: float
    ball_drop: float
    recent_z_range: float
    ball_speed: float
    main_flex: np.ndarray
    distal_flex: np.ndarray
    thumb_pose: np.ndarray
    support_fraction: float
    finger_contact_fractions: np.ndarray
    palm_contact_fraction: float
    validation_z_drop: float
    horizontal_drift: float
    thumb_twist_score: float
    passed: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget Allegro's authored in-hand grasp to LEAP."
    )
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--physics-candidates", type=int, default=24)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=1500,
        help=(
            "Minimum number of MuJoCo physics steps per candidate. The actual "
            "number is increased when --validation-seconds requires more."
        ),
    )
    parser.add_argument(
        "--validation-seconds",
        type=float,
        default=10.0,
        help="Minimum physical simulation duration for every final candidate.",
    )
    parser.add_argument(
        "--validation-window-seconds",
        type=float,
        default=3.0,
        help=(
            "Final time window used to measure sustained contacts and drift."
        ),
    )
    parser.add_argument(
        "--min-support-fraction",
        type=float,
        default=0.90,
        help=(
            "Required fraction of the final validation window with at least "
            "two finger contacts."
        ),
    )
    parser.add_argument(
        "--real-contact-gap",
        type=float,
        default=0.00035,
        help=(
            "Maximum actual geom surface gap for a finger to count "
            "as a real contact, in metres."
        ),
    )
    parser.add_argument(
        "--ball-search-radius",
        type=float,
        default=0.020,
        help="Maximum xyz change from LEAP's authored ball position, in metres.",
    )
    parser.add_argument(
        "--distal-soft-limit",
        type=float,
        default=0.80,
        help=(
            "Soft limit in radians for the last joint of the three main "
            "fingers. Curl above this value is strongly penalized."
        ),
    )
    parser.add_argument(
        "--distal-hard-limit",
        type=float,
        default=1.10,
        help=(
            "Hard search limit in radians for the last joint of the three "
            "main fingers."
        ),
    )
    parser.add_argument(
        "--num-output-seeds",
        type=int,
        default=8,
        help="Number of diverse, stable seed poses to save.",
    )
    parser.add_argument(
        "--seed-bank-path",
        type=str,
        default="caches/leap_hand_seed_bank.npy",
        help=(
            "Output path for the seed bank. Relative paths are resolved "
            "under src/unilab/assets."
        ),
    )
    parser.add_argument(
        "--save-seed-bank",
        action="store_true",
        help="Save diverse passing poses as a (K, 23) NumPy seed bank.",
    )
    parser.add_argument(
        "--min-seed-rms-distance",
        type=float,
        default=0.06,
        help="Minimum RMS hand-joint distance between saved seeds, in radians.",
    )
    parser.add_argument(
        "--max-seed-joint-delta",
        type=float,
        default=0.45,
        help=(
            "Maximum absolute joint difference from seed 0. This keeps seed "
            "transitions small enough for in-hand rotation."
        ),
    )
    parser.add_argument(
        "--contact-fraction-threshold",
        type=float,
        default=0.70,
        help=(
            "A finger is classified as a sustained contact only when it is "
            "in contact for at least this fraction of the final validation "
            "window."
        ),
    )
    parser.add_argument(
        "--min-unique-contact-masks",
        type=int,
        default=3,
        help=(
            "Minimum number of sustained contact masks required before a "
            "seed bank is considered complete."
        ),
    )
    parser.add_argument(
        "--targeted-physics-candidates",
        type=int,
        default=8,
        help=(
            "Extra candidates retained for middle-finger bridge and takeover "
            "states, in addition to the generic physics candidates."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-seed-bank",
        action="store_true",
        help=(
            "Save a seed bank even when strict handoff coverage is missing. "
            "This is intended only for debugging."
        ),
    )
    parser.add_argument(
        "--index-main-soft-limit",
        type=float,
        default=0.90,
        help="Soft upper limit for the index middle-flex joint q2.",
    )
    parser.add_argument(
        "--index-distal-soft-limit",
        type=float,
        default=0.60,
        help="Soft upper limit for the index distal joint q3.",
    )
    parser.add_argument(
        "--index-min-height-offset",
        type=float,
        default=-0.004,
        help=(
            "Lowest allowed index-tip height relative to ball center, in metres."
        ),
    )
    parser.add_argument(
        "--index-max-height-offset",
        type=float,
        default=0.014,
        help=(
            "Highest preferred index-tip height relative to ball center, in metres."
        ),
    )
    parser.add_argument(
        "--index-gap-target",
        type=float,
        default=0.004,
        help="Preferred maximum index fingertip-to-ball surface gap.",
    )
    parser.add_argument(
        "--middle-gap-target",
        type=float,
        default=0.005,
        help="Preferred maximum middle fingertip-to-ball surface gap.",
    )
    parser.add_argument(
        "--write-scene",
        action="store_true",
        help="Write a passing settled result into leap_hand/scene.xml.",
    )
    parser.add_argument(
        "--force-write",
        action="store_true",
        help="Allow --write-scene even when the stability check fails.",
    )

    # Kept for compatibility with commands used earlier. The new implementation
    # always uses the Allegro reference and does not hard-code four directions.
    parser.add_argument(
        "--four-side-layout",
        "--allegro-layout",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _name_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def _load_leap_model() -> mujoco.MjModel:
    materialized = Path(
        materialize_scene_fragments(
            str(LEAP_DIR / "leap_hand.xml"),
            fragment_files=[str(LEAP_DIR / "scene.xml")],
        )
    )
    try:
        return mujoco.MjModel.from_xml_path(str(materialized))
    finally:
        materialized.unlink(missing_ok=True)


def _load_home(
    model: mujoco.MjModel,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _geom_ids(model: mujoco.MjModel, names: Iterable[str]) -> np.ndarray:
    return np.asarray(
        [
            _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in names
        ],
        dtype=np.int32,
    )


def _surface_gaps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first_geom_ids: np.ndarray,
    second_geom_id: int,
) -> np.ndarray:
    return np.asarray(
        [
            mujoco.mj_geomDistance(
                model,
                data,
                int(first_id),
                int(second_geom_id),
                0.25,
                None,
            )
            for first_id in first_geom_ids
        ],
        dtype=np.float64,
    )


def _pairwise_values(points: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            values.append(float(np.linalg.norm(points[first] - points[second])))
    return np.asarray(values, dtype=np.float64)


def _direction_gram(tips: np.ndarray, ball: np.ndarray) -> np.ndarray:
    directions = tips - ball[None, :]
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True),
        1e-9,
    )
    return directions @ directions.T


def _collision_geom_for_body(
    model: mujoco.MjModel,
    body_name: str,
) -> int:
    body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    candidates = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and int(model.geom_contype[geom_id]) != 0
    ]
    if not candidates:
        raise RuntimeError(
            f"No collision geom was found on body {body_name!r}."
        )
    return int(candidates[0])


def _joint_value(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> float:
    joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    return float(data.qpos[qpos_address])


def _joint_normalized_value(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> float:
    joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    lower, upper = model.jnt_range[joint_id]
    value = _joint_value(model, data, joint_name)
    return float((value - lower) / max(float(upper - lower), 1e-9))


def _extract_allegro_reference() -> AllegroReference:
    model = mujoco.MjModel.from_xml_path(str(ALLEGRO_DIR / "scene.xml"))
    data = _load_home(model)

    tip_ids = _geom_ids(model, ALLEGRO_TIP_GEOMS)
    ball_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    palm_id = _collision_geom_for_body(model, "palm")
    palm_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    tips = data.geom_xpos[tip_ids].copy()
    ball = data.geom_xpos[ball_id].copy()
    palm = data.xpos[palm_body_id].copy()
    ball_radius = float(model.geom_size[ball_id, 0])

    gaps = _surface_gaps(model, data, tip_ids, ball_id)
    palm_gap = float(
        mujoco.mj_geomDistance(
            model,
            data,
            palm_id,
            ball_id,
            0.25,
            None,
        )
    )

    mean_tip_reach = float(
        np.mean(np.linalg.norm(tips - palm[None, :], axis=1))
    )
    palm_ball_ratio = float(
        np.linalg.norm(ball - palm) / max(mean_tip_reach, 1e-9)
    )

    normalized_joint_pose: dict[str, float] = {}
    for allegro_names, _ in JOINT_MAP:
        for name in allegro_names:
            normalized_joint_pose[name] = _joint_normalized_value(
                model,
                data,
                name,
            )

    return AllegroReference(
        direction_gram=_direction_gram(tips, ball),
        pairwise_distance_by_radius=_pairwise_values(tips) / ball_radius,
        sorted_surface_gaps_by_radius=np.sort(gaps) / ball_radius,
        palm_gap_by_radius=palm_gap / ball_radius,
        palm_ball_ratio=palm_ball_ratio,
        normalized_joint_pose=normalized_joint_pose,
    )


def _leap_joint_bounds(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(16, -np.inf, dtype=np.float64)
    upper = np.full(16, np.inf, dtype=np.float64)

    for joint_id in range(model.njnt):
        qpos_address = int(model.jnt_qposadr[joint_id])
        if qpos_address >= 16:
            continue
        lower[qpos_address], upper[qpos_address] = model.jnt_range[joint_id]

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("Could not resolve all 16 LEAP joint limits.")
    return lower, upper


def _allegro_mapped_seed(
    leap_model: mujoco.MjModel,
    current_hand_qpos: np.ndarray,
    reference: AllegroReference,
) -> np.ndarray:
    seed = current_hand_qpos.copy()

    for allegro_names, leap_names in JOINT_MAP:
        for allegro_name, leap_name in zip(
            allegro_names,
            leap_names,
            strict=True,
        ):
            leap_joint_id = _name_id(
                leap_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                leap_name,
            )
            qpos_address = int(leap_model.jnt_qposadr[leap_joint_id])
            lower, upper = leap_model.jnt_range[leap_joint_id]
            normalized = reference.normalized_joint_pose[allegro_name]
            seed[qpos_address] = lower + normalized * (upper - lower)

    return seed


def _contact_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    values = []
    for name in LEAP_CONTACT_SENSORS:
        sensor_id = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(model.sensor_adr[sensor_id])
        values.append(float(data.sensordata[address]))
    return np.asarray(values, dtype=np.float64)


def _write_home_keyframe(
    hand_qpos: np.ndarray,
    ball_pos: np.ndarray,
    ball_quat: np.ndarray,
) -> None:
    scene_path = LEAP_DIR / "scene.xml"
    backup_path = scene_path.with_suffix(".xml.bak")
    shutil.copy2(scene_path, backup_path)

    qpos_values = np.concatenate((hand_qpos, ball_pos, ball_quat))
    qpos_text = " ".join(f"{value:.9g}" for value in qpos_values)
    ctrl_text = " ".join(f"{value:.9g}" for value in hand_qpos)

    original = scene_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(<key\s+name="home"\s+qpos=")[^"]*("\s+ctrl=")[^"]*("\s*/>)',
        flags=re.DOTALL,
    )
    replacement = rf'\g<1>{qpos_text}\g<2>{ctrl_text}\g<3>'
    updated, count = pattern.subn(replacement, original, count=1)
    if count != 1:
        raise RuntimeError(
            "Could not uniquely locate <key name=\"home\"> in scene.xml."
        )

    scene_path.write_text(updated, encoding="utf-8")
    print(f"Updated: {scene_path}")
    print(f"Backup:  {backup_path}")


def _resolve_seed_bank_path(seed_bank_path: str) -> Path:
    path = Path(seed_bank_path)
    if path.is_absolute():
        return path
    return ROOT / "src" / "unilab" / "assets" / path


def _contact_mask(contacts: np.ndarray) -> int:
    """Return a 4-bit mask in [ff, mf, rf, th] bit order."""
    finger_contacts = np.asarray(contacts[:4] > 0.5, dtype=np.int32)
    return int(
        np.sum(
            finger_contacts
            * (1 << np.arange(4, dtype=np.int32))
        )
    )


def _sustained_contact_mask(
    result: SettledResult,
    threshold: float,
) -> int:
    """Classify contacts using the whole final validation window."""
    active = np.asarray(
        result.finger_contact_fractions >= threshold,
        dtype=np.int32,
    )
    return int(
        np.sum(active * (1 << np.arange(4, dtype=np.int32)))
    )


def _mask_text(mask: int) -> str:
    """Human-readable mask using ff/mf/rf/th names."""
    names = ("ff", "mf", "rf", "th")
    active = [
        name
        for index, name in enumerate(names)
        if mask & (1 << index)
    ]
    return "+".join(active) if active else "none"


def _joint_rms(first: SettledResult, second: SettledResult) -> float:
    return float(
        np.sqrt(np.mean(np.square(first.qpos - second.qpos)))
    )


def _compatible_with_seed0(
    candidate: SettledResult,
    seed0: SettledResult,
    max_joint_delta: float,
) -> bool:
    return bool(
        np.max(np.abs(candidate.qpos - seed0.qpos))
        <= max_joint_delta
    )


def _select_diverse_seeds(
    passing: list[SettledResult],
    num_output_seeds: int,
    min_rms_distance: float,
    max_joint_delta: float,
    contact_fraction_threshold: float,
    min_unique_contact_masks: int,
) -> tuple[list[SettledResult], list[str]]:
    """Select a local seed sequence with strict finger-handoff coverage.

    A complete seed bank must include all of the following:

    1. At least one seed with sustained middle-finger contact.
    2. At least one middle-takeover seed where the middle finger sustains
       contact while the index finger is released.
    3. At least ``min_unique_contact_masks`` sustained contact masks.

    The function never fills missing stages with another copy of the default
    ff+rf+th support pattern. If the requirements cannot be met, it returns a
    partial bank together with explicit missing requirements.
    """
    if not passing:
        return [], ["no stable passing candidates"]

    ordered = sorted(passing, key=lambda result: result.total_score)
    seed0 = ordered[0]
    selected: list[SettledResult] = [seed0]

    def is_selected(candidate: SettledResult) -> bool:
        return any(candidate is chosen for chosen in selected)

    def is_locally_compatible(candidate: SettledResult) -> bool:
        if not _compatible_with_seed0(
            candidate,
            seed0,
            max_joint_delta,
        ):
            return False
        if is_selected(candidate):
            return False
        return True

    def sustained_active(candidate: SettledResult) -> np.ndarray:
        return (
            candidate.finger_contact_fractions
            >= contact_fraction_threshold
        )

    def choose_required(
        predicate,
        *,
        prefer_new_mask: bool = True,
    ) -> SettledResult | None:
        used_masks = {
            _sustained_contact_mask(
                chosen,
                contact_fraction_threshold,
            )
            for chosen in selected
        }
        eligible: list[SettledResult] = []

        for candidate in ordered:
            if not is_locally_compatible(candidate):
                continue
            if not predicate(candidate):
                continue

            # Required stages may be close to seed 0 because a small transition
            # is desirable. They only need to differ measurably.
            if _joint_rms(candidate, seed0) < min(
                min_rms_distance,
                0.025,
            ):
                continue
            eligible.append(candidate)

        if not eligible:
            return None

        def rank(candidate: SettledResult) -> tuple[float, float, float]:
            mask = _sustained_contact_mask(
                candidate,
                contact_fraction_threshold,
            )
            new_mask = 1.0 if mask not in used_masks else 0.0
            distance = min(
                _joint_rms(candidate, chosen)
                for chosen in selected
            )
            return (
                new_mask if prefer_new_mask else 0.0,
                distance,
                -candidate.total_score,
            )

        return max(eligible, key=rank)

    # Stage A: the middle finger must actually participate for most of the
    # validation window.
    middle_seed = choose_required(
        lambda candidate: bool(
            sustained_active(candidate)[1]
            and np.count_nonzero(sustained_active(candidate)) >= 2
        )
    )
    if middle_seed is not None:
        selected.append(middle_seed)

    # Stage B: explicit handoff coverage. The middle finger supports the ball
    # while the index finger is released. At least one additional finger must
    # remain active so this is still a stable grasp.
    middle_takeover_seed = choose_required(
        lambda candidate: bool(
            sustained_active(candidate)[1]
            and not sustained_active(candidate)[0]
            and np.count_nonzero(sustained_active(candidate)) >= 2
        )
    )
    if (
        middle_takeover_seed is not None
        and not is_selected(middle_takeover_seed)
    ):
        selected.append(middle_takeover_seed)

    # Stage C: fill missing unique masks before filling with generic diversity.
    while len(selected) < num_output_seeds:
        used_masks = {
            _sustained_contact_mask(
                chosen,
                contact_fraction_threshold,
            )
            for chosen in selected
        }
        if len(used_masks) >= min_unique_contact_masks:
            break

        candidates: list[SettledResult] = []
        for candidate in ordered:
            if not is_locally_compatible(candidate):
                continue
            mask = _sustained_contact_mask(
                candidate,
                contact_fraction_threshold,
            )
            if mask in used_masks:
                continue
            distances = [
                _joint_rms(candidate, chosen)
                for chosen in selected
            ]
            if min(distances) < min_rms_distance:
                continue
            candidates.append(candidate)

        if not candidates:
            break

        selected.append(
            max(
                candidates,
                key=lambda candidate: (
                    min(
                        _joint_rms(candidate, chosen)
                        for chosen in selected
                    ),
                    -candidate.total_score,
                ),
            )
        )

    # Fill remaining slots, still preferring unseen masks and local diversity.
    while len(selected) < num_output_seeds:
        used_masks = {
            _sustained_contact_mask(
                chosen,
                contact_fraction_threshold,
            )
            for chosen in selected
        }
        best_candidate: SettledResult | None = None
        best_priority: tuple[float, float, float] | None = None

        for candidate in ordered:
            if not is_locally_compatible(candidate):
                continue

            distances = [
                _joint_rms(candidate, chosen)
                for chosen in selected
            ]
            min_distance = min(distances)
            if min_distance < min_rms_distance:
                continue

            mask = _sustained_contact_mask(
                candidate,
                contact_fraction_threshold,
            )
            priority = (
                1.0 if mask not in used_masks else 0.0,
                min_distance,
                -candidate.total_score,
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)

    masks = [
        _sustained_contact_mask(
            result,
            contact_fraction_threshold,
        )
        for result in selected
    ]
    active_arrays = [
        result.finger_contact_fractions
        >= contact_fraction_threshold
        for result in selected
    ]

    missing: list[str] = []
    if not any(active[1] for active in active_arrays):
        missing.append(
            "no seed has sustained middle-finger contact"
        )
    if not any(
        active[1]
        and not active[0]
        and np.count_nonzero(active) >= 2
        for active in active_arrays
    ):
        missing.append(
            "no middle-takeover seed has mf contact with ff released"
        )
    unique_masks = len(set(masks))
    if unique_masks < min_unique_contact_masks:
        missing.append(
            f"only {unique_masks} unique sustained contact masks "
            f"(need {min_unique_contact_masks})"
        )

    return selected, missing


def _save_seed_bank(
    seed_bank_path: str,
    seeds: list[SettledResult],
) -> Path:
    output_path = _resolve_seed_bank_path(seed_bank_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states = np.stack(
        [
            np.concatenate(
                (seed.qpos, seed.ball_pos, seed.ball_quat)
            )
            for seed in seeds
        ],
        axis=0,
    ).astype(np.float32)
    np.save(output_path, states)
    return output_path


def main() -> int:
    args = _parse_args()
    if args.samples < args.rounds:
        raise ValueError("--samples must be at least --rounds.")
    if args.physics_candidates < 1:
        raise ValueError("--physics-candidates must be positive.")
    if args.validation_seconds <= 0.0:
        raise ValueError("--validation-seconds must be positive.")
    if not 0.0 < args.validation_window_seconds <= args.validation_seconds:
        raise ValueError(
            "--validation-window-seconds must be positive and no greater "
            "than --validation-seconds."
        )
    if not 0.0 <= args.min_support_fraction <= 1.0:
        raise ValueError("--min-support-fraction must be in [0, 1].")
    if args.num_output_seeds < 1:
        raise ValueError("--num-output-seeds must be positive.")
    if args.min_seed_rms_distance < 0.0:
        raise ValueError("--min-seed-rms-distance must be non-negative.")
    if args.max_seed_joint_delta <= 0.0:
        raise ValueError("--max-seed-joint-delta must be positive.")
    if not 0.0 < args.contact_fraction_threshold <= 1.0:
        raise ValueError(
            "--contact-fraction-threshold must be in (0, 1]."
        )
    if args.min_unique_contact_masks < 2:
        raise ValueError(
            "--min-unique-contact-masks must be at least 2."
        )
    if args.targeted_physics_candidates < 1:
        raise ValueError(
            "--targeted-physics-candidates must be positive."
        )
    if args.index_min_height_offset >= args.index_max_height_offset:
        raise ValueError(
            "--index-min-height-offset must be smaller than "
            "--index-max-height-offset."
        )

    reference = _extract_allegro_reference()
    model = _load_leap_model()
    data = _load_home(model)

    tip_ids = _geom_ids(model, LEAP_TIP_GEOMS)
    ball_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    palm_geom_id = _name_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "palm_lower_collision",
    )
    palm_body_id = _name_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "palm_lower",
    )
    home_key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    ball_radius = float(model.geom_size[ball_geom_id, 0])

    current_hand = data.qpos[:16].copy()
    current_ball = data.qpos[16:19].copy()
    mapped_hand = _allegro_mapped_seed(model, current_hand, reference)

    lower_hand, upper_hand = _leap_joint_bounds(model)

    # For the three non-thumb fingers:
    # qpos[2, 6, 10] are the middle flexion joints;
    # qpos[3, 7, 11] are the final/distal flexion joints.
    main_flex_indices = np.asarray([2, 6, 10], dtype=np.int32)
    distal_flex_indices = np.asarray([3, 7, 11], dtype=np.int32)
    thumb_indices = np.asarray([12, 13, 14, 15], dtype=np.int32)

    # A natural LEAP thumb opposition family. This does not prescribe the
    # contact surface; it only prevents the optimizer from twisting the thumb
    # to extreme joint combinations in order to touch the ball.
    thumb_reference = np.asarray(
        [1.20, 0.78, 0.85, 0.25],
        dtype=np.float64,
    )
    thumb_scale = np.asarray(
        [0.38, 0.42, 0.55, 0.42],
        dtype=np.float64,
    )
    thumb_search_lower = np.asarray(
        [0.55, -0.05, -0.20, -0.20],
        dtype=np.float64,
    )
    thumb_search_upper = np.asarray(
        [1.65, 1.35, 1.55, 1.15],
        dtype=np.float64,
    )

    if args.distal_soft_limit <= 0.0:
        raise ValueError("--distal-soft-limit must be positive.")
    if args.distal_hard_limit <= args.distal_soft_limit:
        raise ValueError(
            "--distal-hard-limit must be greater than --distal-soft-limit."
        )

    # Do not allow the optimizer to solve the grasp by curling the fingertip
    # joints to roughly 1.8-2.0 rad, which was the main visual mismatch with
    # Allegro.
    upper_hand[distal_flex_indices] = np.minimum(
        upper_hand[distal_flex_indices],
        float(args.distal_hard_limit),
    )

    lower_hand[thumb_indices] = np.maximum(
        lower_hand[thumb_indices],
        thumb_search_lower,
    )
    upper_hand[thumb_indices] = np.minimum(
        upper_hand[thumb_indices],
        thumb_search_upper,
    )
    if np.any(lower_hand[thumb_indices] >= upper_hand[thumb_indices]):
        raise RuntimeError("The natural thumb search bounds are invalid.")

    # Range-normalized Allegro-to-LEAP mapping drives the LEAP thumb close to
    # its joint limits because the two hands use different axes and zero
    # conventions. Use a LEAP-native opposition seed for the thumb instead.
    mapped_hand[thumb_indices] = np.clip(
        thumb_reference,
        lower_hand[thumb_indices],
        upper_hand[thumb_indices],
    )

    ball_delta = float(args.ball_search_radius)
    lower = np.concatenate(
        (
            lower_hand,
            current_ball - ball_delta,
        )
    )
    upper = np.concatenate(
        (
            upper_hand,
            current_ball + ball_delta,
        )
    )

    # Prevent the search from moving the ball so far down that the palm/floor
    # becomes an accidental support.
    lower[18] = max(lower[18], current_ball[2] - 0.010)
    upper[18] = current_ball[2] + ball_delta

    mapped_seed = np.concatenate((mapped_hand, current_ball))
    current_seed = np.concatenate((current_hand, current_ball))
    center = 0.80 * mapped_seed + 0.20 * current_seed
    center = np.clip(center, lower, upper)

    range_width = upper - lower
    sigma = 0.28 * range_width
    sigma[16:19] = min(0.012, ball_delta)

    min_sigma = 0.012 * range_width
    min_sigma[16:19] = 0.0015

    seed_normalized = (mapped_hand - lower_hand) / np.maximum(
        upper_hand - lower_hand,
        1e-9,
    )

    target_gram = reference.direction_gram
    target_pairs = reference.pairwise_distance_by_radius
    target_sorted_gaps = reference.sorted_surface_gaps_by_radius

    def score(vector: np.ndarray) -> CandidateMetrics:
        hand_qpos = vector[:16]
        ball_pos = vector[16:19]

        data.qpos[:16] = hand_qpos
        data.qpos[16:19] = ball_pos
        data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        tips = data.geom_xpos[tip_ids].copy()
        ball = data.geom_xpos[ball_geom_id].copy()
        palm = data.xpos[palm_body_id].copy()

        tip_gaps = _surface_gaps(
            model,
            data,
            tip_ids,
            ball_geom_id,
        )
        palm_gap = float(
            mujoco.mj_geomDistance(
                model,
                data,
                palm_geom_id,
                ball_geom_id,
                0.25,
                None,
            )
        )

        gram = _direction_gram(tips, ball)
        triangle = np.triu_indices(4, k=1)
        gram_error = gram[triangle] - target_gram[triangle]
        layout_rmse = float(np.sqrt(np.mean(np.square(gram_error))))
        layout_penalty = np.square(gram_error / 0.22).sum()

        pairwise = _pairwise_values(tips) / ball_radius
        pairwise_error = pairwise - target_pairs
        pairwise_rmse = float(np.sqrt(np.mean(np.square(pairwise_error))))
        pairwise_penalty = np.square(pairwise_error / 0.35).sum()

        sorted_gaps = np.sort(tip_gaps) / ball_radius
        gap_error = sorted_gaps - target_sorted_gaps
        gap_match = np.square(gap_error / 0.16).sum()

        # All four fingertips stay in the manipulation region. At least two
        # should be almost touching, and a third should be close enough to
        # take over during rotation.
        positive_gap = np.maximum(tip_gaps, 0.0)
        all_near = np.square(
            np.maximum(positive_gap - 0.012, 0.0) / 0.010
        ).sum()
        closest = np.sort(positive_gap)
        two_support = np.square(
            np.maximum(
                closest[:2] - float(args.real_contact_gap),
                0.0,
            )
            / 0.0015
        ).sum()
        third_ready = float(
            np.square(
                max(closest[2] - 0.004, 0.0)
                / 0.004
            )
        )

        penetration = np.square(
            np.maximum(-tip_gaps - 0.0015, 0.0) / 0.0025
        ).sum()

        # Match Allegro's palm clearance, while strongly rejecting a ball
        # that is resting on the LEAP palm.
        target_palm_gap = reference.palm_gap_by_radius * ball_radius
        palm_match = float(
            np.square((palm_gap - target_palm_gap) / 0.008)
        )
        palm_too_close = float(
            np.square(max(0.007 - palm_gap, 0.0) / 0.004)
        )

        mean_tip_reach = float(
            np.mean(np.linalg.norm(tips - palm[None, :], axis=1))
        )
        palm_ball_ratio = float(
            np.linalg.norm(ball - palm) / max(mean_tip_reach, 1e-9)
        )
        palm_ball_error = float(
            np.square(
                (palm_ball_ratio - reference.palm_ball_ratio) / 0.12
            )
        )

        normalized = (hand_qpos - lower_hand) / np.maximum(
            upper_hand - lower_hand,
            1e-9,
        )
        limit_penalty = np.square(
            np.maximum(np.abs(normalized - 0.5) - 0.47, 0.0) / 0.03
        ).sum()

        # Keep the overall pose somewhat Allegro-like, but do not prescribe
        # whether contact uses the pad, tip, or side.
        pose_prior = 0.60 * np.square(
            (normalized - seed_normalized) / 0.40
        ).sum()

        main_flex = hand_qpos[main_flex_indices]
        distal_flex = hand_qpos[distal_flex_indices]

        # The middle flexion joints may bend naturally. Only very straight or
        # excessively curled values are discouraged.
        main_flex_penalty = (
            np.square(
                np.maximum(0.15 - main_flex, 0.0) / 0.25
            ).sum()
            + np.square(
                np.maximum(main_flex - 1.25, 0.0) / 0.30
            ).sum()
        )

        # Allegro's authored grasp uses modest distal flexion. This is only a
        # weak shape preference around 0.30 rad.
        distal_shape_penalty = np.square(
            (distal_flex - 0.30) / 0.45
        ).sum()

        # Strongly reject the previous failure mode where qpos[3], qpos[7],
        # and qpos[11] curled toward 1.8-2.0 rad.
        distal_curl_penalty = np.square(
            np.maximum(
                distal_flex - float(args.distal_soft_limit),
                0.0,
            )
            / 0.18
        ).sum()

        thumb_pose = hand_qpos[thumb_indices]
        thumb_pose_penalty = np.square(
            (thumb_pose - thumb_reference) / thumb_scale
        ).sum()

        # Specifically discourage the visually twisted combination observed
        # previously: excessive base/proximal rotation together with a
        # strongly negative final joint.
        thumb_twist_penalty = (
            np.square(max(thumb_pose[0] - 1.50, 0.0) / 0.15)
            + np.square(max(thumb_pose[1] - 1.15, 0.0) / 0.15)
            + np.square(max(-0.08 - thumb_pose[3], 0.0) / 0.14)
        )

        # Transition-ready neutral pose:
        # - keep the index straighter than the previous default pose;
        # - place the index near the side of the ball rather than underneath it;
        # - keep the middle finger close enough to take over with a small motion.
        index_main = float(hand_qpos[2])
        index_distal = float(hand_qpos[3])
        index_shape_penalty = (
            np.square(
                max(index_main - float(args.index_main_soft_limit), 0.0)
                / 0.20
            )
            + np.square(
                max(index_distal - float(args.index_distal_soft_limit), 0.0)
                / 0.16
            )
        )

        index_height_offset = float(tips[0, 2] - ball[2])
        index_side_height_penalty = (
            np.square(
                max(
                    float(args.index_min_height_offset) - index_height_offset,
                    0.0,
                )
                / 0.006
            )
            + np.square(
                max(
                    index_height_offset - float(args.index_max_height_offset),
                    0.0,
                )
                / 0.008
            )
        )
        index_gap_penalty = np.square(
            max(
                float(tip_gaps[0]) - float(args.index_gap_target),
                0.0,
            )
            / 0.004
        )
        middle_gap_penalty = np.square(
            max(
                float(tip_gaps[1]) - float(args.middle_gap_target),
                0.0,
            )
            / 0.004
        )

        non_tip_ball_penetration = 0.0
        self_penetration = 0.0
        tip_id_set = {int(value) for value in tip_ids}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            distance = float(contact.dist)

            if distance >= -0.001:
                continue

            pair = {first, second}
            if ball_geom_id in pair:
                other = second if first == ball_geom_id else first
                if other not in tip_id_set:
                    non_tip_ball_penetration += float(
                        np.square((-distance - 0.001) / 0.0025)
                    )
            elif first != 0 and second != 0:
                self_penetration += float(
                    np.square((-distance - 0.001) / 0.0025)
                )

        score_value = (
            8.0 * layout_penalty
            + 2.5 * pairwise_penalty
            + 1.5 * gap_match
            + 6.0 * all_near
            + 14.0 * two_support
            + 5.0 * third_ready
            + 14.0 * penetration
            + 1.5 * palm_match
            + 24.0 * palm_too_close
            + 1.0 * palm_ball_error
            + limit_penalty
            + pose_prior
            + 2.0 * main_flex_penalty
            + 0.75 * distal_shape_penalty
            + 12.0 * distal_curl_penalty
            + 3.0 * thumb_pose_penalty
            + 10.0 * thumb_twist_penalty
            + 8.0 * index_shape_penalty
            + 8.0 * index_side_height_penalty
            + 10.0 * index_gap_penalty
            + 8.0 * middle_gap_penalty
            + 18.0 * non_tip_ball_penetration
            + 5.0 * self_penetration
        )

        return CandidateMetrics(
            score=float(score_value),
            tip_gaps=tip_gaps,
            layout_rmse=layout_rmse,
            pairwise_rmse=pairwise_rmse,
            palm_gap=palm_gap,
            palm_ball_ratio=palm_ball_ratio,
        )

    rng = np.random.default_rng(args.seed)
    per_round = max(1, args.samples // args.rounds)
    elite_count = min(64, max(8, per_round // 250))
    top_heap: list[tuple[float, int, np.ndarray]] = []
    middle_bridge_heap: list[tuple[float, int, np.ndarray]] = []
    middle_takeover_heap: list[tuple[float, int, np.ndarray]] = []
    heap_counter = 0

    def push_candidate(
        heap: list[tuple[float, int, np.ndarray]],
        candidate_score: float,
        candidate: np.ndarray,
        capacity: int,
    ) -> None:
        nonlocal heap_counter
        entry = (
            -float(candidate_score),
            heap_counter,
            candidate.copy(),
        )
        heap_counter += 1
        if len(heap) < capacity:
            heapq.heappush(heap, entry)
        elif candidate_score < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    best_vector = center.copy()
    best_metrics = score(best_vector)

    print("Allegro reference loaded automatically.")
    print(
        "target pairwise cosines:",
        np.round(
            reference.direction_gram[np.triu_indices(4, k=1)],
            4,
        ).tolist(),
    )
    print(
        "target sorted tip gaps / radius:",
        np.round(reference.sorted_surface_gaps_by_radius, 4).tolist(),
    )
    print()

    for round_index in range(args.rounds):
        candidates = center + rng.normal(
            size=(per_round, len(center))
        ) * sigma

        if round_index == 0:
            uniform_count = max(1, per_round // 5)
            candidates[:uniform_count, :16] = rng.uniform(
                lower_hand,
                upper_hand,
                size=(uniform_count, 16),
            )
            candidates[:uniform_count, 16:19] = rng.uniform(
                lower[16:19],
                upper[16:19],
                size=(uniform_count, 3),
            )

        candidates = np.clip(candidates, lower, upper)
        candidates[0] = best_vector
        scores = np.empty(per_round, dtype=np.float64)

        for candidate_index, candidate in enumerate(candidates):
            metrics = score(candidate)
            scores[candidate_index] = metrics.score

            if metrics.score < best_metrics.score:
                best_metrics = metrics
                best_vector = candidate.copy()

            push_candidate(
                top_heap,
                metrics.score,
                candidate,
                int(args.physics_candidates),
            )

            # Targeted pool 1: middle finger nearly contacts while index and
            # thumb remain close enough to form a bridge state.
            positive_gaps = np.maximum(metrics.tip_gaps, 0.0)
            bridge_score = (
                metrics.score
                + 18.0
                * np.square(
                    max(
                        float(positive_gaps[1])
                        - float(args.real_contact_gap),
                        0.0,
                    )
                    / 0.004
                )
                + 5.0
                * np.square(
                    max(float(positive_gaps[0]) - 0.008, 0.0)
                    / 0.006
                )
                + 5.0
                * np.square(
                    max(float(positive_gaps[3]) - 0.006, 0.0)
                    / 0.006
                )
            )
            push_candidate(
                middle_bridge_heap,
                bridge_score,
                candidate,
                int(args.targeted_physics_candidates),
            )

            # Targeted pool 2: middle finger is at contact distance while the
            # index is intentionally 4-10 mm away. This creates candidates
            # where the ball can be supported after the index releases.
            target_index_gap = 0.007
            takeover_score = (
                metrics.score
                + 22.0
                * np.square(
                    max(
                        float(positive_gaps[1])
                        - float(args.real_contact_gap),
                        0.0,
                    )
                    / 0.0035
                )
                + 8.0
                * np.square(
                    (
                        float(positive_gaps[0])
                        - target_index_gap
                    )
                    / 0.004
                )
                + 5.0
                * np.square(
                    max(
                        min(
                            float(positive_gaps[2]),
                            float(positive_gaps[3]),
                        )
                        - 0.006,
                        0.0,
                    )
                    / 0.006
                )
            )
            push_candidate(
                middle_takeover_heap,
                takeover_score,
                candidate,
                int(args.targeted_physics_candidates),
            )

        elite_indices = np.argpartition(
            scores,
            elite_count - 1,
        )[:elite_count]
        elites = candidates[elite_indices]
        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0)

        center = 0.25 * center + 0.75 * elite_mean
        sigma = np.maximum(
            0.45 * sigma + 0.55 * elite_std,
            min_sigma,
        )

        print(
            f"round={round_index + 1} "
            f"score={best_metrics.score:.6f} "
            f"tip_gap={np.round(best_metrics.tip_gaps, 5).tolist()} "
            f"layout_rmse={best_metrics.layout_rmse:.4f} "
            f"pair_rmse={best_metrics.pairwise_rmse:.4f} "
            f"palm_gap={best_metrics.palm_gap:.5f} "
            f"ball={np.round(best_vector[16:19], 6).tolist()}"
        )

    candidate_entries = (
        sorted(top_heap, key=lambda item: -item[0])
        + sorted(middle_bridge_heap, key=lambda item: -item[0])
        + sorted(middle_takeover_heap, key=lambda item: -item[0])
    )

    # A candidate may appear in multiple pools. Deduplicate exact numerical
    # states before the expensive long-duration MuJoCo validation.
    top_candidates: list[np.ndarray] = []
    seen_candidates: set[bytes] = set()
    for _, _, candidate in candidate_entries:
        key = np.round(candidate, decimals=9).tobytes()
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        top_candidates.append(candidate)

    print(
        "Candidate pools: "
        f"generic={len(top_heap)}, "
        f"middle_bridge={len(middle_bridge_heap)}, "
        f"middle_takeover={len(middle_takeover_heap)}, "
        f"unique_total={len(top_candidates)}"
    )

    def settle_candidate(vector: np.ndarray) -> SettledResult:
        metrics = score(vector)
        hand_qpos = vector[:16]
        ball_pos = vector[16:19]

        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        data.qpos[:16] = hand_qpos
        data.qpos[16:19] = ball_pos
        data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[:] = 0.0
        data.ctrl[:16] = hand_qpos
        mujoco.mj_forward(model, data)

        initial_z = float(data.qpos[18])

        total_steps = max(
            int(args.settle_steps),
            int(np.ceil(args.validation_seconds / model.opt.timestep)),
        )
        window_steps = max(
            2,
            int(
                np.ceil(
                    args.validation_window_seconds / model.opt.timestep
                )
            ),
        )
        window_start = max(0, total_steps - window_steps)

        recent_z: list[float] = []
        recent_xy: list[np.ndarray] = []
        recent_contact_counts: list[int] = []
        recent_finger_contacts: list[np.ndarray] = []
        recent_palm_contacts: list[bool] = []

        for step_index in range(total_steps):
            mujoco.mj_step(model, data)

            if step_index >= window_start:
                step_sensor_contacts = _contact_values(model, data)

                step_tip_gaps = _surface_gaps(
                    model,
                    data,
                    tip_ids,
                    ball_geom_id,
                )

                step_palm_gap = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        palm_geom_id,
                        ball_geom_id,
                        0.25,
                        None,
                    )
                )

                recent_z.append(float(data.qpos[18]))
                recent_xy.append(data.qpos[16:18].copy())

                # Sensor contact is not enough because MuJoCo margin may create
                # a contact while the visual surfaces are still separated.
                step_finger_contacts = (
                    (step_sensor_contacts[:4] > 0.5)
                    & (
                        step_tip_gaps
                        <= float(args.real_contact_gap)
                    )
                )

                step_palm_contact = bool(
                    step_sensor_contacts[4] > 0.5
                    and step_palm_gap
                    <= float(args.real_contact_gap)
                )

                recent_contact_counts.append(
                    int(np.count_nonzero(step_finger_contacts))
                )
                recent_finger_contacts.append(
                    step_finger_contacts.copy()
                )
                recent_palm_contacts.append(step_palm_contact)

        raw_contacts = _contact_values(model, data)
        settled_gaps = _surface_gaps(
            model,
            data,
            tip_ids,
            ball_geom_id,
        )
        settled_palm_gap = float(
            mujoco.mj_geomDistance(
                model,
                data,
                palm_geom_id,
                ball_geom_id,
                0.25,
                None,
            )
        )

        settled_finger_contacts = (
            (raw_contacts[:4] > 0.5)
            & (
                settled_gaps
                <= float(args.real_contact_gap)
            )
        )

        settled_palm_contact = bool(
            raw_contacts[4] > 0.5
            and settled_palm_gap
            <= float(args.real_contact_gap)
        )

        # Store honest contact results instead of raw margin contacts.
        contacts = np.concatenate(
            (
                settled_finger_contacts.astype(np.float64),
                np.asarray(
                    [float(settled_palm_contact)],
                    dtype=np.float64,
                ),
            )
        )

        final_z = float(data.qpos[18])
        ball_drop = initial_z - final_z

        recent_z_array = np.asarray(recent_z, dtype=np.float64)
        recent_xy_array = np.asarray(recent_xy, dtype=np.float64)
        recent_z_range = (
            float(np.ptp(recent_z_array))
            if recent_z_array.size
            else 0.0
        )
        validation_z_drop = (
            float(recent_z_array[0] - recent_z_array[-1])
            if recent_z_array.size >= 2
            else 0.0
        )
        horizontal_drift = (
            float(
                np.linalg.norm(
                    recent_xy_array[-1] - recent_xy_array[0]
                )
            )
            if len(recent_xy_array) >= 2
            else 0.0
        )
        support_fraction = (
            float(
                np.mean(
                    np.asarray(recent_contact_counts, dtype=np.int32) >= 2
                )
            )
            if recent_contact_counts
            else 0.0
        )
        finger_contact_fractions = (
            np.mean(
                np.asarray(
                    recent_finger_contacts,
                    dtype=np.float64,
                ),
                axis=0,
            )
            if recent_finger_contacts
            else np.zeros(4, dtype=np.float64)
        )
        palm_contact_fraction = (
            float(np.mean(recent_palm_contacts))
            if recent_palm_contacts
            else 1.0
        )

        ball_speed = float(np.linalg.norm(data.qvel[16:19]))
        finger_contact_count = int(
            np.count_nonzero(contacts[:4] > 0.5)
        )
        palm_contact = bool(contacts[4] > 0.5)
        settled_main_flex = data.qpos[main_flex_indices].copy()
        settled_distal_flex = data.qpos[distal_flex_indices].copy()
        settled_thumb_pose = data.qpos[thumb_indices].copy()

        settled_distal_excess = np.maximum(
            settled_distal_flex - float(args.distal_soft_limit),
            0.0,
        )
        settled_distal_penalty = np.square(
            settled_distal_excess / 0.18
        ).sum()
        distal_shape_ok = bool(
            np.max(settled_distal_flex)
            <= float(args.distal_hard_limit) + 0.05
        )

        thumb_twist_score = float(
            np.square(
                max(settled_thumb_pose[0] - 1.50, 0.0) / 0.15
            )
            + np.square(
                max(settled_thumb_pose[1] - 1.15, 0.0) / 0.15
            )
            + np.square(
                max(-0.08 - settled_thumb_pose[3], 0.0) / 0.14
            )
        )
        thumb_bounds_ok = bool(
            np.all(
                settled_thumb_pose
                >= thumb_search_lower - 0.05
            )
            and np.all(
                settled_thumb_pose
                <= thumb_search_upper + 0.05
            )
        )
        thumb_shape_ok = thumb_bounds_ok and thumb_twist_score <= 1.5

        passed = (
            finger_contact_count >= 2
            and bool(contacts[3] > 0.5)
            and (
                finger_contact_fractions[3]
                >= float(args.contact_fraction_threshold)
            )
            and not palm_contact
            and support_fraction >= float(args.min_support_fraction)
            and palm_contact_fraction <= 0.02
            and ball_drop <= 0.012
            and recent_z_range <= 0.003
            and validation_z_drop <= 0.0015
            and horizontal_drift <= 0.003
            and ball_speed <= 0.020
            and float(np.max(settled_gaps)) <= 0.018
            and float(settled_gaps[0]) <= 0.007
            and float(settled_gaps[1]) <= 0.010
            and distal_shape_ok
            and thumb_shape_ok
        )

        physics_penalty = (
            250.0 * np.square(max(ball_drop - 0.003, 0.0) / 0.010)
            + 35.0 * max(2 - finger_contact_count, 0)
            + 45.0 * float(palm_contact)
            + 20.0 * np.square(ball_speed / 0.040)
            + 15.0 * np.square(recent_z_range / 0.003)
            + 20.0
            * np.square(
                max(float(np.max(settled_gaps)) - 0.015, 0.0)
                / 0.010
            )
            + 18.0 * settled_distal_penalty
            + 80.0
            * np.square(
                max(
                    float(args.min_support_fraction) - support_fraction,
                    0.0,
                )
                / 0.20
            )
            + 80.0 * np.square(palm_contact_fraction / 0.05)
            + 80.0
            * np.square(max(validation_z_drop, 0.0) / 0.002)
            + 40.0 * np.square(horizontal_drift / 0.003)
            + 15.0 * thumb_twist_score
        )

        return SettledResult(
            total_score=float(metrics.score + physics_penalty),
            kinematic_score=metrics.score,
            qpos=data.qpos[:16].copy(),
            ball_pos=data.qpos[16:19].copy(),
            ball_quat=data.qpos[19:23].copy(),
            contacts=contacts,
            tip_gaps=settled_gaps,
            palm_gap=settled_palm_gap,
            ball_drop=ball_drop,
            recent_z_range=recent_z_range,
            ball_speed=ball_speed,
            main_flex=settled_main_flex,
            distal_flex=settled_distal_flex,
            thumb_pose=settled_thumb_pose,
            support_fraction=support_fraction,
            finger_contact_fractions=np.asarray(
                finger_contact_fractions,
                dtype=np.float64,
            ),
            palm_contact_fraction=palm_contact_fraction,
            validation_z_drop=validation_z_drop,
            horizontal_drift=horizontal_drift,
            thumb_twist_score=thumb_twist_score,
            passed=passed,
        )

    print()
    print(
        f"Running MuJoCo settling for {len(top_candidates)} candidates..."
    )

    settled_results: list[SettledResult] = []
    for candidate_index, candidate in enumerate(top_candidates):
        result = settle_candidate(candidate)
        settled_results.append(result)
        print(
            f"candidate={candidate_index + 1:02d} "
            f"total={result.total_score:.4f} "
            f"contacts={np.round(result.contacts, 2).tolist()} "
            f"drop_mm={result.ball_drop * 1000:.2f} "
            f"speed={result.ball_speed:.4f} "
            f"support={result.support_fraction:.2f} "
            f"palm_frac={result.palm_contact_fraction:.2f} "
            f"max_distal={np.max(result.distal_flex):.3f} "
            f"thumb_twist={result.thumb_twist_score:.2f} "
            f"palm_gap={result.palm_gap:.5f} "
            f"pass={result.passed}"
        )

    passing = [result for result in settled_results if result.passed]
    if passing:
        selected_seeds, missing_seed_requirements = _select_diverse_seeds(
            passing,
            num_output_seeds=int(args.num_output_seeds),
            min_rms_distance=float(args.min_seed_rms_distance),
            max_joint_delta=float(args.max_seed_joint_delta),
            contact_fraction_threshold=float(
                args.contact_fraction_threshold
            ),
            min_unique_contact_masks=int(
                args.min_unique_contact_masks
            ),
        )
        best_result = selected_seeds[0]
    else:
        selected_seeds = []
        missing_seed_requirements = [
            "no stable passing candidates"
        ]
        best_result = min(
            settled_results,
            key=lambda result: result.total_score,
        )

    print()
    print("===== BEST SETTLED RESULT =====")
    print("status:", "PASS" if best_result.passed else "FAIL")
    print(
        "settled_pose:",
        np.round(best_result.qpos, 6).tolist(),
    )
    print(
        "settled_ball:",
        np.round(best_result.ball_pos, 6).tolist(),
    )
    print(
        "settled_ball_quat:",
        np.round(best_result.ball_quat, 6).tolist(),
    )
    print(
        "settled_contacts [ff,mf,rf,th,palm]:",
        np.round(best_result.contacts, 3).tolist(),
    )
    print(
        "settled_tip_gap:",
        np.round(best_result.tip_gaps, 6).tolist(),
    )
    print(
        "settled_main_flex [q2,q6,q10]:",
        np.round(best_result.main_flex, 6).tolist(),
    )
    print(
        "settled_distal_flex [q3,q7,q11]:",
        np.round(best_result.distal_flex, 6).tolist(),
    )
    print(
        "settled_thumb_pose [q12,q13,q14,q15]:",
        np.round(best_result.thumb_pose, 6).tolist(),
    )
    print(f"thumb_twist_score: {best_result.thumb_twist_score:.4f}")
    print(f"support_fraction: {best_result.support_fraction:.4f}")
    print(
        f"palm_contact_fraction: "
        f"{best_result.palm_contact_fraction:.4f}"
    )
    print(
        f"validation_window_z_drop_mm: "
        f"{best_result.validation_z_drop * 1000:.3f}"
    )
    print(
        f"validation_horizontal_drift_mm: "
        f"{best_result.horizontal_drift * 1000:.3f}"
    )
    print(f"settled_palm_gap: {best_result.palm_gap:.6f}")
    print(f"ball_drop_mm: {best_result.ball_drop * 1000:.3f}")
    print(
        f"last_window_z_range_mm: "
        f"{best_result.recent_z_range * 1000:.3f}"
    )
    print(f"ball_speed: {best_result.ball_speed:.6f}")

    print()
    print("Copy-paste keyframe values:")
    print(
        "qpos=\""
        + " ".join(
            f"{value:.9g}"
            for value in np.concatenate(
                (
                    best_result.qpos,
                    best_result.ball_pos,
                    best_result.ball_quat,
                )
            )
        )
        + "\""
    )
    print(
        "ctrl=\""
        + " ".join(f"{value:.9g}" for value in best_result.qpos)
        + "\""
    )

    print()
    print("===== DIVERSE SEED BANK =====")
    if selected_seeds:
        for seed_index, seed_result in enumerate(selected_seeds):
            rms_from_seed0 = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            seed_result.qpos - selected_seeds[0].qpos
                        )
                    )
                )
            )
            max_delta_from_seed0 = float(
                np.max(
                    np.abs(
                        seed_result.qpos - selected_seeds[0].qpos
                    )
                )
            )
            sustained_mask = _sustained_contact_mask(
                seed_result,
                float(args.contact_fraction_threshold),
            )
            print(
                f"seed={seed_index:02d} "
                f"mask={sustained_mask:04b} "
                f"support={_mask_text(sustained_mask):<11} "
                f"fractions="
                f"{np.round(seed_result.finger_contact_fractions, 2).tolist()} "
                f"score={seed_result.total_score:.4f} "
                f"rms_from_seed0={rms_from_seed0:.4f} "
                f"max_delta_from_seed0={max_delta_from_seed0:.4f} "
                f"gaps={np.round(seed_result.tip_gaps, 5).tolist()}"
            )
    else:
        print("No passing seeds were found.")

    print()
    if missing_seed_requirements:
        print("STRICT SEED COVERAGE: FAIL")
        for requirement in missing_seed_requirements:
            print(f"  - {requirement}")
    else:
        print("STRICT SEED COVERAGE: PASS")
        print(
            "  - middle finger has a sustained-contact seed"
        )
        print(
            "  - a middle-takeover seed exists with index released"
        )
        print(
            f"  - at least {args.min_unique_contact_masks} "
            "unique sustained contact masks exist"
        )

    if args.save_seed_bank:
        can_save_complete_bank = (
            bool(selected_seeds)
            and (
                not missing_seed_requirements
                or bool(args.allow_incomplete_seed_bank)
            )
        )
        if not can_save_complete_bank:
            print(
                "Seed bank was not saved because strict handoff coverage "
                "is incomplete."
            )
            print(
                "Increase --physics-candidates, "
                "--targeted-physics-candidates, or --samples. "
                "Use --allow-incomplete-seed-bank only for debugging."
            )
        else:
            output_path = _save_seed_bank(
                str(args.seed_bank_path),
                selected_seeds,
            )
            print(
                f"Saved seed bank: {output_path}, "
                f"shape=({len(selected_seeds)}, 23)"
            )

    if args.write_scene:
        if best_result.passed or args.force_write:
            _write_home_keyframe(
                best_result.qpos,
                best_result.ball_pos,
                best_result.ball_quat,
            )
        else:
            print()
            print(
                "scene.xml was not changed because the best result failed "
                "the stability check. Use --force-write only for debugging."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
