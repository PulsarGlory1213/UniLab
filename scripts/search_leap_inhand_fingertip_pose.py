"""Create a task-relevant seed bank for LEAP in-hand rotation.

The search starts from one stable grasp but preserves several exploration
branches instead of retaining only the numerically best near-duplicates. It
does not require finger handoff or different contact masks. Saved seeds must
be stable and moderately separated in joint geometry, fingertip surface
location, or safe micro-control response.
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
ASSETS = ROOT / "src" / "unilab" / "assets"
LEAP = ASSETS / "robots" / "leap_hand"

TIP_NAMES = (
    "fingertip_collision",
    "fingertip_2_collision",
    "fingertip_3_collision",
    "thumb_fingertip_collision",
)
SENSOR_NAMES = (
    "leap_ff_contact",
    "leap_mf_contact",
    "leap_rf_contact",
    "leap_th_contact",
    "leap_rotation_palm_contact",
)

# Latest stable pose with real contacts and upright side-thumb support.
LATEST_HAND = np.array([
    1.36443656, 0.10942905, -0.171255665, 0.044722767,
    1.35535491, 0.244312366, -0.35984426, 0.135288809,
    1.11234716, 0.0674886114, 1.02259938, 0.31313848,
    1.44714432, 1.23481299, 0.657685982, 0.0693644554,
])
LATEST_BALL = np.array([-0.0156183066, 0.0361272729, 0.621480116])
LATEST_QUAT = np.array(
    [0.999815531, 0.0169408131, -0.00894579191, 0.00137337012]
)


@dataclass(frozen=True)
class Ids:
    tips: np.ndarray
    ball: int
    palm: int
    thumb_dip: int
    thumb_tip: int
    home: int
    sensors: np.ndarray


@dataclass
class Candidate:
    score: float
    hand: np.ndarray
    ball: np.ndarray
    middle_gap: float = float("inf")


@dataclass
class Seed:
    passed: bool
    score: float
    hand: np.ndarray
    ball: np.ndarray
    quat: np.ndarray
    fractions: np.ndarray
    directions: np.ndarray
    gaps: np.ndarray
    drop: float
    window_drop: float
    drift: float
    speed: float
    thumb_side: float
    thumb_height: float
    thumb_up: float
    probe_signature: np.ndarray
    probe_peak_speed: float
    probe_safe_fraction: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--center-source", choices=("latest", "scene"), default="latest")
    p.add_argument("--samples", type=int, default=160_000)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--physics-candidates", type=int, default=144)
    p.add_argument(
        "--archive-candidates",
        type=int,
        default=1200,
        help="Maximum quick-search cells retained before physics selection.",
    )
    p.add_argument(
        "--archive-joint-bin",
        type=float,
        default=0.035,
        help="Joint-space descriptor bin width used to preserve diversity.",
    )
    p.add_argument(
        "--stable-candidate-fraction",
        type=float,
        default=0.30,
        help=(
            "Fraction of physics candidates reserved for the lowest quick "
            "scores; the remainder is diversity-selected."
        ),
    )
    p.add_argument(
        "--middle-candidate-fraction",
        type=float,
        default=0.30,
        help=(
            "Fraction of physics candidates reserved for poses whose middle "
            "fingertip is close to the ball."
        ),
    )
    p.add_argument(
        "--middle-ready-gap",
        type=float,
        default=0.0015,
        help=(
            "Quick-search middle fingertip gap considered ready for physics "
            "validation, in metres."
        ),
    )
    p.add_argument(
        "--min-middle-seeds",
        type=int,
        default=2,
        help=(
            "Minimum saved seeds with sustained middle-finger contact. "
            "No index release or handoff sequence is required."
        ),
    )
    p.add_argument(
        "--middle-contact-fraction",
        type=float,
        default=0.70,
        help=(
            "Final-window contact fraction required for a seed to count as "
            "middle-engaged."
        ),
    )
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--joint-sigma", type=float, default=0.075)
    p.add_argument("--ball-sigma", type=float, default=0.0010)
    p.add_argument("--max-center-joint-delta", type=float, default=0.38)
    p.add_argument("--max-seed-joint-delta", type=float, default=0.40)
    p.add_argument("--max-seed-ball-delta", type=float, default=0.005)
    p.add_argument(
        "--max-bank-ball-radius",
        type=float,
        default=0.012,
        help=(
            "Maximum Euclidean ball-position distance of every saved seed "
            "from seed 0, in metres."
        ),
    )
    p.add_argument(
        "--saved-thumb-upright-min-cos",
        type=float,
        default=0.72,
        help=(
            "Stricter thumb-upright threshold used only for saved seeds. "
            "The broader physics search may still inspect lower values."
        ),
    )
    p.add_argument(
        "--default-min-contacts",
        type=int,
        default=3,
        help=(
            "Minimum number of sustained fingertip contacts preferred for "
            "seed 0."
        ),
    )
    p.add_argument("--min-seed-rms", type=float, default=0.035)
    p.add_argument("--min-tip-surface-shift", type=float, default=0.003)
    p.add_argument("--min-probe-signature-distance", type=float, default=0.0035)
    p.add_argument("--real-contact-gap", type=float, default=0.00035)
    p.add_argument("--validation-seconds", type=float, default=10.0)
    p.add_argument("--validation-window-seconds", type=float, default=3.0)
    p.add_argument(
        "--max-upward-shift",
        type=float,
        default=0.010,
        help=(
            "Maximum upward ball displacement during settling, in metres. "
            "This prevents compressed middle-finger poses from being accepted "
            "only because they lift the ball."
        ),
    )
    p.add_argument("--min-support-fraction", type=float, default=0.90)
    p.add_argument("--thumb-contact-min-height", type=float, default=-0.012)
    p.add_argument("--thumb-contact-max-height", type=float, default=0.015)
    p.add_argument("--thumb-side-fraction", type=float, default=0.70)
    p.add_argument("--thumb-upright-min-cos", type=float, default=0.05)
    p.add_argument("--probe-seconds", type=float, default=0.35)
    p.add_argument("--probe-delta", type=float, default=0.055)
    p.add_argument("--probe-max-drop", type=float, default=0.004)
    p.add_argument("--probe-max-drift", type=float, default=0.004)
    p.add_argument("--min-probe-peak-speed", type=float, default=0.015)
    p.add_argument("--seed-bank-path", default="caches/leap_hand_seed_bank.npy")
    p.add_argument("--write-scene", action="store_true")
    return p.parse_args()


def check_args(a):
    for name in (
        "samples", "rounds", "physics_candidates",
        "archive_candidates", "archive_joint_bin", "num_seeds",
        "middle_ready_gap", "min_middle_seeds",
        "joint_sigma", "ball_sigma", "max_center_joint_delta",
        "max_seed_joint_delta", "max_seed_ball_delta",
        "max_bank_ball_radius", "default_min_contacts",
        "validation_seconds", "validation_window_seconds",
        "max_upward_shift", "probe_seconds", "probe_delta",
    ):
        if getattr(a, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if a.validation_window_seconds > a.validation_seconds:
        raise ValueError("validation window cannot exceed validation time")
    if not 0.0 < a.stable_candidate_fraction < 1.0:
        raise ValueError("--stable-candidate-fraction must be in (0, 1)")
    if not 0.0 < a.middle_candidate_fraction < 1.0:
        raise ValueError("--middle-candidate-fraction must be in (0, 1)")
    if (
        a.stable_candidate_fraction
        + a.middle_candidate_fraction
        >= 0.90
    ):
        raise ValueError(
            "--stable-candidate-fraction + --middle-candidate-fraction "
            "must be below 0.90"
        )
    if a.min_middle_seeds > a.num_seeds:
        raise ValueError("--min-middle-seeds cannot exceed --num-seeds")
    if not -1.0 <= a.saved_thumb_upright_min_cos <= 1.0:
        raise ValueError(
            "--saved-thumb-upright-min-cos must be in [-1, 1]"
        )
    if not 1 <= a.default_min_contacts <= 4:
        raise ValueError("--default-min-contacts must be in [1, 4]")
    if not 0.0 <= a.middle_contact_fraction <= 1.0:
        raise ValueError("--middle-contact-fraction must be in [0, 1]")
    if not 0 < a.real_contact_gap <= 0.001:
        raise ValueError("--real-contact-gap must be in (0, 0.001]")
    if a.thumb_contact_min_height >= a.thumb_contact_max_height:
        raise ValueError("invalid thumb contact height band")


def name_id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(value)


def load_model():
    xml = Path(materialize_scene_fragments(
        str(LEAP / "leap_hand.xml"),
        fragment_files=[str(LEAP / "scene.xml")],
    ))
    try:
        return mujoco.MjModel.from_xml_path(str(xml))
    finally:
        xml.unlink(missing_ok=True)


def resolve_ids(model):
    return Ids(
        tips=np.array([
            name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in TIP_NAMES
        ], dtype=np.int32),
        ball=name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball"),
        palm=name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm_lower_collision"),
        thumb_dip=name_id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb_dip"),
        thumb_tip=name_id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb_fingertip"),
        home=name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home"),
        sensors=np.array([
            name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in SENSOR_NAMES
        ], dtype=np.int32),
    )


def joint_limits(model):
    low = np.full(16, -np.inf)
    high = np.full(16, np.inf)
    for jid in range(model.njnt):
        address = int(model.jnt_qposadr[jid])
        if 0 <= address < 16:
            low[address], high[address] = model.jnt_range[jid]
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise RuntimeError("Could not resolve all LEAP joint limits")
    return low, high


def center_state(model, ids, source):
    if source == "latest":
        return LATEST_HAND.copy(), LATEST_BALL.copy(), LATEST_QUAT.copy()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, ids.home)
    mujoco.mj_forward(model, data)
    return data.qpos[:16].copy(), data.qpos[16:19].copy(), data.qpos[19:23].copy()


def apply_state(model, data, hand, ball, quat):
    mujoco.mj_resetData(model, data)
    data.qpos[:16] = hand
    data.qpos[16:19] = ball
    norm = np.linalg.norm(quat)
    data.qpos[19:23] = quat / norm if norm > 1e-12 else [1, 0, 0, 0]
    data.qvel[:] = 0
    if model.nu >= 16:
        data.ctrl[:16] = hand
    mujoco.mj_forward(model, data)


def sensor_values(model, data, ids):
    return np.array([
        data.sensordata[int(model.sensor_adr[sid])]
        for sid in ids.sensors
    ])


def gaps(model, data, geom_ids: Iterable[int], ball_id):
    return np.array([
        mujoco.mj_geomDistance(model, data, int(gid), ball_id, 0.25, None)
        for gid in geom_ids
    ])


def palm_gap(model, data, ids):
    return float(mujoco.mj_geomDistance(
        model, data, ids.palm, ids.ball, 0.25, None
    ))


def thumb_up(data, ids):
    axis = data.xpos[ids.thumb_tip] - data.xpos[ids.thumb_dip]
    return float(axis[2] / max(np.linalg.norm(axis), 1e-9))


def thumb_contact_heights(data, ids):
    values = []
    center_z = float(data.geom_xpos[ids.ball, 2])
    thumb_geom = int(ids.tips[3])
    for i in range(data.ncon):
        contact = data.contact[i]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair == {thumb_geom, ids.ball}:
            values.append(float(contact.pos[2]) - center_z)
    return np.asarray(values)


def directions(data, ids):
    vec = data.geom_xpos[ids.tips] - data.geom_xpos[ids.ball][None, :]
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


def quick_evaluate(
    model,
    data,
    ids,
    hand,
    ball,
    quat,
    base_hand,
    base_ball,
):
    apply_state(model, data, hand, ball, quat)
    signed_gaps = gaps(model, data, ids.tips, ids.ball)
    d = np.maximum(signed_gaps, 0)
    closest = np.sort(d)
    move = np.mean(((hand - base_hand) / 0.08) ** 2)
    ball_move = np.mean(((ball - base_ball) / 0.0015) ** 2)
    support = np.sum(
        (np.maximum(closest[:2] - 0.0008, 0) / 0.0015) ** 2
    )
    third = (max(float(closest[2]) - 0.005, 0) / 0.004) ** 2
    thumb = (max(float(d[3]) - 0.0008, 0) / 0.0015) ** 2
    palm = (
        max(0.030 - palm_gap(model, data, ids), 0) / 0.010
    ) ** 2
    upright = (
        max(0.35 - thumb_up(data, ids), 0) / 0.25
    ) ** 2
    distal = np.sum(
        (
            np.maximum(hand[[3, 7, 11]] - 0.85, 0)
            / 0.20
        ) ** 2
    )
    thumb_distal = (
        max(float(hand[15]) - 0.70, 0) / 0.18
    ) ** 2

    score = float(
        1.5 * move
        + ball_move
        + 35 * support
        + 8 * third
        + 28 * thumb
        + 30 * palm
        + 8 * upright
        + 12 * distal
        + 10 * thumb_distal
    )
    return score, float(signed_gaps[1])


def quick_score(
    model,
    data,
    ids,
    hand,
    ball,
    quat,
    base_hand,
    base_ball,
):
    score, _ = quick_evaluate(
        model,
        data,
        ids,
        hand,
        ball,
        quat,
        base_hand,
        base_ball,
    )
    return score


def make_noise(rng, sigma, ball_sigma):
    hand = rng.normal(0, sigma, 16)
    mode = int(rng.integers(0, 7))

    if mode in (1, 2, 3):
        block = (slice(0, 4), slice(4, 8), slice(8, 12))[mode - 1]
        hand *= 0.28
        hand[block] = rng.normal(0, 1.45 * sigma, 4)
    elif mode == 4:
        hand *= 0.30
        phase = rng.normal(0, sigma, 4)
        hand[0:4] += phase
        hand[4:8] -= 0.70 * phase
    elif mode == 5:
        hand *= 0.30
        phase = rng.normal(0, sigma, 4)
        hand[4:8] += phase
        hand[8:12] -= 0.70 * phase
    elif mode == 6:
        hand *= 0.25
        hand[12:16] = rng.normal(0, 0.60 * sigma, 4)

    hand[[3, 7, 11, 15]] *= 0.55
    return hand, rng.normal(0, ball_sigma, 3), mode


def push_heap(heap, counter, candidate, capacity):
    entry = (-candidate.score, counter, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif candidate.score < -heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def middle_candidate_score(item, ready_gap):
    """Prefer middle contact while retaining general grasp quality."""
    positive_gap = max(item.middle_gap, 0.0)
    gap_penalty = (
        max(positive_gap - ready_gap, 0.0) / 0.0025
    ) ** 2
    penetration_penalty = (
        max(-0.0010 - item.middle_gap, 0.0) / 0.0010
    ) ** 2
    return float(
        0.20 * item.score
        + 45.0 * gap_penalty
        + 20.0 * penetration_penalty
    )


def push_middle_heap(
    heap,
    counter,
    candidate,
    capacity,
    ready_gap,
):
    metric = middle_candidate_score(candidate, ready_gap)
    entry = (-metric, counter, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif metric < -heap[0][0]:
        heapq.heapreplace(heap, entry)
    return counter + 1


def anchor_centers(base_hand, base_ball, low, high):
    """Create moderate task-oriented branches around the stable grasp."""
    centers = [(base_hand.copy(), base_ball.copy())]
    offsets = []

    for first, second in ((1, 2), (5, 6), (9, 10)):
        for sign in (-1.0, 1.0):
            delta = np.zeros(16)
            delta[first] = sign * 0.10
            delta[second] = sign * 0.06
            offsets.append(delta)

    for block_a, block_b in ((0, 4), (4, 8), (0, 8)):
        for sign in (-1.0, 1.0):
            delta = np.zeros(16)
            delta[block_a + 1] = sign * 0.075
            delta[block_a + 2] = sign * 0.045
            delta[block_b + 1] = -sign * 0.055
            delta[block_b + 2] = -sign * 0.035
            offsets.append(delta)

    for sign in (-1.0, 1.0):
        delta = np.zeros(16)
        delta[13] = sign * 0.040
        delta[14] = -sign * 0.035
        offsets.append(delta)

    # Dedicated middle-finger exploration. These are only search centers;
    # physical validation still decides whether the resulting grasp is safe.
    for amount in (-0.24, -0.16, -0.10, 0.10, 0.16, 0.24):
        delta = np.zeros(16)
        delta[5] = amount
        delta[6] = 0.70 * amount
        delta[7] = 0.25 * amount
        offsets.append(delta)

    for amount in (-0.18, -0.12, 0.12, 0.18):
        delta = np.zeros(16)
        delta[4] = 0.50 * amount
        delta[5] = amount
        delta[6] = -0.45 * amount
        offsets.append(delta)

    for delta in offsets:
        centers.append((
            np.clip(base_hand + delta, low, high),
            base_ball.copy(),
        ))
    return centers


def archive_key(hand, ball, base_hand, base_ball, joint_bin):
    delta = hand - base_hand
    descriptor = np.array([
        np.mean(delta[0:4]),
        np.mean(delta[4:8]),
        np.mean(delta[8:12]),
        np.mean(delta[12:16]),
        delta[1], delta[5], delta[9],
        delta[2], delta[6], delta[10],
        (ball[0] - base_ball[0]) / 0.001,
        (ball[1] - base_ball[1]) / 0.001,
        (ball[2] - base_ball[2]) / 0.001,
    ])
    widths = np.array(
        [joint_bin] * 10 + [1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    return tuple(np.floor(descriptor / widths).astype(np.int32))


def quick_hand_distance(first, second):
    return float(
        np.sqrt(np.mean(np.square(first.hand - second.hand)))
    )


def archive_group(item, base_hand, shell_width=0.025):
    """Group by RMS shell, dominant finger block, and motion sign."""
    delta = item.hand - base_hand
    joint_rms = float(np.sqrt(np.mean(np.square(delta))))
    shell = int(np.floor(joint_rms / shell_width))

    block_energy = np.array([
        np.sqrt(np.mean(np.square(delta[0:4]))),
        np.sqrt(np.mean(np.square(delta[4:8]))),
        np.sqrt(np.mean(np.square(delta[8:12]))),
        np.sqrt(np.mean(np.square(delta[12:16]))),
    ])
    dominant = int(np.argmax(block_energy))
    begin = 4 * dominant
    sign = 1 if float(np.sum(delta[begin:begin + 4])) >= 0.0 else -1
    return shell, dominant, sign


def prune_archive_stratified(archive, base_hand, capacity):
    """Cap memory without deleting every moderate-distance branch."""
    if len(archive) <= capacity:
        return archive

    grouped = {}
    for key, item in archive.items():
        group = archive_group(item, base_hand)
        grouped.setdefault(group, []).append((key, item))

    for values in grouped.values():
        values.sort(key=lambda pair: pair[1].score)

    groups = sorted(grouped)
    retained = {}
    depth = 0

    # Round-robin through joint-distance/finger/sign regions.
    while len(retained) < capacity:
        added = False
        for group in groups:
            values = grouped[group]
            if depth >= len(values):
                continue
            key, item = values[depth]
            retained[key] = item
            added = True
            if len(retained) >= capacity:
                break
        if not added:
            break
        depth += 1

    return retained


def select_spaced_stable(ordered, count, minimum_rms=0.010):
    """Keep low-score poses without wasting slots on near duplicates."""
    selected = []

    for item in ordered:
        if not selected:
            selected.append(item)
        else:
            nearest = min(
                quick_hand_distance(item, old)
                for old in selected
            )
            if nearest >= minimum_rms:
                selected.append(item)

        if len(selected) >= count:
            return selected

    # Fall back only when the low-score region is extremely narrow.
    for item in ordered:
        if any(item is old for old in selected):
            continue
        selected.append(item)
        if len(selected) >= count:
            break

    return selected


def preselect_physics_candidates(
    candidates,
    count,
    stable_fraction,
    middle_fraction,
    middle_ready_gap,
):
    """Mix stable, middle-ready, and globally diverse poses."""
    if len(candidates) <= count:
        return sorted(candidates, key=lambda item: item.score)

    ordered = sorted(candidates, key=lambda item: item.score)
    stable_count = max(
        1,
        min(count - 2, int(round(count * stable_fraction))),
    )
    middle_count = max(
        1,
        min(
            count - stable_count - 1,
            int(round(count * middle_fraction)),
        ),
    )

    selected = select_spaced_stable(
        ordered,
        stable_count,
        minimum_rms=0.010,
    )
    selected_ids = {id(item) for item in selected}

    middle_ordered = sorted(
        (
            item for item in candidates
            if id(item) not in selected_ids
        ),
        key=lambda item: middle_candidate_score(
            item,
            middle_ready_gap,
        ),
    )

    middle_selected = []
    # Use a little more spacing here so the middle quota is not filled by
    # several copies of one middle-ready pose.
    for item in middle_ordered:
        if (
            item.middle_gap
            > max(0.004, 2.5 * middle_ready_gap)
        ):
            continue
        if middle_selected:
            nearest = min(
                quick_hand_distance(item, old)
                for old in middle_selected
            )
            if nearest < 0.015:
                continue
        middle_selected.append(item)
        if len(middle_selected) >= middle_count:
            break

    selected.extend(middle_selected)
    selected_ids = {id(item) for item in selected}
    remaining = [
        item for item in ordered
        if id(item) not in selected_ids
    ]

    slots = count - len(selected)
    if slots <= 0:
        return selected[:count]

    best_score = ordered[0].score
    feasible = [
        item
        for item in remaining
        if item.score <= best_score + 60.0
    ]
    if len(feasible) < slots:
        feasible = remaining

    if feasible:
        hands = np.stack(
            [item.hand for item in feasible],
            axis=0,
        )
        nearest = np.full(
            len(feasible),
            np.inf,
            dtype=np.float64,
        )

        for old in selected:
            distances = np.sqrt(
                np.mean(
                    np.square(hands - old.hand[None, :]),
                    axis=1,
                )
            )
            nearest = np.minimum(nearest, distances)

        scores = np.asarray(
            [item.score for item in feasible],
            dtype=np.float64,
        )
        active = np.ones(len(feasible), dtype=bool)

        while len(selected) < count and np.any(active):
            middle_bonus = np.asarray(
                [
                    max(
                        middle_ready_gap - item.middle_gap,
                        0.0,
                    )
                    / max(middle_ready_gap, 1e-9)
                    for item in feasible
                ],
                dtype=np.float64,
            )
            priority = (
                nearest / 0.050
                - 0.018 * np.log1p(np.maximum(scores, 0.0))
                + 0.08 * middle_bonus
            )
            priority[~active] = -np.inf
            index = int(np.argmax(priority))
            chosen = feasible[index]
            selected.append(chosen)
            active[index] = False

            distances = np.sqrt(
                np.mean(
                    np.square(hands - chosen.hand[None, :]),
                    axis=1,
                )
            )
            nearest = np.minimum(nearest, distances)

    if len(selected) < count:
        used = {id(item) for item in selected}
        selected.extend(
            item
            for item in ordered
            if id(item) not in used
        )

    ready_count = sum(
        item.middle_gap <= middle_ready_gap
        for item in selected
    )
    print(
        "middle_preselection: "
        f"requested={middle_count} "
        f"ready_selected={ready_count} "
        f"ready_gap_mm={1000 * middle_ready_gap:.2f}"
    )
    return selected[:count]



def search_candidates(model, ids, a, base_hand, base_ball, quat, low, high):
    rng = np.random.default_rng(a.seed)
    data = mujoco.MjData(model)

    anchors = anchor_centers(base_hand, base_ball, low, high)
    centers = anchors.copy()
    stability_heap = []
    middle_heap = []
    archive = {}
    counter = 0

    def retain(item):
        nonlocal counter
        counter = push_heap(
            stability_heap,
            counter,
            item,
            max(24, int(a.physics_candidates * 0.45)),
        )
        key = archive_key(
            item.hand,
            item.ball,
            base_hand,
            base_ball,
            a.archive_joint_bin,
        )
        old = archive.get(key)
        if old is None or item.score < old.score:
            archive[key] = item

    for hand, ball in anchors:
        score_value, middle_gap = quick_evaluate(
            model,
            data,
            ids,
            hand,
            ball,
            quat,
            base_hand,
            base_ball,
        )
        item = Candidate(
            score_value,
            hand.copy(),
            ball.copy(),
            middle_gap,
        )
        retain(item)
        counter = push_middle_heap(
            middle_heap,
            counter,
            item,
            max(48, int(a.physics_candidates * 0.55)),
            a.middle_ready_gap,
        )

    per_round = max(1, a.samples // a.rounds)

    for round_index in range(a.rounds):
        factor = 0.68 ** round_index
        sigma = max(0.012, a.joint_sigma * factor)
        ball_sigma = max(0.00020, a.ball_sigma * factor)
        local_heaps = [[] for _ in range(7)]
        overall_local = []
        middle_local = []

        for _ in range(per_round):
            center_hand, center_ball = centers[
                int(rng.integers(0, len(centers)))
            ]
            hand_noise, ball_noise, mode = make_noise(
                rng, sigma, ball_sigma
            )
            hand = np.clip(center_hand + hand_noise, low, high)
            hand = np.clip(
                hand,
                base_hand - a.max_center_joint_delta,
                base_hand + a.max_center_joint_delta,
            )
            ball = np.clip(
                center_ball + ball_noise,
                base_ball - 0.004,
                base_ball + 0.004,
            )
            score_value, middle_gap = quick_evaluate(
                model,
                data,
                ids,
                hand,
                ball,
                quat,
                base_hand,
                base_ball,
            )
            item = Candidate(
                score_value,
                hand.copy(),
                ball.copy(),
                middle_gap,
            )
            retain(item)
            counter = push_middle_heap(
                middle_heap,
                counter,
                item,
                max(64, int(a.physics_candidates * 0.70)),
                a.middle_ready_gap,
            )
            counter = push_middle_heap(
                middle_local,
                counter,
                item,
                24,
                a.middle_ready_gap,
            )
            counter = push_heap(
                local_heaps[mode],
                counter,
                item,
                8,
            )
            counter = push_heap(
                overall_local,
                counter,
                item,
                24,
            )

        # Keep permanent anchors and the best branch-specific centers.
        centers = anchors.copy()
        for heap in local_heaps:
            entries = sorted(heap, key=lambda entry: -entry[0])
            centers.extend(
                (entry[2].hand.copy(), entry[2].ball.copy())
                for entry in entries[:3]
            )
        entries = sorted(overall_local, key=lambda entry: -entry[0])
        centers.extend(
            (entry[2].hand.copy(), entry[2].ball.copy())
            for entry in entries[:8]
        )
        middle_entries = sorted(
            middle_local,
            key=lambda entry: -entry[0],
        )
        centers.extend(
            (entry[2].hand.copy(), entry[2].ball.copy())
            for entry in middle_entries[:8]
        )

        if entries:
            middle_gap_text = (
                f"{1000 * middle_entries[0][2].middle_gap:.2f}"
                if middle_entries
                else "nan"
            )
            print(
                f"round={round_index + 1} "
                f"quick_score={entries[0][2].score:.4f} "
                f"joint_sigma={sigma:.4f} "
                f"ball_sigma_mm={ball_sigma * 1000:.2f} "
                f"archive_cells={len(archive)} "
                f"best_middle_gap_mm={middle_gap_text}"
            )


        # Preserve moderate-distance branches instead of globally sorting
        # every archive cell by quick score.
        if len(archive) > 4 * a.archive_candidates:
            archive = prune_archive_stratified(
                archive,
                base_hand,
                2 * a.archive_candidates,
            )

    combined = list(archive.values())
    combined.extend(entry[2] for entry in stability_heap)
    combined.extend(entry[2] for entry in middle_heap)

    for hand, ball in anchors:
        score_value, middle_gap = quick_evaluate(
            model,
            data,
            ids,
            hand,
            ball,
            quat,
            base_hand,
            base_ball,
        )
        combined.append(
            Candidate(
                score_value,
                hand.copy(),
                ball.copy(),
                middle_gap,
            )
        )

    unique = {}
    for item in combined:
        key = np.round(np.r_[item.hand, item.ball], 8).tobytes()
        old = unique.get(key)
        if old is None or item.score < old.score:
            unique[key] = item

    result = preselect_physics_candidates(
        list(unique.values()),
        a.physics_candidates,
        a.stable_candidate_fraction,
        a.middle_candidate_fraction,
        a.middle_ready_gap,
    )

    if result:
        pairwise = []
        root_distances = []
        root = result[0]

        for index, item in enumerate(result):
            root_distances.append(quick_hand_distance(item, root))
            if index == 0:
                continue
            pairwise.append(min(
                quick_hand_distance(item, old)
                for old in result[:index]
            ))

        print(
            "physics_preselection: "
            f"archive_unique={len(unique)} "
            f"selected={len(result)} "
            f"nearest_rms_mean="
            f"{np.mean(pairwise) if pairwise else 0.0:.4f} "
            f"nearest_rms_max="
            f"{np.max(pairwise) if pairwise else 0.0:.4f} "
            f"root_rms_max={np.max(root_distances):.4f}"
        )
    return result


def validate_candidate(model, ids, item, quat, a):
    data = mujoco.MjData(model)
    apply_state(model, data, item.hand, item.ball, quat)
    initial_z = float(data.qpos[18])
    steps = max(1, int(np.ceil(a.validation_seconds / model.opt.timestep)))
    window = max(2, int(np.ceil(a.validation_window_seconds / model.opt.timestep)))
    start = max(0, steps - window)

    z_log, xy_log, count_log = [], [], []
    finger_log, palm_log, side_log, height_log = [], [], [], []

    for step in range(steps):
        mujoco.mj_step(model, data)
        if step < start:
            continue

        raw = sensor_values(model, data, ids)
        current_gaps = gaps(model, data, ids.tips, ids.ball)
        fingers = (raw[:4] > 0.5) & (current_gaps <= a.real_contact_gap)
        palm_contact = bool(
            raw[4] > 0.5
            and palm_gap(model, data, ids) <= a.real_contact_gap
        )
        heights = thumb_contact_heights(data, ids)
        side = False
        if fingers[3] and heights.size:
            height = float(heights[np.argmin(np.abs(heights))])
            height_log.append(height)
            side = a.thumb_contact_min_height <= height <= a.thumb_contact_max_height

        z_log.append(float(data.qpos[18]))
        xy_log.append(data.qpos[16:18].copy())
        count_log.append(np.count_nonzero(fingers))
        finger_log.append(fingers)
        palm_log.append(palm_contact)
        side_log.append(side)

    raw = sensor_values(model, data, ids)
    final_gaps = gaps(model, data, ids.tips, ids.ball)
    final_fingers = (raw[:4] > 0.5) & (final_gaps <= a.real_contact_gap)
    final_palm = bool(
        raw[4] > 0.5
        and palm_gap(model, data, ids) <= a.real_contact_gap
    )

    z = np.asarray(z_log)
    xy = np.asarray(xy_log)
    fractions = np.mean(np.asarray(finger_log, float), axis=0)
    support = float(np.mean(np.asarray(count_log) >= 2))
    palm_fraction = float(np.mean(palm_log))
    side_fraction = float(np.mean(side_log))
    contact_height = float(np.mean(height_log)) if height_log else np.nan
    drop = initial_z - float(data.qpos[18])
    window_drop = float(z[0] - z[-1])
    drift = float(np.linalg.norm(xy[-1] - xy[0]))
    speed = float(np.linalg.norm(data.qvel[16:19]))
    upright = thumb_up(data, ids)

    passed = bool(
        np.count_nonzero(final_fingers) >= 2
        and final_fingers[3]
        and not final_palm
        and support >= a.min_support_fraction
        and fractions[3] >= 0.70
        and palm_fraction <= 0.02
        and side_fraction >= a.thumb_side_fraction
        and upright >= a.thumb_upright_min_cos
        and -a.max_upward_shift <= drop <= 0.012
        and np.ptp(z) <= 0.003
        and window_drop <= 0.0015
        and drift <= 0.003
        and speed <= 0.020
    )
    score = float(
        item.score
        + 120 * (max(drop, 0) / 0.010) ** 2
        + 100
        * (
            max(-drop - a.max_upward_shift, 0)
            / max(a.max_upward_shift, 1e-9)
        ) ** 2
        + 100 * (max(window_drop, 0) / 0.0015) ** 2
        + 60 * (drift / 0.003) ** 2
        + (0 if passed else 500)
    )
    return Seed(
        passed=passed,
        score=score,
        hand=data.qpos[:16].copy(),
        ball=data.qpos[16:19].copy(),
        quat=data.qpos[19:23].copy(),
        fractions=fractions,
        directions=directions(data, ids),
        gaps=final_gaps,
        drop=drop,
        window_drop=window_drop,
        drift=drift,
        speed=speed,
        thumb_side=side_fraction,
        thumb_height=contact_height,
        thumb_up=upright,
        probe_signature=np.zeros(18, dtype=np.float64),
        probe_peak_speed=0.0,
        probe_safe_fraction=0.0,
    )



def run_micro_probes(model, ids, seed, a):
    """Measure safe ball angular response to six small finger commands."""
    low, high = joint_limits(model)
    patterns = []

    for block in ((1, 2), (5, 6), (9, 10)):
        for sign in (1.0, -1.0):
            delta = np.zeros(16, dtype=np.float64)
            delta[block[0]] = sign * a.probe_delta
            delta[block[1]] = sign * 0.65 * a.probe_delta
            patterns.append(delta)

    signature = []
    safe_count = 0
    peak_speed = 0.0
    steps = max(2, int(np.ceil(a.probe_seconds / model.opt.timestep)))
    collect_start = steps // 2

    for delta in patterns:
        data = mujoco.MjData(model)
        apply_state(model, data, seed.hand, seed.ball, seed.quat)
        initial_ball = data.qpos[16:19].copy()

        target = np.clip(seed.hand + delta, low, high)
        if model.nu >= 16:
            data.ctrl[:16] = target

        angular_log = []
        for step in range(steps):
            mujoco.mj_step(model, data)
            if step >= collect_start:
                angular_log.append(data.qvel[19:22].copy())

        final_gaps = gaps(model, data, ids.tips, ids.ball)
        raw = sensor_values(model, data, ids)
        real_fingers = (
            (raw[:4] > 0.5)
            & (final_gaps <= a.real_contact_gap)
        )
        drop = float(initial_ball[2] - data.qpos[18])
        drift = float(
            np.linalg.norm(data.qpos[16:18] - initial_ball[:2])
        )
        safe = bool(
            np.count_nonzero(real_fingers) >= 2
            and real_fingers[3]
            and drop <= a.probe_max_drop
            and drift <= a.probe_max_drift
        )

        mean_angular = (
            np.mean(np.asarray(angular_log), axis=0)
            if angular_log
            else np.zeros(3, dtype=np.float64)
        )
        if safe:
            safe_count += 1
            peak_speed = max(
                peak_speed,
                float(np.max(np.linalg.norm(angular_log, axis=1)))
                if angular_log else 0.0,
            )
            signature.extend(mean_angular.tolist())
        else:
            signature.extend([0.0, 0.0, 0.0])

    seed.probe_signature = np.asarray(signature, dtype=np.float64)
    seed.probe_peak_speed = peak_speed
    seed.probe_safe_fraction = safe_count / len(patterns)


def tip_surface_shift(a, b, radius):
    dots = np.sum(a.directions * b.directions, axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    return float(radius * np.max(angles))


def probe_distance(a, b):
    return float(
        np.sqrt(
            np.mean(
                np.square(a.probe_signature - b.probe_signature)
            )
        )
    )


def rms(a, b):
    return float(np.sqrt(np.mean((a.hand - b.hand) ** 2)))


def feature_distance(a, b, radius, args):
    joint = rms(a, b) / max(args.min_seed_rms, 1e-9)
    tip = (
        tip_surface_shift(a, b, radius)
        / max(args.min_tip_surface_shift, 1e-9)
    )
    probe = (
        probe_distance(a, b)
        / max(args.min_probe_signature_distance, 1e-9)
    )
    contact = float(
        np.sqrt(np.mean(np.square(a.fractions - b.fractions)))
    )
    return float(
        0.40 * joint
        + 0.35 * tip
        + 0.20 * probe
        + 0.05 * contact
    )


def sustained_contact_count(seed, fraction_threshold):
    return int(
        np.count_nonzero(
            seed.fractions >= fraction_threshold
        )
    )


def default_seed_cost(seed, args):
    """Lower is better for the main reset/default grasp."""
    contact_count = sustained_contact_count(
        seed,
        args.middle_contact_fraction,
    )
    contact_shortfall = max(
        args.default_min_contacts - contact_count,
        0,
    )
    middle_bonus = float(
        seed.fractions[1] >= args.middle_contact_fraction
    )

    return float(
        12.0 * contact_shortfall
        - 1.8 * middle_bonus
        - 3.0 * seed.thumb_up
        - 55.0 * seed.probe_peak_speed
        + 0.35 * abs(seed.drop) / 0.001
        + 0.20 * seed.drift / 0.001
        + 0.003 * seed.score
    )


def choose_seeds(passing, model, ids, args):
    if not passing:
        return []

    radius = float(model.geom_size[ids.ball, 0])

    # Final cache quality is intentionally stricter than broad search
    # acceptance. Low-thumb states may help connect the search but cannot be
    # written to the seed bank.
    eligible_seeds = [
        item
        for item in passing
        if item.thumb_up >= args.saved_thumb_upright_min_cos
    ]

    print(
        "saved_seed_filter: "
        f"passing={len(passing)} "
        f"thumb_eligible={len(eligible_seeds)} "
        f"thumb_min={args.saved_thumb_upright_min_cos:.2f}"
    )

    if len(eligible_seeds) < args.num_seeds:
        print(
            "saved_seed_filter: FAIL — not enough stable candidates meet "
            "the saved-seed thumb-upright threshold."
        )
        return []

    count = len(eligible_seeds)
    middle_flags = np.asarray(
        [
            item.fractions[1] >= args.middle_contact_fraction
            for item in eligible_seeds
        ],
        dtype=bool,
    )
    hands = np.stack(
        [item.hand for item in eligible_seeds],
        axis=0,
    )
    balls = np.stack(
        [item.ball for item in eligible_seeds],
        axis=0,
    )

    # Graph edges only describe moderate neighboring stable states. They do
    # not impose a finger handoff or a required sequence.
    adjacency = [[] for _ in range(count)]
    edge_count = 0

    for first in range(count):
        hand_delta = np.max(
            np.abs(hands[first + 1:] - hands[first]),
            axis=1,
        )
        ball_delta = np.linalg.norm(
            balls[first + 1:] - balls[first],
            axis=1,
        )
        neighbors = np.nonzero(
            (hand_delta <= args.max_seed_joint_delta)
            & (ball_delta <= args.max_seed_ball_delta)
        )[0]

        for offset in neighbors:
            second = first + 1 + int(offset)
            adjacency[first].append(second)
            adjacency[second].append(first)
            edge_count += 1

    components = []
    visited = np.zeros(count, dtype=bool)

    for start_index in range(count):
        if visited[start_index]:
            continue

        stack = [start_index]
        visited[start_index] = True
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        components.append(component)

    largest_component = max(
        (len(component) for component in components),
        default=0,
    )
    largest_middle_count = max(
        (
            int(np.count_nonzero(middle_flags[component]))
            for component in components
        ),
        default=0,
    )

    # A valid root must have enough useful candidates inside the final
    # absolute ball-radius limit. This prevents graph-chain accumulation from
    # producing seeds 15–20 mm away from the main reset pose.
    root_options = []

    for component_index, component in enumerate(components):
        if len(component) < args.num_seeds:
            continue

        for root_index in component:
            root = eligible_seeds[root_index]
            distances = np.linalg.norm(
                balls[component] - root.ball[None, :],
                axis=1,
            )
            local_indices = [
                component[position]
                for position, distance in enumerate(distances)
                if distance <= args.max_bank_ball_radius
            ]
            local_middle_count = int(
                np.count_nonzero(middle_flags[local_indices])
            )

            if (
                len(local_indices) < args.num_seeds
                or local_middle_count < args.min_middle_seeds
            ):
                continue

            contact_count = sustained_contact_count(
                root,
                args.middle_contact_fraction,
            )
            has_preferred_contacts = (
                contact_count >= args.default_min_contacts
            )
            root_options.append(
                (
                    not has_preferred_contacts,
                    default_seed_cost(root, args),
                    -len(local_indices),
                    -local_middle_count,
                    component_index,
                    root_index,
                    local_indices,
                )
            )

    print(
        "selection_graph: "
        f"eligible={count} "
        f"edges={edge_count} "
        f"components={len(components)} "
        f"largest_component={largest_component} "
        f"largest_middle_count={largest_middle_count} "
        f"valid_roots={len(root_options)}"
    )

    if not root_options:
        print(
            "default_selection: FAIL — no candidate has at least "
            f"{args.num_seeds} eligible neighbors, including "
            f"{args.min_middle_seeds} middle-engaged candidates, within "
            f"{1000 * args.max_bank_ball_radius:.1f} mm."
        )
        return []

    (
        _,
        _,
        _,
        _,
        component_index,
        root_index,
        local_indices,
    ) = min(root_options)

    root = eligible_seeds[root_index]
    local = [
        eligible_seeds[index]
        for index in local_indices
    ]
    middle_pool = [
        item
        for item in local
        if item.fractions[1] >= args.middle_contact_fraction
    ]

    root_contacts = sustained_contact_count(
        root,
        args.middle_contact_fraction,
    )
    root_middle = bool(
        root.fractions[1] >= args.middle_contact_fraction
    )
    max_ball_radius = max(
        (
            float(np.linalg.norm(item.ball - root.ball))
            for item in local
        ),
        default=0.0,
    )

    print(
        "default_selection: "
        f"component={component_index} "
        f"local_pool={len(local)} "
        f"middle_available={len(middle_pool)} "
        f"root_contacts={root_contacts} "
        f"root_middle={root_middle} "
        f"root_thumb_up={root.thumb_up:.2f} "
        f"root_probe_peak={root.probe_peak_speed:.3f} "
        f"root_drop_mm={1000 * root.drop:.2f} "
        f"pool_ball_radius_mm={1000 * max_ball_radius:.2f}"
    )

    selected = [root]

    # First satisfy middle-finger coverage.
    required_middle = max(
        0,
        args.min_middle_seeds - int(root_middle),
    )

    while required_middle > 0:
        choices = []

        for item in middle_pool:
            if any(item is old for old in selected):
                continue

            diversity = min(
                feature_distance(
                    item,
                    old,
                    radius,
                    args,
                )
                for old in selected
            )
            ball_radius_penalty = (
                np.linalg.norm(item.ball - root.ball)
                / max(args.max_bank_ball_radius, 1e-9)
            )
            vertical_penalty = (
                abs(float(item.ball[2] - root.ball[2]))
                / max(args.max_bank_ball_radius, 1e-9)
            )
            utility = min(
                item.probe_peak_speed / 0.05,
                2.0,
            )

            priority = (
                diversity
                + 0.14 * utility
                + 0.06 * item.probe_safe_fraction
                + 0.04 * item.thumb_up
                - 0.08 * ball_radius_penalty
                - 0.05 * vertical_penalty
                - 0.0015 * item.score
            )
            choices.append((priority, item))

        if not choices:
            print(
                "middle_seed_quota: FAIL — no remaining middle-engaged "
                "candidate satisfies the final bank-radius constraints."
            )
            return []

        selected.append(
            max(choices, key=lambda pair: pair[0])[1]
        )
        required_middle -= 1

    # Fill the remaining slots by meaningful task-space diversity.
    factors = (1.0, 0.85, 0.70, 0.60)

    for factor in factors:
        while len(selected) < args.num_seeds:
            choices = []

            for item in local:
                if any(item is old for old in selected):
                    continue

                nearest_rms = min(
                    rms(item, old)
                    for old in selected
                )
                nearest_tip = min(
                    tip_surface_shift(
                        item,
                        old,
                        radius,
                    )
                    for old in selected
                )
                nearest_probe = min(
                    probe_distance(item, old)
                    for old in selected
                )

                meaningful = bool(
                    nearest_rms
                    >= factor * args.min_seed_rms
                    or nearest_tip
                    >= factor
                    * args.min_tip_surface_shift
                    or nearest_probe
                    >= factor
                    * args.min_probe_signature_distance
                )
                if not meaningful:
                    continue

                diversity = min(
                    feature_distance(
                        item,
                        old,
                        radius,
                        args,
                    )
                    for old in selected
                )
                ball_radius_penalty = (
                    np.linalg.norm(item.ball - root.ball)
                    / max(args.max_bank_ball_radius, 1e-9)
                )
                utility = min(
                    item.probe_peak_speed / 0.05,
                    2.0,
                )

                priority = (
                    diversity
                    + 0.12 * utility
                    + 0.05 * item.probe_safe_fraction
                    + 0.04 * item.thumb_up
                    - 0.06 * ball_radius_penalty
                    - 0.0015 * item.score
                )
                choices.append((priority, item))

            if not choices:
                break

            selected.append(
                max(choices, key=lambda pair: pair[0])[1]
            )

        if len(selected) == args.num_seeds:
            break

    print(
        f"selection_result: selected={len(selected)} "
        f"relaxation_floor={factors[-1]:.2f}"
    )

    if len(selected) < args.num_seeds:
        return []

    middle_selected = sum(
        seed.fractions[1] >= args.middle_contact_fraction
        for seed in selected
    )
    minimum_thumb = min(seed.thumb_up for seed in selected)
    selected_ball_radius = max(
        np.linalg.norm(seed.ball - root.ball)
        for seed in selected
    )

    if middle_selected < args.min_middle_seeds:
        print(
            "middle_seed_quota: FAIL — "
            f"selected={middle_selected}, "
            f"required={args.min_middle_seeds}"
        )
        return []

    if minimum_thumb < args.saved_thumb_upright_min_cos:
        print(
            "saved_thumb_quota: FAIL — "
            f"minimum={minimum_thumb:.3f}, "
            f"required={args.saved_thumb_upright_min_cos:.3f}"
        )
        return []

    if selected_ball_radius > args.max_bank_ball_radius + 1e-12:
        print(
            "bank_radius_quota: FAIL — "
            f"radius_mm={1000 * selected_ball_radius:.2f}, "
            f"limit_mm={1000 * args.max_bank_ball_radius:.2f}"
        )
        return []

    print(
        "final_bank_checks: PASS — "
        f"middle={middle_selected}/{args.num_seeds} "
        f"min_thumb_up={minimum_thumb:.2f} "
        f"ball_radius_mm={1000 * selected_ball_radius:.2f}"
    )

    # Seed 0 remains fixed. Remaining seeds are ordered only for convenient
    # viewing; no semantic handoff sequence is implied.
    ordered = [root]
    remaining = selected[1:]

    while remaining:
        current = ordered[-1]
        next_seed = min(
            remaining,
            key=lambda item: (
                rms(current, item)
                + 0.50
                * tip_surface_shift(
                    current,
                    item,
                    radius,
                )
            ),
        )
        ordered.append(next_seed)
        remaining.remove(next_seed)

    return ordered

def support_text(seed):
    names = ("ff", "mf", "rf", "th")
    active = [
        name for index, name in enumerate(names)
        if seed.fractions[index] >= 0.70
    ]
    return "+".join(active) if active else "none"


def save_bank(path_text, seeds):
    path = Path(path_text)
    if not path.is_absolute():
        path = ASSETS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.stack([
        np.concatenate((seed.hand, seed.ball, seed.quat))
        for seed in seeds
    ])
    np.save(path, states)
    print(f"Saved seed bank: {path}, shape={states.shape}")


def write_home(seed):
    scene = LEAP / "scene.xml"
    backup = scene.with_suffix(".xml.bak")
    shutil.copy2(scene, backup)
    qpos = " ".join(
        f"{value:.9g}"
        for value in np.concatenate((seed.hand, seed.ball, seed.quat))
    )
    ctrl = " ".join(f"{value:.9g}" for value in seed.hand)
    text = scene.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(<key\s+name="home"\s+qpos=")[^"]*'
        r'("\s+ctrl=")[^"]*("\s*/>)',
        re.DOTALL,
    )
    text, count = pattern.subn(
        rf"\g<1>{qpos}\g<2>{ctrl}\g<3>", text, count=1
    )
    if count != 1:
        raise RuntimeError('Could not locate <key name="home">')
    scene.write_text(text, encoding="utf-8")
    print(f"Updated: {scene}")
    print(f"Backup:  {backup}")


def main():
    a = parse_args()
    check_args(a)
    model = load_model()
    ids = resolve_ids(model)
    low, high = joint_limits(model)
    base_hand, base_ball, quat = center_state(model, ids, a.center_source)
    base_hand = np.clip(base_hand, low, high)

    print("Local seed-manifold strategy")
    print(f"center_source={a.center_source}")
    print(
        "No forced handoff or contact-mask requirement. The quick search "
        "preserves multiple finger-motion branches instead of collapsing "
        "to the lowest-score near-duplicates."
    )

    candidates = search_candidates(
        model, ids, a, base_hand, base_ball, quat, low, high
    )
    print()
    print(f"Running MuJoCo validation for {len(candidates)} candidates...")

    passing = []
    for index, item in enumerate(candidates, 1):
        result = validate_candidate(model, ids, item, quat, a)

        if result.passed:
            run_micro_probes(model, ids, result, a)
            useful_probe = bool(
                result.probe_safe_fraction >= 0.50
                and result.probe_peak_speed >= a.min_probe_peak_speed
            )
            result.passed = useful_probe

        if result.passed:
            passing.append(result)

        height = (
            result.thumb_height * 1000
            if np.isfinite(result.thumb_height)
            else float("nan")
        )
        print(
            f"candidate={index:03d} pass={result.passed} "
            f"support={support_text(result):<11} "
            f"drop_mm={result.drop * 1000:.2f} "
            f"window_drop_mm={result.window_drop * 1000:.2f} "
            f"drift_mm={result.drift * 1000:.2f} "
            f"thumb_side={result.thumb_side:.2f} "
            f"thumb_h_mm={height:.1f} "
            f"thumb_up={result.thumb_up:.2f} "
            f"probe_safe={result.probe_safe_fraction:.2f} "
            f"probe_peak={result.probe_peak_speed:.3f}"
        )

    selected = choose_seeds(passing, model, ids, a)

    print()
    print("===== LOCAL SEED BANK =====")
    print(f"passing_candidates={len(passing)}")
    if len(selected) != a.num_seeds:
        print(
            f"status: FAIL — found {len(selected)} suitable local seeds; "
            f"need {a.num_seeds}."
        )
        print(
            "Increase --physics-candidates first. When necessary, "
            "reduce one diversity threshold slightly: --min-seed-rms, "
            "--min-tip-surface-shift, or "
            "--min-probe-signature-distance. No seed bank was written."
        )
        return 1

    root = selected[0]
    previous = root
    ball_radius = float(model.geom_size[ids.ball, 0])

    for index, seed in enumerate(selected):
        rms_previous = rms(seed, previous) if index else 0.0
        tip_shift_previous = (
            tip_surface_shift(seed, previous, ball_radius) * 1000
            if index else 0.0
        )
        probe_distance_previous = (
            probe_distance(seed, previous)
            if index else 0.0
        )
        max_from_root = float(np.max(np.abs(seed.hand - root.hand)))
        ball_from_root = float(np.linalg.norm(seed.ball - root.ball) * 1000)

        print(
            f"seed={index:02d} support={support_text(seed):<11} "
            f"rms_from_prev={rms_previous:.4f} "
            f"tip_shift_prev_mm={tip_shift_previous:.2f} "
            f"probe_dist_prev={probe_distance_previous:.4f} "
            f"max_joint_from_seed0={max_from_root:.4f} "
            f"ball_from_seed0_mm={ball_from_root:.2f} "
            f"probe_safe={seed.probe_safe_fraction:.2f} "
            f"probe_peak={seed.probe_peak_speed:.3f} "
            f"thumb_side={seed.thumb_side:.2f} "
            f"thumb_up={seed.thumb_up:.2f} "
            f"drop_mm={seed.drop * 1000:.2f}"
        )
        previous = seed

    save_bank(a.seed_bank_path, selected)
    if a.write_scene:
        write_home(selected[0])

    print()
    print("status: PASS")
    print(
        "The seed bank contains a high-quality three-contact default, "
        "the requested number of middle-engaged seeds, upright thumbs, and "
        "a bounded ball-position distribution. No forced handoff was used."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())