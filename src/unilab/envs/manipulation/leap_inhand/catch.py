"""LEAP Hand falling-ball catch task.

這個檔案是目前「LEAP 手接球」任務的 base code。

設計目標：
- 球從上方落下，位置和初速有小幅隨機誤差。
- 手不要只是用掌心托住球，而是把球穩定在「手掌與手指根部交界」的 pocket。
- 食指 / 中指 / 無名指和大拇指要形成包覆，而不是只有單點碰到。
- Reward 會同時鼓勵：提早閉合、自然握姿、包覆幾何、球穩定、不要掌心-only、不要大拇指交叉。

注意：
- backend 差異不能寫在這裡；MuJoCo / Motrix 的差異應留在 backend 或 YAML。
- 這裡只在 init / reward pass 查 body/sensor 狀態，避免 hot path 解析 XML asset。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse

from .rotation import (
    LeapInhandRotationCfg,
    LeapInhandRotationEnv,
    LeapRotationDomainRandomizationProvider,
)


def _sample_range(bounds: tuple[float, float], size: int | tuple[int, ...]) -> np.ndarray:
    """從 YAML 給定的上下界取 uniform random sample。"""
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"Invalid randomization range: lower={lower} > upper={upper}")
    return np.random.uniform(lower, upper, size=size)


def _planar_enclosure(
    local_tip_vectors: np.ndarray,
    contacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """判斷嚴格包覆：四個 fingertip 是否真的把球圍起來。

    只看接觸數量不夠，因為 policy 可能從同一側碰到球。
    這裡把 fingertip 相對球心的方向投影到 palm 平面，只要最大角度缺口 <= 180 度，
    就代表球心在手指方向形成的圈內。
    """
    planar_vectors = local_tip_vectors[:, :, :2]
    # 每根手指相對球心的 palm-plane 角度。
    angles = np.sort(
        np.arctan2(planar_vectors[:, :, 1], planar_vectors[:, :, 0]),
        axis=1,
    )
    angular_gaps = np.diff(
        np.concatenate([angles, angles[:, :1] + 2.0 * np.pi], axis=1),
        axis=1,
    )
    # 最大角度缺口越小，代表包覆越完整。
    max_angular_gap = np.max(angular_gaps, axis=1)
    all_fingers_contact = np.all(contacts > 0.5, axis=1)
    wrap = np.asarray(
        all_fingers_contact & (max_angular_gap <= np.pi),
        dtype=local_tip_vectors.dtype,
    )
    # wrap 是二值成功；quality 給成功後更細的分數。
    quality = wrap * np.clip(
        (np.pi - max_angular_gap) / np.deg2rad(30.0),
        0.0,
        1.0,
    )
    return wrap, np.asarray(quality, dtype=local_tip_vectors.dtype)


def _soft_planar_enclosure(
    local_tip_vectors: np.ndarray,
    contacts: np.ndarray,
    per_finger: np.ndarray,
) -> np.ndarray:
    """連續版包覆分數，讓早期訓練還沒完全接觸時也有學習梯度。"""
    planar_vectors = local_tip_vectors[:, :, :2]
    angles = np.sort(
        np.arctan2(planar_vectors[:, :, 1], planar_vectors[:, :, 0]),
        axis=1,
    )
    angular_gaps = np.diff(
        np.concatenate([angles, angles[:, :1] + 2.0 * np.pi], axis=1),
        axis=1,
    )
    max_angular_gap = np.max(angular_gaps, axis=1)
    angular_quality = np.clip(
        (np.deg2rad(285.0) - max_angular_gap) / np.deg2rad(105.0),
        0.0,
        1.0,
    )
    # contact 是真的碰到，per_finger 是接近球面；取最大讓「快碰到」也有訊號。
    touch_or_near = np.maximum(contacts, per_finger)
    weakest_finger = np.min(touch_or_near, axis=1)
    average_finger = np.mean(touch_or_near, axis=1)
    contact_quality = 0.65 * weakest_finger + 0.35 * average_finger
    return np.asarray(angular_quality * contact_quality, dtype=local_tip_vectors.dtype)


def _four_pad_contact_quality(
    contacts: np.ndarray,
    pad_proximity: np.ndarray,
    pad_alignment: np.ndarray,
    posture_quality: np.ndarray,
) -> np.ndarray:
    """Return nonzero quality only when all four calibrated pads contact the ball."""
    pad_scores = pad_proximity * (0.10 + 0.90 * pad_alignment)
    effective_contacts = np.clip(contacts, 0.0, 1.0) * posture_quality
    all_contact_gate = np.prod(effective_contacts, axis=1)
    balanced_pad_quality = 0.70 * np.min(pad_scores, axis=1) + 0.30 * np.mean(pad_scores, axis=1)
    return np.asarray(all_contact_gate * balanced_pad_quality, dtype=pad_scores.dtype)


def _staged_wrap_progress(
    contacts: np.ndarray,
    pad_readiness: np.ndarray,
    enclosure_quality: np.ndarray,
) -> np.ndarray:
    """Provide monotonic milestones from first contact to four-pad opposition."""
    main_contact_count = np.sum(np.clip(contacts[:, :3], 0.0, 1.0), axis=1)
    one_main = np.clip(main_contact_count, 0.0, 1.0)
    two_main = np.clip(main_contact_count - 1.0, 0.0, 1.0)
    three_main = np.clip(main_contact_count - 2.0, 0.0, 1.0)
    thumb_after_three = three_main * np.clip(contacts[:, 3], 0.0, 1.0)
    contact_milestones = (one_main + two_main + three_main + thumb_after_three) / 4.0

    balanced_readiness = 0.60 * np.min(pad_readiness, axis=1) + 0.40 * np.mean(
        pad_readiness, axis=1
    )
    return np.asarray(
        0.40 * balanced_readiness + 0.40 * contact_milestones + 0.20 * enclosure_quality,
        dtype=pad_readiness.dtype,
    )


def _finger_posture_quality(dof_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return broad per-finger safety quality and its raw violation penalty.

    The envelope rejects telemetry-confirmed exploits (large spread and reversed
    distal joints) while leaving a wide range of valid tennis-ball grasps. It
    intentionally does not prescribe a fixed pose or a PIP/DIP ratio.
    """
    main_base = dof_pos[:, [0, 4, 8]]
    main_spread = dof_pos[:, [1, 5, 9]]
    main_pip = dof_pos[:, [2, 6, 10]]
    main_tip = dof_pos[:, [3, 7, 11]]

    main_violation = (
        np.square(np.maximum(np.abs(main_spread) - 0.35, 0.0))
        + np.square(np.maximum(0.15 - main_base, 0.0))
        + np.square(np.maximum(main_base - 1.45, 0.0))
        + np.square(np.maximum(0.20 - main_pip, 0.0))
        + np.square(np.maximum(main_pip - 1.75, 0.0))
        + np.square(np.maximum(0.08 - main_tip, 0.0))
        + np.square(np.maximum(main_tip - 1.55, 0.0))
        + np.square(np.maximum(main_tip - main_pip - 0.35, 0.0))
    )

    thumb = dof_pos[:, 12:16]
    thumb_violation = (
        np.square(np.maximum(0.20 - thumb[:, 0], 0.0))
        + np.square(np.maximum(thumb[:, 0] - 1.55, 0.0))
        + np.square(np.maximum(0.00 - thumb[:, 1], 0.0))
        + np.square(np.maximum(thumb[:, 1] - 1.00, 0.0))
        + np.square(np.maximum(0.00 - thumb[:, 2], 0.0))
        + np.square(np.maximum(thumb[:, 2] - 1.30, 0.0))
        + np.square(np.maximum(0.00 - thumb[:, 3], 0.0))
        + np.square(np.maximum(thumb[:, 3] - 1.10, 0.0))
    )
    per_finger_violation = np.concatenate(
        (main_violation, thumb_violation[:, None]),
        axis=1,
    )
    quality = np.exp(-6.0 * per_finger_violation)
    return (
        np.asarray(quality, dtype=dof_pos.dtype),
        np.asarray(np.sum(per_finger_violation, axis=1), dtype=dof_pos.dtype),
    )


def _grasp_surface_layout_quality(local_pad_vectors: np.ndarray) -> np.ndarray:
    """Score whether pads occupy distinct, natural sectors around the ball.

    Distance-to-surface alone has a degenerate solution: all fingers can line up
    on the same side of the sphere. These palm-frame sectors provide a smooth
    gradient toward a three-finger fan opposed by the thumb, without prescribing
    joint angles. The negative local-z component places the pads over the ball
    rather than leaving straight fingertips beside it.
    """
    target_directions = np.asarray(
        (
            (0.78, 0.50, -0.35),
            (0.90, 0.00, -0.43),
            (0.78, -0.50, -0.35),
            (-0.62, 0.00, -0.78),
        ),
        dtype=local_pad_vectors.dtype,
    )
    target_directions /= np.linalg.norm(target_directions, axis=1, keepdims=True)
    unit_vectors = local_pad_vectors / np.maximum(
        np.linalg.norm(local_pad_vectors, axis=2, keepdims=True),
        1e-6,
    )
    cosine = np.sum(unit_vectors * target_directions[None, :, :], axis=2)
    return np.asarray(
        np.exp(-4.0 * np.square(1.0 - np.clip(cosine, -1.0, 1.0))),
        dtype=local_pad_vectors.dtype,
    )


def _opposition_geometry(
    local_tip_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """評分自然握球幾何：主三指在前、大拇指在後側對掌、無名指側有支撐。"""
    main = local_tip_vectors[:, :3, :]
    thumb = local_tip_vectors[:, 3, :]

    # 主三指需要從球前方包過來。
    main_front = np.min(np.clip((main[:, :, 0] - 0.025) / 0.035, 0.0, 1.0), axis=1)
    # 食指和無名指要分布在左右兩側，避免三根手指擠在同一邊。
    index_side = np.clip((main[:, 0, 1] - 0.015) / 0.025, 0.0, 1.0)
    ring_side = np.clip((-main[:, 2, 1] - 0.015) / 0.025, 0.0, 1.0)
    # 大拇指要在球的後側/側後方形成 opposition。
    thumb_back = np.clip((-thumb[:, 0] - 0.020) / 0.035, 0.0, 1.0)

    opposition = main_front * index_side * ring_side * thumb_back
    # 專門懲罰大拇指穿到主手指那邊、或無名指側位置錯誤的情況。
    crossing_penalty = 400.0 * np.square(np.maximum(thumb[:, 0] + 0.005, 0.0)) + 250.0 * np.square(
        np.maximum(main[:, 2, 1] + 0.005, 0.0)
    )
    return (
        np.asarray(opposition, dtype=local_tip_vectors.dtype),
        np.asarray(ring_side, dtype=local_tip_vectors.dtype),
        np.asarray(crossing_penalty, dtype=local_tip_vectors.dtype),
    )


@dataclass(frozen=True)
class _CatchRewardFeatures:
    """一個 reward pass 內共用的接球幾何特徵快取。"""

    pocket_center: np.ndarray
    pocket_quality: np.ndarray
    per_finger: np.ndarray
    wrap: np.ndarray
    enclosure_quality: np.ndarray
    soft_enclosure_quality: np.ndarray
    tip_vectors: np.ndarray
    local_tip_vectors: np.ndarray
    pad_positions: np.ndarray
    pad_normals: np.ndarray
    pad_proximity: np.ndarray
    pad_alignment: np.ndarray
    pad_to_ball: np.ndarray
    contacts: np.ndarray
    palm_contact: np.ndarray
    opposition_quality: np.ndarray
    ring_side_quality: np.ndarray
    crossing_penalty: np.ndarray


class LeapBallCatchDomainRandomizationProvider(LeapRotationDomainRandomizationProvider):
    """Randomize the falling ball on reset without touching assets in the hot path."""

    def _sample_reset_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 先沿用 base env 的 reset state，再覆蓋球的落下狀態。
        hand_qpos, ball_pos, ball_quat, qvel = super()._sample_reset_state(env, num_reset)
        cfg = env.cfg
        # The owner YAML is the single source of truth for the initial pose. The
        # XML keyframe remains a cold-start fallback, but reset must immediately
        # reflect initial_pose edits without requiring duplicated XML changes.
        hand_qpos = np.broadcast_to(env._initial_pose, (num_reset, env._NUM_HAND_DOF)).copy()
        hand_qpos += np.random.uniform(
            -cfg.domain_rand.joint_noise,
            cfg.domain_rand.joint_noise,
            hand_qpos.shape,
        ).astype(hand_qpos.dtype)
        hand_qpos = np.clip(hand_qpos, env._ctrl_lower, env._ctrl_upper)
        # 球的 spawn position 有小範圍誤差，用來模擬真實落球不是完全精準。
        ball_pos[:, 0] = _sample_range(cfg.ball_spawn_x_range, num_reset)
        ball_pos[:, 1] = _sample_range(cfg.ball_spawn_y_range, num_reset)
        ball_pos[:, 2] = _sample_range(cfg.ball_spawn_height_range, num_reset)
        # 初速也加一點誤差，避免 policy 只背固定軌跡。
        qvel[:, env._NUM_HAND_DOF] = _sample_range(cfg.ball_horizontal_velocity_range, num_reset)
        qvel[:, env._NUM_HAND_DOF + 1] = _sample_range(
            cfg.ball_horizontal_velocity_range, num_reset
        )
        qvel[:, env._NUM_HAND_DOF + 2] = _sample_range(cfg.ball_vertical_velocity_range, num_reset)
        return hand_qpos, ball_pos, ball_quat, qvel


@registry.envcfg("LeapBallCatch")
@dataclass
class LeapBallCatchCfg(LeapInhandRotationCfg):
    """LEAP 接球任務的可調參數。"""

    # 使用 LEAP hand robot XML，再加上接球任務自己的 scene fragment。
    # keyframe / ball / floor / contact sensor 都屬於 task scene，不放進 robot XML。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "leap_hand.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene_ball_catch.xml")
            ],
        )
    )
    # 目標接球 pocket：以三根主手指 MCP 關節中心為基準，再在 palm frame 裡偏移。
    # 這代表你要的「手掌與手指交界處」，不是掌心中央。
    # LEAP palm 在模型中 local x 有 180 度旋轉，所以 local z 負方向會對應世界中的上方。
    grasp_pocket_offset: tuple[float, float, float] = (0.005, 0.012, -0.045)
    # Open waiting pose used at reset and as the early-flight reward reference.
    initial_pose: tuple[float, ...] = (
        0.35,
        -0.25,
        0.60,
        0.45,
        0.30,
        0.00,
        0.66,
        0.50,
        0.35,
        0.30,
        0.72,
        0.55,
        1.0,
        1.0,
        0.20,
        0.40,
    )
    # Desired final catch pose. This is a reward reference, not a scripted
    # actuator trajectory: APPO remains responsible for reaching it.
    catch_pose: tuple[float, ...] = (
        0.65,
        -0.50,
        1.35,
        0.90,
        0.60,
        0.00,
        1.36,
        0.70,
        0.80,
        0.50,
        1.52,
        1.10,
        1.0,
        1.0,
        0.7,
        1.0,
    )
    catch_close_trigger_height: float = 0.11
    catch_full_close_height: float = 0.015
    early_closure_tolerance: float = 0.12
    reference_pose_sigma: float = 4.0
    reference_pose_joint_weights: tuple[float, ...] = (
        1.3,
        4.0,
        1.5,
        1.6,
        1.3,
        4.0,
        1.5,
        1.6,
        1.3,
        4.0,
        1.5,
        1.6,
        4.0,
        1.6,
        1.7,
        1.7,
    )
    catch_spread_tolerance: float = 0.02
    catch_spread_limit: float = 0.08
    locked_catch_dof_indices: tuple[int, ...] = (12,)
    pose_preview: str = "none"
    thumb_pregrasp_pose: tuple[float, float, float, float] = (0.8, 0.6, 0.9, 0.6)
    natural_grasp_pose: tuple[float, ...] = (
        0.62,
        -0.14,
        0.78,
        0.38,
        0.74,
        0.00,
        0.88,
        0.46,
        0.82,
        0.13,
        0.96,
        0.52,
        0.66,
        0.42,
        0.50,
        0.24,
    )
    closing_grasp_pose: tuple[float, ...] = (
        0.82,
        -0.16,
        0.95,
        0.42,
        0.94,
        0.00,
        1.06,
        0.52,
        1.04,
        0.14,
        1.16,
        0.60,
        0.72,
        0.48,
        0.58,
        0.28,
    )
    ball_spawn_x_range: tuple[float, float] = (0.008, 0.024)
    ball_spawn_y_range: tuple[float, float] = (0.016, 0.028)
    # 落下高度範圍。越高球速越快，任務越難。
    ball_spawn_height_range: tuple[float, float] = (0.82, 0.88)
    # 球初速誤差。太大早期會學不到，太小又不夠真實。
    ball_horizontal_velocity_range: tuple[float, float] = (-0.08, 0.08)
    ball_vertical_velocity_range: tuple[float, float] = (-0.08, 0.0)
    # 網球半徑約 33.5 mm。
    ball_radius: float = 0.0335
    # fingertip 接近球面時的容忍距離。
    fingertip_surface_margin: float = 0.006
    # 保留 base env 的 grasp cache path 介面；目前主要靠 keyframe + falling ball randomization。
    grasp_cache_path: str = "caches/leap_ball_catch.npy"


@registry.env("LeapBallCatch", sim_backend="mujoco")
@registry.env("LeapBallCatch", sim_backend="motrix")
class LeapBallCatchEnv(LeapInhandRotationEnv):
    """Catch a falling ball in the finger-root pocket and hold it with opposition."""

    _cfg: LeapBallCatchCfg
    # base rotation obs + ball velocity + finger contacts + palm contact + time-to-contact。
    _NUM_OBS_PER_STEP = 51
    _INCLUDE_FINGERTIP_CONTACT_OBS = False
    _INCLUDE_BALL_ANGVEL_OBS = False
    _INCLUDE_ROTATION_PHASE_OBS = False
    # 這四個 sensor 名稱來自 scene_ball_catch.xml，順序是食指/中指/無名指/大拇指。
    _CONTACT_SENSORS = (
        "leap_ff_contact",
        "leap_mf_contact",
        "leap_rf_contact",
        "leap_th_contact",
    )
    _PAD_POSITION_SENSORS = (
        "leap_ff_pad_pos",
        "leap_mf_pad_pos",
        "leap_rf_pad_pos",
        "leap_th_pad_pos",
    )
    _PAD_NORMAL_POSITION_SENSORS = (
        "leap_ff_pad_normal_pos",
        "leap_mf_pad_normal_pos",
        "leap_rf_pad_normal_pos",
        "leap_th_pad_normal_pos",
    )
    # 掌心 sensor 用來懲罰「球只是躺在掌心上」的作弊策略。
    _PALM_CONTACT_SENSOR = "leap_palm_contact"

    def __init__(
        self,
        cfg: LeapBallCatchCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # 三根主手指 MCP body 用來定義接球 pocket 的基準點。
        self._main_mcp_body_ids = self._backend.get_body_ids(
            ("mcp_joint", "mcp_joint_2", "mcp_joint_3")
        )
        # palm body quaternion 用來把 pocket offset / fingertip geometry 轉到 palm frame。
        self._palm_body_ids = self._backend.get_body_ids((cfg.base_body_name,))
        # YAML tuple 轉成 numpy，訓練時不用反覆轉型。
        self._grasp_pocket_offset = np.asarray(cfg.grasp_pocket_offset, dtype=get_global_dtype())
        self._initial_pose = np.asarray(cfg.initial_pose, dtype=get_global_dtype())
        self._catch_pose = np.asarray(cfg.catch_pose, dtype=get_global_dtype())
        self._reference_pose_joint_weights = np.asarray(
            cfg.reference_pose_joint_weights,
            dtype=get_global_dtype(),
        )
        self._locked_catch_dof_ids = np.asarray(
            cfg.locked_catch_dof_indices,
            dtype=np.intp,
        )
        self._thumb_pregrasp_pose = np.asarray(cfg.thumb_pregrasp_pose, dtype=get_global_dtype())
        self._natural_grasp_pose = np.asarray(cfg.natural_grasp_pose, dtype=get_global_dtype())
        self._closing_grasp_pose = np.asarray(cfg.closing_grasp_pose, dtype=get_global_dtype())
        self._finger_surface_distance = np.asarray(
            cfg.ball_radius + cfg.fingertip_surface_margin,
            dtype=get_global_dtype(),
        )
        # pose 長度必須和 LEAP 16 DOF 完全一致，否則 reward 會對錯關節。
        if self._natural_grasp_pose.shape != (self._NUM_HAND_DOF,):
            raise ValueError(
                f"natural_grasp_pose must have {self._NUM_HAND_DOF} values, "
                f"got {self._natural_grasp_pose.shape[0]}"
            )
        if self._closing_grasp_pose.shape != (self._NUM_HAND_DOF,):
            raise ValueError(
                f"closing_grasp_pose must have {self._NUM_HAND_DOF} values, "
                f"got {self._closing_grasp_pose.shape[0]}"
            )
        for name, value in (
            ("initial_pose", self._initial_pose),
            ("catch_pose", self._catch_pose),
            ("reference_pose_joint_weights", self._reference_pose_joint_weights),
        ):
            if value.shape != (self._NUM_HAND_DOF,):
                raise ValueError(
                    f"{name} must have {self._NUM_HAND_DOF} values, got {value.shape[0]}"
                )
        if np.any(self._reference_pose_joint_weights <= 0.0):
            raise ValueError("reference_pose_joint_weights must all be positive")
        if (
            np.any(self._locked_catch_dof_ids < 0)
            or np.any(self._locked_catch_dof_ids >= self._NUM_HAND_DOF)
            or np.unique(self._locked_catch_dof_ids).size != self._locked_catch_dof_ids.size
        ):
            raise ValueError(
                f"locked_catch_dof_indices must contain unique indices in [0, {self._NUM_HAND_DOF})"
            )
        if cfg.catch_close_trigger_height <= cfg.catch_full_close_height:
            raise ValueError(
                "catch_close_trigger_height must be greater than catch_full_close_height"
            )
        if self._locked_catch_dof_ids.size and not np.allclose(
            self._initial_pose[self._locked_catch_dof_ids],
            self._catch_pose[self._locked_catch_dof_ids],
        ):
            raise ValueError("locked catch DOFs must match in initial_pose and catch_pose")
        if not 0.0 <= cfg.early_closure_tolerance < 1.0:
            raise ValueError("early_closure_tolerance must be in [0, 1)")
        if cfg.reference_pose_sigma <= 0.0:
            raise ValueError("reference_pose_sigma must be positive")
        if cfg.catch_spread_limit < 0.0:
            raise ValueError("catch_spread_limit must be non-negative")
        if cfg.pose_preview not in {"none", "initial", "catch", "both"}:
            raise ValueError("pose_preview must be one of: none, initial, catch, both")
        # 只在 `_compute_reward()` 的一次 reward pass 裡暫存，離開後清掉。
        self._reward_features: _CatchRewardFeatures | None = None

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        # 把任務專用 reward term 掛進 base env 的 reward registry；權重由 YAML 控制。
        self._reward_fns["intercept"] = self._reward_intercept
        self._reward_fns["finger_approach"] = self._reward_finger_approach
        self._reward_fns["finger_pad_progress"] = self._reward_finger_pad_progress
        self._reward_fns["grasp_surface_layout"] = self._reward_grasp_surface_layout
        self._reward_fns["finger_closing"] = self._reward_finger_closing
        self._reward_fns["reference_pose_tracking"] = self._reward_reference_pose_tracking
        self._reward_fns["early_closure"] = self._reward_early_closure
        self._reward_fns["finger_gap"] = self._reward_finger_gap
        self._reward_fns["finger_synergy"] = self._reward_finger_synergy
        self._reward_fns["finger_curl_velocity"] = self._reward_finger_curl_velocity
        self._reward_finger_flex_pose = self._reward_closing_grasp_pose
        self._reward_fns["closing_grasp_pose"] = self._reward_closing_grasp_pose
        self._reward_fns["thumb_pregrasp"] = self._reward_thumb_pregrasp
        self._reward_fns["thumb_approach"] = self._reward_thumb_approach
        self._reward_fns["thumb_contact"] = self._reward_thumb_contact
        self._reward_fns["thumb_top_approach"] = self._reward_thumb_top_approach
        self._reward_fns["thumb_over_ball"] = self._reward_thumb_over_ball
        self._reward_fns["thumb_missing_contact"] = self._reward_thumb_missing_contact
        self._reward_fns["thumb_pad_press"] = self._reward_thumb_pad_press
        self._reward_fns["thumb_pad_alignment"] = self._reward_thumb_pad_alignment
        self._reward_fns["thumb_nail_contact"] = self._reward_thumb_nail_contact
        self._reward_fns["thumb_misplaced"] = self._reward_thumb_misplaced
        self._reward_fns["finger_pad_press"] = self._reward_finger_pad_press
        self._reward_fns["main_finger_fan_pose"] = self._reward_main_finger_fan_pose
        self._reward_fns["ring_pad_press"] = self._reward_ring_pad_press
        self._reward_fns["bad_pad_contact"] = self._reward_bad_pad_contact
        self._reward_fns["ring_inward_closing"] = self._reward_ring_inward_closing
        self._reward_fns["grasp_tightness"] = self._reward_grasp_tightness
        self._reward_fns["natural_grasp_pose"] = self._reward_natural_grasp_pose
        self._reward_fns["opposition_geometry"] = self._reward_opposition_geometry
        self._reward_fns["ring_support"] = self._reward_ring_support
        self._reward_fns["wrap_progress"] = self._reward_wrap_progress
        self._reward_fns["finger_wrap"] = self._reward_finger_wrap
        self._reward_fns["pre_bounce_catch"] = self._reward_pre_bounce_catch
        self._reward_fns["hold"] = self._reward_hold
        self._reward_fns["touch_without_wrap"] = self._reward_touch_without_wrap
        self._reward_fns["palm_only"] = self._reward_palm_only
        self._reward_fns["palm_guidance"] = self._reward_palm_guidance
        self._reward_fns["palm_assist"] = self._reward_palm_assist
        self._reward_fns["palm_contact"] = self._reward_palm_contact
        self._reward_fns["ball_bounce"] = self._reward_ball_bounce
        self._reward_fns["pocket_without_finger_contact"] = (
            self._reward_pocket_without_finger_contact
        )
        self._reward_fns["pocket_depth"] = self._reward_pocket_depth
        self._reward_fns["high_ball"] = self._reward_high_ball
        self._reward_fns["drop_risk"] = self._reward_drop_risk
        self._reward_fns["grasp_stability"] = self._reward_grasp_stability
        self._reward_fns["finger_jitter"] = self._reward_finger_jitter
        self._reward_fns["action_rate"] = self._reward_action_rate
        self._reward_fns["settled_action"] = self._reward_settled_action
        self._reward_fns["unnatural_joint_pose"] = self._reward_unnatural_joint_pose
        self._reward_fns["thumb_crossing"] = self._reward_thumb_crossing

    def _domain_randomization_provider(
        self,
    ) -> LeapBallCatchDomainRandomizationProvider:
        # reset 時使用上面定義的 falling-ball randomization。
        return LeapBallCatchDomainRandomizationProvider()

    def _compute_grasp_pocket_center(self) -> np.ndarray:
        """計算每個 env 當前的理想接球 pocket 世界座標。"""
        mcp_center = np.mean(self._backend.get_body_pos_w(self._main_mcp_body_ids), axis=1)
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[:, 0, :]
        offset = np.broadcast_to(self._grasp_pocket_offset, (self._num_envs, 3))
        return np.asarray(mcp_center + np_quat_apply(palm_quat, offset), dtype=self._np_dtype)

    def _grasp_pocket_center(self) -> np.ndarray:
        """Return the desired ball center at the palm/main-finger boundary."""
        if self._reward_features is not None:
            return self._reward_features.pocket_center
        return self._compute_grasp_pocket_center()

    def _catch_close_phase(
        self,
        ball_pos: np.ndarray,
        contacts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return 0 at the ready cup and 1 at the fully curled catch target."""
        del contacts
        height = ball_pos[:, 2] - self._grasp_pocket_center()[:, 2]
        phase = (self._cfg.catch_close_trigger_height - height) / (
            self._cfg.catch_close_trigger_height - self._cfg.catch_full_close_height
        )
        phase = np.clip(phase, 0.0, 1.0)
        return np.asarray(phase, dtype=self._np_dtype)

    def _catch_reference_pose(
        self,
        ball_pos: np.ndarray,
        contacts: np.ndarray | None = None,
    ) -> np.ndarray:
        phase = self._catch_close_phase(ball_pos, contacts)
        # Smoothstep avoids an abrupt reward-target change at the trigger height.
        phase = phase * phase * (3.0 - 2.0 * phase)
        return np.asarray(
            self._initial_pose[None, :]
            + phase[:, None] * (self._catch_pose - self._initial_pose)[None, :],
            dtype=self._np_dtype,
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Apply independent APPO residual actions with explicit safety constraints."""
        clipped_actions = np.asarray(np.clip(actions, -1.0, 1.0), dtype=self._np_dtype)
        ball_pos = np.asarray(
            state.info.get("curr_ball_pos", state.info.get("prev_ball_pos", self.get_ball_pos())),
            dtype=self._np_dtype,
        )
        contacts = np.asarray(
            state.info.get(
                "curr_contacts",
                state.info.get("prev_contacts", np.zeros((self._num_envs, 4))),
            ),
            dtype=self._np_dtype,
        )
        phase = self._catch_close_phase(ball_pos, contacts)
        reference_pose = self._catch_reference_pose(ball_pos, contacts)

        if self._cfg.pose_preview == "none":
            # Actual RL path: every policy action stays independent. The poses
            # below shape reward only and do not generate a closing trajectory.
            ctrl = super().apply_action(clipped_actions, state)
        else:
            state.info["last_actions"] = state.info.get(
                "current_actions", np.zeros_like(clipped_actions)
            )
            state.info["current_actions"] = np.zeros_like(clipped_actions)
            if self._cfg.pose_preview == "initial":
                ctrl = np.broadcast_to(self._initial_pose, clipped_actions.shape).copy()
            elif self._cfg.pose_preview == "catch":
                ctrl = np.broadcast_to(self._catch_pose, clipped_actions.shape).copy()
            else:
                ctrl = np.broadcast_to(self._initial_pose, clipped_actions.shape).copy()
                ctrl[clipped_actions.shape[0] // 2 :] = self._catch_pose

        # The thumb root is the only fully locked joint requested by the task.
        ctrl[:, self._locked_catch_dof_ids] = self._initial_pose[self._locked_catch_dof_ids]

        # Spread remains trainable inside static safety bounds spanning both
        # authored poses. The reference phase never writes actuator targets.
        spread_ids = np.asarray([1, 5, 9], dtype=np.intp)
        spread_limit = float(self._cfg.catch_spread_limit)
        spread_lower = (
            np.minimum(self._initial_pose[spread_ids], self._catch_pose[spread_ids]) - spread_limit
        )
        spread_upper = (
            np.maximum(self._initial_pose[spread_ids], self._catch_pose[spread_ids]) + spread_limit
        )
        ctrl[:, spread_ids] = np.clip(ctrl[:, spread_ids], spread_lower, spread_upper)
        ctrl = np.clip(ctrl, self._ctrl_lower, self._ctrl_upper)
        ctrl = np.asarray(ctrl, dtype=self._np_dtype)
        state.info["catch_phase"] = phase
        state.info["catch_reference_pose"] = reference_pose
        state.info["prev_ctrl"] = ctrl
        return ctrl

    def _pocket_quality(self, ball_pos: np.ndarray) -> np.ndarray:
        """球越接近 pocket，分數越高；如果 reward pass 已算過就直接重用。"""
        if self._reward_features is not None:
            return self._reward_features.pocket_quality
        pocket_center = self._grasp_pocket_center()
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[:, 0, :]
        return self._pocket_quality_from_error(ball_pos, pocket_center, palm_quat)

    def _pocket_quality_from_error(
        self,
        ball_pos: np.ndarray,
        pocket_center: np.ndarray,
        palm_quat: np.ndarray,
    ) -> np.ndarray:
        """根據 ball center 與 pocket center 的 palm-local error 轉成 0~1 quality。"""
        local_error = np_quat_apply_inverse(palm_quat, ball_pos - pocket_center)
        weights = np.asarray([1100.0, 1200.0, 2200.0], dtype=self._np_dtype)
        return np.asarray(
            np.exp(-np.sum(weights * np.square(local_error), axis=1)), dtype=self._np_dtype
        )

    def _time_to_contact(
        self,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
    ) -> np.ndarray:
        """估計球還要多久會落到 pocket 高度，用來控制閉合 timing。"""
        pocket_center = self._grasp_pocket_center()
        if pocket_center.shape[0] != ball_pos.shape[0]:
            # Partial reset observations contain only reset rows. The LEAP base
            # is fixed, so every environment has the same finger-root pocket.
            pocket_center = np.broadcast_to(
                pocket_center[:1],
                (ball_pos.shape[0], 3),
            )
        height = np.maximum(ball_pos[:, 2] - pocket_center[:, 2], 0.0)
        downward_speed = np.maximum(-ball_linvel[:, 2], 0.0)
        gravity = 9.81
        return np.asarray(
            (np.sqrt(np.square(downward_speed) + 2.0 * gravity * height) - downward_speed)
            / gravity,
            dtype=self._np_dtype,
        )

    @staticmethod
    def _sensor_scalar(sensor_data: np.ndarray) -> np.ndarray:
        """把不同 backend 回來的 sensor shape 統一成每個 env 一個 scalar。"""
        sensor_data = np.asarray(sensor_data)
        if sensor_data.ndim == 1:
            return sensor_data
        return sensor_data.reshape(sensor_data.shape[0], -1)[:, 0]

    @staticmethod
    def _sensor_vector(sensor_data: np.ndarray) -> np.ndarray:
        """Normalize a batched frame-position sensor to shape (num_envs, 3)."""
        values = np.asarray(sensor_data)
        values = values.reshape(values.shape[0], -1)
        if values.shape[1] != 3:
            raise ValueError(f"Expected a 3-D frame-position sensor, got {values.shape}")
        return values

    def _read_pad_geometry(
        self, ball_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read calibrated pad centers/normals and derive ball-relative geometry."""
        pad_positions = np.stack(
            [
                self._sensor_vector(self.get_sensor_data(name))
                for name in self._PAD_POSITION_SENSORS
            ],
            axis=1,
        )
        normal_positions = np.stack(
            [
                self._sensor_vector(self.get_sensor_data(name))
                for name in self._PAD_NORMAL_POSITION_SENSORS
            ],
            axis=1,
        )
        pad_normals = normal_positions - pad_positions
        normal_lengths = np.maximum(np.linalg.norm(pad_normals, axis=2, keepdims=True), 1e-6)
        pad_normals = pad_normals / normal_lengths
        pad_to_ball = ball_pos[:, None, :] - pad_positions
        distances = np.maximum(np.linalg.norm(pad_to_ball, axis=2), 1e-6)
        surface_error = distances - self._finger_surface_distance
        pad_proximity = np.exp(-500.0 * np.square(surface_error))
        pad_alignment = np.clip(
            np.sum(pad_normals * (pad_to_ball / distances[:, :, None]), axis=2),
            0.0,
            1.0,
        )
        return (
            np.asarray(pad_positions, dtype=self._np_dtype),
            np.asarray(pad_normals, dtype=self._np_dtype),
            np.asarray(pad_proximity, dtype=self._np_dtype),
            np.asarray(pad_alignment, dtype=self._np_dtype),
            np.asarray(pad_to_ball, dtype=self._np_dtype),
        )

    def _finger_contacts(self) -> np.ndarray:
        """讀取四根手指 contact sensor，回傳 shape = (num_envs, 4)。"""
        return np.asarray(
            np.stack(
                [self._sensor_scalar(self.get_sensor_data(name)) for name in self._CONTACT_SENSORS],
                axis=1,
            )
            > 0.5,
            dtype=self._np_dtype,
        )

    def _palm_contact(self) -> np.ndarray:
        """讀取掌心 contact sensor，用於 palm-only / palm-contact penalty。"""
        return np.asarray(
            self._sensor_scalar(self.get_sensor_data(self._PALM_CONTACT_SENSOR)) > 0.5,
            dtype=self._np_dtype,
        )

    def _compute_catch_features(
        self,
        ball_pos: np.ndarray,
        contacts: np.ndarray | None = None,
        palm_contact: np.ndarray | None = None,
    ) -> _CatchRewardFeatures:
        """集中計算接球任務所有 reward 會用到的幾何特徵。"""
        pocket_center = self._compute_grasp_pocket_center()
        palm_quat = self._backend.get_body_quat_w(self._palm_body_ids)[:, 0, :]
        pocket_quality = self._pocket_quality_from_error(ball_pos, pocket_center, palm_quat)

        # fingertip 相對球心的向量。後面「接近、包覆、opposition」都基於它。
        (
            pad_positions,
            pad_normals,
            pad_proximity,
            pad_alignment,
            pad_to_ball,
        ) = self._read_pad_geometry(ball_pos)
        tip_vectors = np.asarray(pad_positions - ball_pos[:, None, :], dtype=self._np_dtype)
        # per_finger 是「每根 fingertip 是否接近球面」的連續分數。
        per_finger = pad_proximity
        pad_scores = pad_proximity * pad_alignment

        # update_state 會把 contacts 放進 info；如果沒有，這裡才直接讀 sensor。
        if contacts is None:
            contacts = self._finger_contacts()
        if palm_contact is None:
            palm_contact = self._palm_contact()

        # Contact count alone is not a grasp: a policy can touch the sphere
        # with several fingers from the same side. Project the four fingertip
        # directions into the palm plane. The ball center is enclosed only
        # when those directions cover the full circle (largest angular gap is
        # at most pi), which is the 2-D convex-hull containment condition.
        palm_quat_per_tip = np.repeat(palm_quat[:, None, :], 4, axis=1)
        local_tip_vectors = np_quat_apply_inverse(
            palm_quat_per_tip.reshape(-1, 4),
            tip_vectors.reshape(-1, 3),
        ).reshape(self._num_envs, 4, 3)
        wrap, enclosure_quality = _planar_enclosure(local_tip_vectors, contacts)
        soft_enclosure_quality = _soft_planar_enclosure(
            local_tip_vectors,
            contacts,
            np.asarray(pad_scores, dtype=self._np_dtype),
        )
        opposition_quality, ring_side_quality, crossing_penalty = _opposition_geometry(
            local_tip_vectors
        )
        return _CatchRewardFeatures(
            pocket_center=pocket_center,
            pocket_quality=pocket_quality,
            per_finger=np.asarray(per_finger, dtype=self._np_dtype),
            wrap=np.asarray(wrap, dtype=self._np_dtype),
            enclosure_quality=np.asarray(enclosure_quality, dtype=self._np_dtype),
            soft_enclosure_quality=np.asarray(soft_enclosure_quality, dtype=self._np_dtype),
            tip_vectors=tip_vectors,
            local_tip_vectors=np.asarray(local_tip_vectors, dtype=self._np_dtype),
            pad_positions=pad_positions,
            pad_normals=pad_normals,
            pad_proximity=pad_proximity,
            pad_alignment=pad_alignment,
            pad_to_ball=pad_to_ball,
            contacts=contacts,
            palm_contact=np.asarray(palm_contact, dtype=self._np_dtype),
            opposition_quality=np.asarray(opposition_quality, dtype=self._np_dtype),
            ring_side_quality=np.asarray(ring_side_quality, dtype=self._np_dtype),
            crossing_penalty=np.asarray(crossing_penalty, dtype=self._np_dtype),
        )

    def _finger_geometry(self, ball_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return surface proximity, opposition-aware wrap, and tip-to-ball vectors."""
        features = self._reward_features or self._compute_catch_features(ball_pos)
        return (
            features.per_finger,
            features.wrap,
            features.tip_vectors,
        )

    def _finger_support_quality(self, features: _CatchRewardFeatures) -> np.ndarray:
        """回傳手指是否真的能支撐球，而不只是球剛好在 pocket 裡。

        這個分數同時看兩件事：
        - fingertip 是否靠近球面，讓早期訓練仍有連續訊號；
        - contact sensor 是否真的被觸發，讓最終 hold 不能靠空氣包覆作弊。
        """
        soft_alignment = 0.10 + 0.90 * features.pad_alignment
        pad_scores = features.pad_proximity * soft_alignment
        touch_or_near = np.maximum(features.contacts * soft_alignment, pad_scores)
        main_near = 0.65 * np.min(touch_or_near[:, :3], axis=1) + 0.35 * np.mean(
            touch_or_near[:, :3], axis=1
        )
        thumb_near = touch_or_near[:, 3]
        soft_support = 0.75 * main_near + 0.25 * thumb_near

        main_contact_count = np.sum(features.contacts[:, :3], axis=1)
        # 至少兩根主手指接觸才開始算真正支撐，三根主手指接觸才滿分。
        main_contact_gate = np.clip((main_contact_count - 1.0) / 2.0, 0.0, 1.0)
        thumb_contact_gate = features.contacts[:, 3]
        contact_support = 0.80 * main_contact_gate + 0.20 * thumb_contact_gate
        return np.asarray(soft_support * contact_support, dtype=self._np_dtype)

    def _finger_near_quality(self, features: _CatchRewardFeatures) -> np.ndarray:
        """比 contact 更軟的手指接近分數，用來鋪到真正 finger_wrap。"""
        pad_scores = features.pad_proximity * (0.10 + 0.90 * features.pad_alignment)
        main_near = 0.65 * np.min(pad_scores[:, :3], axis=1) + 0.35 * np.mean(
            pad_scores[:, :3], axis=1
        )
        thumb_near = pad_scores[:, 3]
        return np.asarray(0.75 * main_near + 0.25 * thumb_near, dtype=self._np_dtype)

    def _thumb_opposition_lane(self, features: _CatchRewardFeatures) -> np.ndarray:
        """大拇指是否在球後側/側後方，而不是跟三指擠同一側。"""
        thumb = features.local_tip_vectors[:, 3, :]
        thumb_back = np.clip((-thumb[:, 0] - 0.010) / 0.040, 0.0, 1.0)
        return np.asarray(thumb_back, dtype=self._np_dtype)

    def _loose_pocket_quality(
        self, ball_pos: np.ndarray, features: _CatchRewardFeatures
    ) -> np.ndarray:
        """較寬鬆的掌根/pocket 接近分，給早期 shaping 使用。"""
        loose_error = ball_pos - features.pocket_center
        return np.asarray(
            np.exp(
                -70.0 * np.sum(np.square(loose_error[:, :2]), axis=1)
                - 25.0 * np.square(loose_error[:, 2])
            ),
            dtype=self._np_dtype,
        )

    def _palm_or_pocket_gate(
        self, ball_pos: np.ndarray, features: _CatchRewardFeatures
    ) -> np.ndarray:
        """球是否已經進到掌根/指根區域，或真的碰到掌心。"""
        return np.asarray(
            np.maximum(features.palm_contact, self._loose_pocket_quality(ball_pos, features)),
            dtype=self._np_dtype,
        )

    def _vertical_pocket_error(
        self, ball_pos: np.ndarray, features: _CatchRewardFeatures
    ) -> np.ndarray:
        """球心相對 pocket 的世界 z 高度；正值代表球還停得太高。"""
        return np.asarray(ball_pos[:, 2] - features.pocket_center[:, 2], dtype=self._np_dtype)

    def _finger_pad_proxy(
        self,
        ball_pos: np.ndarray,
        finger_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated pad proximity, alignment, and pad-to-ball vectors."""
        if self._reward_features is not None:
            return (
                self._reward_features.pad_proximity[:, finger_index],
                self._reward_features.pad_alignment[:, finger_index],
                self._reward_features.pad_to_ball[:, finger_index, :],
            )
        _, _, proximity, alignment, pad_to_ball = self._read_pad_geometry(ball_pos)
        return (
            proximity[:, finger_index],
            alignment[:, finger_index],
            pad_to_ball[:, finger_index, :],
        )

    def _thumb_top_approach_quality(
        self, ball_pos: np.ndarray, pad_to_ball: np.ndarray
    ) -> np.ndarray:
        """Dense world-frame shaping toward the top cap without rewarding joint angles."""
        del ball_pos
        horizontal_error_sq = np.sum(np.square(pad_to_ball[:, :2]), axis=1)
        top_height_error = pad_to_ball[:, 2] + self._finger_surface_distance
        return np.asarray(
            np.exp(-25.0 * horizontal_error_sq - 10.0 * np.square(top_height_error)),
            dtype=self._np_dtype,
        )

    def _thumb_world_top_quality(self, pad_to_ball: np.ndarray) -> np.ndarray:
        """Dense world-frame target for putting the thumb pad above the ball."""
        horizontal_quality = np.exp(-180.0 * np.sum(np.square(pad_to_ball[:, :2]), axis=1))
        top_height_error = pad_to_ball[:, 2] + self._finger_surface_distance
        height_quality = np.exp(-650.0 * np.square(top_height_error))
        above_gate = np.clip((-pad_to_ball[:, 2] - 0.006) / 0.060, 0.0, 1.0)
        return np.asarray(horizontal_quality * height_quality * above_gate, dtype=self._np_dtype)

    def _compute_reward(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        # All catch terms consume the same body state. Cache it only for this
        # reward pass so 1024-env training does not repeat backend queries.
        contacts = info.get("curr_contacts")
        palm_contact = info.get("curr_palm_contact")
        self._reward_features = self._compute_catch_features(
            ball_pos,
            None if contacts is None else np.asarray(contacts, dtype=self._np_dtype),
            None if palm_contact is None else np.asarray(palm_contact, dtype=self._np_dtype),
        )
        try:
            return super()._compute_reward(
                info,
                dof_pos,
                dof_vel,
                ball_pos,
                ball_linvel,
                ball_angvel,
                torques,
                terminated,
            )
        finally:
            self._reward_features = None

    def _reward_intercept(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵球的水平軌跡對準接球 pocket。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        horizontal_error = ball_pos[:, :2] - self._grasp_pocket_center()[:, :2]
        return np.asarray(
            np.exp(-300.0 * np.sum(np.square(horizontal_error), axis=1)),
            dtype=self._np_dtype,
        )

    def _reward_finger_approach(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵主三指接近球面，而且球必須真的靠近 pocket。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        per_finger, _, _ = self._finger_geometry(ball_pos)
        main_proximity = 0.70 * np.min(per_finger[:, :3], axis=1) + 0.30 * np.mean(
            per_finger[:, :3], axis=1
        )
        capture_gate = np.exp(
            -20.0 * np.sum(np.square(ball_pos - self._grasp_pocket_center()), axis=1)
        )
        return np.asarray(
            main_proximity * capture_gate,
            dtype=self._np_dtype,
        )

    def _reward_finger_pad_progress(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Dense per-pad progress that cannot be farmed by repeated opening and closing."""
        del info, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        posture_quality, _ = _finger_posture_quality(dof_pos)
        soft_pad_scores = (
            features.pad_proximity
            * (0.20 + 0.80 * features.pad_alignment)
            * (0.30 + 0.70 * posture_quality)
        )
        main_progress = np.mean(soft_pad_scores[:, :3], axis=1)
        thumb_progress = soft_pad_scores[:, 3] * self._thumb_top_approach_quality(
            ball_pos, features.pad_to_ball[:, 3, :]
        )
        aligned_contact_progress = np.mean(
            features.contacts * features.pad_alignment,
            axis=1,
        )
        return np.asarray(
            (0.60 * main_progress + 0.30 * thumb_progress + 0.10 * aligned_contact_progress)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.10 + 0.90 * self._capture_window(ball_pos, ball_linvel)),
            dtype=self._np_dtype,
        )

    def _reward_grasp_surface_layout(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Dense geometry reward for a fanned, opposed grasp around the sphere."""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        sector_quality = _grasp_surface_layout_quality(features.local_tip_vectors)
        # Keep a broad radial gradient before contact, then demand pad alignment
        # increasingly strongly as each finger reaches the tennis-ball surface.
        radial_error = np.linalg.norm(features.pad_to_ball, axis=2) - self._finger_surface_distance
        broad_proximity = np.exp(-120.0 * np.square(radial_error))
        pad_quality = broad_proximity * (0.35 + 0.65 * features.pad_alignment)
        balanced_main = 0.55 * np.min(
            sector_quality[:, :3] * pad_quality[:, :3], axis=1
        ) + 0.45 * np.mean(sector_quality[:, :3] * pad_quality[:, :3], axis=1)
        thumb = sector_quality[:, 3] * pad_quality[:, 3]
        return np.asarray(
            (0.70 * balanced_main + 0.30 * thumb)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.05 + 0.95 * self._capture_window(ball_pos, ball_linvel)),
            dtype=self._np_dtype,
        )

    def _reward_finger_closing(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵 fingertip 相對球心往內閉合，解決手指太慢/不主動抓的問題。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        _, _, tip_vectors = self._finger_geometry(ball_pos)
        tip_distances = np.linalg.norm(tip_vectors, axis=2)
        radial_direction = tip_vectors / np.maximum(tip_distances[:, :, None], 1e-6)
        tip_velocity = np.asarray(
            self._backend.get_body_lin_vel_w(self._fingertip_body_ids),
            dtype=self._np_dtype,
        )
        relative_velocity = tip_velocity - ball_linvel[:, None, :]
        # closing_speed > 0 表示 fingertip 正朝球心靠近。
        closing_speed = -np.sum(relative_velocity * radial_direction, axis=2)
        main_closing = np.clip(closing_speed[:, :3] / 0.7, 0.0, 1.0)
        closing_quality = 0.65 * np.min(main_closing, axis=1) + 0.35 * np.mean(main_closing, axis=1)
        horizontal_error_sq = np.sum(
            np.square(ball_pos[:, :2] - self._grasp_pocket_center()[:, :2]),
            axis=1,
        )
        trajectory_gate = np.exp(-100.0 * horizontal_error_sq)
        return np.asarray(
            closing_quality * trajectory_gate * self._capture_window(ball_pos, ball_linvel),
            dtype=self._np_dtype,
        )

    def _capture_window(self, ball_pos: np.ndarray, ball_linvel: np.ndarray) -> np.ndarray:
        """球快進入 pocket 時才打開的時間窗，用來控制閉合 timing。"""
        horizontal_error_sq = np.sum(
            np.square(ball_pos[:, :2] - self._grasp_pocket_center()[:, :2]),
            axis=1,
        )
        trajectory_gate = np.exp(-120.0 * horizontal_error_sq)
        return np.asarray(
            trajectory_gate * self._catch_close_phase(ball_pos),
            dtype=self._np_dtype,
        )

    def _reward_finger_curl_velocity(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵各關節在接觸前快速 curl，避免球落下時手還慢慢動。"""
        del info, dof_pos, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        # flex_ids 只挑會讓手指彎曲/包球的關節，不把 spread 關節當成 curl。
        flex_ids = np.asarray([0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15], dtype=np.intp)
        curl_speed = np.clip(dof_vel[:, flex_ids] / 4.0, 0.0, 1.0)
        main_speed = 0.70 * np.min(curl_speed[:, :9], axis=1) + 0.30 * np.mean(
            curl_speed[:, :9], axis=1
        )
        thumb_speed = np.mean(curl_speed[:, 9:], axis=1)
        contact_started = np.clip(np.sum(features.contacts, axis=1), 0.0, 1.0)
        # 接觸前才鼓勵「快捲」；接觸後再快抖會被 stability penalty 管住。
        pre_contact_gate = 1.0 - contact_started
        return np.asarray(
            (0.75 * main_speed + 0.25 * thumb_speed)
            * self._capture_window(ball_pos, ball_linvel)
            * pre_contact_gate,
            dtype=self._np_dtype,
        )

    def _reward_reference_pose_tracking(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Exponentially reward the phase-dependent initial-to-catch reference pose."""
        del dof_vel, ball_linvel, ball_angvel, torques, terminated
        contacts = np.asarray(
            info.get("curr_contacts", np.zeros((self._num_envs, 4))),
            dtype=self._np_dtype,
        )
        reference_pose = self._catch_reference_pose(ball_pos, contacts)
        pose_error = dof_pos - reference_pose
        weighted_mse = np.mean(
            self._reference_pose_joint_weights[None, :] * np.square(pose_error),
            axis=1,
        )
        return np.asarray(
            np.exp(-float(self._cfg.reference_pose_sigma) * weighted_mse),
            dtype=self._np_dtype,
        )

    def _reward_early_closure(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize closing farther than the current ball-height reference permits."""
        del dof_vel, ball_linvel, ball_angvel, torques, terminated
        contacts = np.asarray(
            info.get("curr_contacts", np.zeros((self._num_envs, 4))),
            dtype=self._np_dtype,
        )
        phase = self._catch_close_phase(ball_pos, contacts)
        pose_delta = self._catch_pose - self._initial_pose
        trainable_ids = np.flatnonzero(
            (np.abs(pose_delta) > 1e-6)
            & ~np.isin(np.arange(self._NUM_HAND_DOF), self._locked_catch_dof_ids)
        ).astype(np.intp)
        progress = (dof_pos[:, trainable_ids] - self._initial_pose[trainable_ids]) / pose_delta[
            trainable_ids
        ]
        allowed_progress = phase[:, None] + float(self._cfg.early_closure_tolerance)
        excess = np.maximum(progress - allowed_progress, 0.0)
        weights = self._reference_pose_joint_weights[trainable_ids]
        return np.asarray(
            np.mean(weights[None, :] * np.square(excess), axis=1),
            dtype=self._np_dtype,
        )

    def _reward_finger_gap(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize any pad that stays far from the ball after closing begins."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        distance = np.linalg.norm(features.pad_to_ball, axis=2)
        excess = np.maximum(distance - self._finger_surface_distance - 0.008, 0.0) / 0.05
        balanced_gap = 0.70 * np.max(np.square(excess), axis=1) + 0.30 * np.mean(
            np.square(excess), axis=1
        )
        return np.asarray(
            balanced_gap * self._catch_close_phase(ball_pos, features.contacts),
            dtype=self._np_dtype,
        )

    def _reward_finger_synergy(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize sideways spread and flexion opposite to the authored catch motion."""
        del info, dof_vel, ball_linvel, ball_angvel, torques, terminated
        spread_reference = self._catch_reference_pose(ball_pos)[:, [1, 5, 9]]
        spread_error = np.maximum(
            np.abs(dof_pos[:, [1, 5, 9]] - spread_reference) - self._cfg.catch_spread_tolerance,
            0.0,
        )
        desired_direction = np.sign(self._catch_pose - self._initial_pose)
        flex_ids = np.asarray([0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15], dtype=np.intp)
        signed_progress = (dof_pos[:, flex_ids] - self._initial_pose[flex_ids]) * desired_direction[
            flex_ids
        ]
        reverse_motion = np.maximum(-signed_progress - 0.04, 0.0)
        return np.asarray(
            8.0 * np.mean(np.square(spread_error), axis=1)
            + 4.0 * np.mean(np.square(reverse_motion), axis=1),
            dtype=self._np_dtype,
        )

    def _reward_closing_grasp_pose(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """球快到時，鼓勵手走向非對稱的快速閉合姿勢。"""
        del info, dof_vel, ball_angvel, torques, terminated
        pose_error = dof_pos - self._closing_grasp_pose
        weights = np.asarray(
            [
                1.2,
                0.8,
                1.4,
                1.4,
                1.2,
                0.8,
                1.4,
                1.4,
                1.2,
                0.8,
                1.4,
                1.4,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=self._np_dtype,
        )
        pose_quality = np.exp(-2.5 * np.mean(weights * np.square(pose_error), axis=1))
        return np.asarray(
            pose_quality * self._capture_window(ball_pos, ball_linvel), dtype=self._np_dtype
        )

    def _reward_thumb_pregrasp(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """控制大拇指先站到 opposition 位置，再參與包球。"""
        del info, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        pregrasp_error_sq = np.sum(
            np.square(dof_pos[:, 12:16] - self._thumb_pregrasp_pose),
            axis=1,
        )
        pregrasp_quality = np.exp(-8.0 * pregrasp_error_sq)
        # Once contact begins, keep the thumb base behind the sphere while
        # leaving its two distal joints free to close around it.
        opposition_error_sq = np.sum(
            np.square(dof_pos[:, 12:14] - self._thumb_pregrasp_pose[:2]),
            axis=1,
        )
        opposition_quality = np.exp(-12.0 * opposition_error_sq)
        height_above_pocket = ball_pos[:, 2] - features.pocket_center[:, 2]
        approach_phase = np.clip(height_above_pocket / 0.12, 0.0, 1.0)
        no_thumb_contact = 1.0 - features.contacts[:, 3]
        return np.asarray(
            pregrasp_quality * approach_phase * no_thumb_contact
            + opposition_quality * features.contacts[:, 3],
            dtype=self._np_dtype,
        )

    def _reward_thumb_approach(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """在接觸前鼓勵大拇指從對掌方向靠近球，避免最後只有三指接。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        pad_scores = features.pad_proximity * features.pad_alignment
        thumb_near = np.maximum(
            pad_scores[:, 3], features.contacts[:, 3] * features.pad_alignment[:, 3]
        )
        main_ready = 0.50 * np.min(pad_scores[:, :3], axis=1) + 0.50 * np.mean(
            pad_scores[:, :3], axis=1
        )
        thumb_lane = self._thumb_opposition_lane(features)
        return np.asarray(
            thumb_near
            * (0.35 + 0.65 * main_ready)
            * thumb_lane
            * self._capture_window(ball_pos, ball_linvel),
            dtype=self._np_dtype,
        )

    def _reward_thumb_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """大拇指真的碰到球且在對掌側時給中間分，幫 finger_wrap 起步。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        main_pad_scores = features.pad_proximity[:, :3] * features.pad_alignment[:, :3]
        main_support_near = 0.65 * np.min(main_pad_scores, axis=1) + 0.35 * np.mean(
            main_pad_scores, axis=1
        )
        return np.asarray(
            features.contacts[:, 3]
            * features.pad_alignment[:, 3]
            * self._thumb_world_top_quality(features.pad_to_ball[:, 3, :])
            * self._thumb_opposition_lane(features)
            * (0.40 + 0.60 * main_support_near)
            * features.pocket_quality,
            dtype=self._np_dtype,
        )

    def _reward_thumb_pad_press(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵大拇指紅色側/指腹方向貼近球並壓住，而不是只懸在旁邊。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        thumb_pad_proximity, thumb_pad_alignment, pad_to_ball = self._finger_pad_proxy(ball_pos, 3)
        thumb_pad_press = (
            thumb_pad_proximity
            * thumb_pad_alignment
            * self._thumb_world_top_quality(pad_to_ball)
            * (0.10 + 0.90 * features.contacts[:, 3])
        )
        main_pad_scores = features.pad_proximity[:, :3] * features.pad_alignment[:, :3]
        main_ready = 0.55 * np.min(main_pad_scores, axis=1) + 0.45 * np.mean(
            main_pad_scores, axis=1
        )
        return np.asarray(
            thumb_pad_press
            * (0.35 + 0.65 * self._thumb_opposition_lane(features))
            * (0.35 + 0.65 * main_ready)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.40 + 0.60 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_thumb_top_approach(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Dense shaping that moves the thumb pad toward the top of the ball."""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        _, pad_alignment, pad_to_ball = self._finger_pad_proxy(ball_pos, 3)
        top_approach_quality = self._thumb_top_approach_quality(ball_pos, pad_to_ball)
        return np.asarray(
            top_approach_quality
            * (0.25 + 0.75 * pad_alignment)
            * self._capture_window(ball_pos, ball_linvel)
            * (0.30 + 0.70 * self._finger_near_quality(features)),
            dtype=self._np_dtype,
        )

    def _reward_thumb_over_ball(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Reward the thumb pad moving onto the top of the ball."""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        _, pad_alignment, pad_to_ball = self._finger_pad_proxy(ball_pos, 3)
        top_quality = self._thumb_world_top_quality(pad_to_ball)
        support = self._finger_support_quality(features)
        return np.asarray(
            top_quality
            * (0.35 + 0.65 * pad_alignment)
            * (0.35 + 0.65 * support)
            * self._capture_window(ball_pos, ball_linvel),
            dtype=self._np_dtype,
        )

    def _reward_thumb_missing_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize leaving the thumb open after the other fingers support the ball."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        main_contact_progress = np.mean(features.contacts[:, :3], axis=1)
        main_pad_scores = features.pad_proximity[:, :3] * features.pad_alignment[:, :3]
        main_near_progress = np.mean(main_pad_scores, axis=1)
        main_support = 0.65 * main_contact_progress + 0.35 * main_near_progress
        thumb_pad_proximity, thumb_pad_alignment, _ = self._finger_pad_proxy(ball_pos, 3)
        thumb_pad_support = (
            0.75 * features.contacts[:, 3] * thumb_pad_alignment
            + 0.25 * thumb_pad_proximity * thumb_pad_alignment
        )
        return np.asarray(
            main_support
            * np.clip(1.0 - thumb_pad_support, 0.0, 1.0)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.30 + 0.70 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_finger_pad_press(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵三根主手指都貼近球面，避免其中一根離球太遠造成掉落風險。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        main_pad_scores = []
        for finger_index in range(3):
            pad_proximity, pad_alignment, _ = self._finger_pad_proxy(ball_pos, finger_index)
            main_pad_scores.append(pad_proximity * (0.20 + 0.80 * pad_alignment))
        main_pad_scores = np.stack(main_pad_scores, axis=1)
        all_main_close = 0.40 * np.min(main_pad_scores, axis=1) + 0.60 * np.mean(
            main_pad_scores, axis=1
        )
        contact_progress = np.mean(features.contacts[:, :3], axis=1)
        return np.asarray(
            all_main_close
            * (0.25 + 0.75 * contact_progress)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.50 + 0.50 * features.ring_side_quality)
            * (0.35 + 0.65 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_main_finger_fan_pose(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Reward the three main fingers keeping a natural fan instead of pointing sideways."""
        del info, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        spread = dof_pos[:, [1, 5, 9]]
        target_spread = self._natural_grasp_pose[[1, 5, 9]]
        spread_error = spread - target_spread
        fan_quality = np.exp(-18.0 * np.mean(np.square(spread_error), axis=1))
        return np.asarray(
            fan_quality
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.35 + 0.65 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_thumb_pad_alignment(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Reward the thumb tip approaching from the opposing pad side."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        thumb_pad_proximity, pad_alignment, pad_to_ball = self._finger_pad_proxy(ball_pos, 3)
        top_contact_quality = self._thumb_world_top_quality(pad_to_ball)
        top_approach_quality = self._thumb_top_approach_quality(ball_pos, pad_to_ball)
        thumb_press = pad_alignment * (
            0.35 * top_approach_quality + 0.65 * thumb_pad_proximity * top_contact_quality
        )
        main_pad_scores = features.pad_proximity[:, :3] * features.pad_alignment[:, :3]
        main_ready = 0.45 * np.min(main_pad_scores, axis=1) + 0.55 * np.mean(
            main_pad_scores, axis=1
        )
        return np.asarray(
            thumb_press
            * (0.35 + 0.65 * self._thumb_opposition_lane(features))
            * (0.30 + 0.70 * main_ready)
            * self._palm_or_pocket_gate(ball_pos, features),
            dtype=self._np_dtype,
        )

    def _reward_ring_pad_press(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Give the outer main finger its own dense reward for staying close to the ball."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        ring_pad_proximity, ring_pad_alignment, _ = self._finger_pad_proxy(ball_pos, 2)
        ring_press = ring_pad_proximity * (0.20 + 0.80 * ring_pad_alignment)
        return np.asarray(
            ring_press
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.45 + 0.55 * features.ring_side_quality)
            * (0.35 + 0.65 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_bad_pad_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize contact that is not made with the finger pad facing the ball."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        pad_scores = features.pad_proximity * features.pad_alignment
        bad_contact = features.contacts * np.clip(1.0 - pad_scores, 0.0, 1.0)
        return np.asarray(
            np.mean(bad_contact, axis=1) * self._palm_or_pocket_gate(ball_pos, features),
            dtype=self._np_dtype,
        )

    def _reward_thumb_nail_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize touching the ball with the back/nail side of the thumb tip."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        return np.asarray(
            features.contacts[:, 3]
            * np.clip(1.0 - features.pad_alignment[:, 3], 0.0, 1.0)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.35 + 0.65 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_thumb_misplaced(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize thumb contact/proximity that is not on the top of the ball."""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        thumb_pad_proximity, thumb_pad_alignment, pad_to_ball = self._finger_pad_proxy(ball_pos, 3)
        thumb_activity = np.maximum(
            features.contacts[:, 3], thumb_pad_proximity * thumb_pad_alignment
        )
        top_quality = self._thumb_world_top_quality(pad_to_ball)
        return np.asarray(
            thumb_activity
            * np.clip(1.0 - top_quality, 0.0, 1.0)
            * self._palm_or_pocket_gate(ball_pos, features),
            dtype=self._np_dtype,
        )

    def _reward_ring_inward_closing(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Reward the outer main finger moving inward toward the ball before capture."""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        tip_vectors = features.tip_vectors
        distances = np.maximum(np.linalg.norm(tip_vectors, axis=2), 1e-6)
        radial_dirs = tip_vectors / distances[:, :, None]
        ring_tip_vel = self._backend.get_body_lin_vel_w(self._fingertip_body_ids)[:, 2, :]
        ring_rel_vel = ring_tip_vel - ball_linvel
        ring_closing_speed = -np.sum(ring_rel_vel * radial_dirs[:, 2, :], axis=1)
        ring_closing_quality = np.clip(ring_closing_speed / 0.60, 0.0, 1.0)
        return np.asarray(
            ring_closing_quality
            * self._capture_window(ball_pos, ball_linvel)
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.40 + 0.60 * features.ring_side_quality),
            dtype=self._np_dtype,
        )

    def _reward_grasp_tightness(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """四指與掌根同時形成貼合支撐時加分，讓球不只是停住而是被握住。"""
        del info, dof_pos, dof_vel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        soft_alignment = 0.10 + 0.90 * features.pad_alignment
        pad_scores = features.pad_proximity * soft_alignment
        aligned_contacts = features.contacts * soft_alignment
        main_touch_or_near = np.maximum(pad_scores[:, :3], aligned_contacts[:, :3])
        main_close = np.min(main_touch_or_near, axis=1)
        thumb_press = np.maximum(pad_scores[:, 3], aligned_contacts[:, 3])
        slow_ball = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        low_spin = np.exp(-0.15 * np.sum(np.square(ball_angvel), axis=1))
        return np.asarray(
            main_close
            * thumb_press
            * self._thumb_opposition_lane(features)
            * self._palm_or_pocket_gate(ball_pos, features)
            * features.pocket_quality
            * slow_ball
            * low_spin,
            dtype=self._np_dtype,
        )

    def _reward_natural_grasp_pose(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵穩定握住後的自然握球姿勢，避免手指扭曲成奇怪形狀。"""
        del info, dof_vel, ball_angvel, torques, terminated
        pose_error = dof_pos - self._natural_grasp_pose
        weights = np.asarray(
            [
                0.7,
                1.2,
                1.0,
                1.0,
                0.7,
                1.2,
                1.0,
                1.0,
                0.7,
                1.2,
                1.0,
                1.0,
                1.6,
                1.8,
                1.2,
                1.0,
            ],
            dtype=self._np_dtype,
        )
        pose_quality = np.exp(-3.0 * np.mean(weights * np.square(pose_error), axis=1))
        horizontal_error_sq = np.sum(
            np.square(ball_pos[:, :2] - self._grasp_pocket_center()[:, :2]),
            axis=1,
        )
        trajectory_gate = np.exp(-120.0 * horizontal_error_sq)
        time_to_contact = self._time_to_contact(ball_pos, ball_linvel)
        capture_window = np.clip((0.45 - time_to_contact) / 0.30, 0.0, 1.0)
        return np.asarray(pose_quality * trajectory_gate * capture_window, dtype=self._np_dtype)

    def _reward_opposition_geometry(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵大拇指和主三指形成對掌夾持，而不是同側亂碰。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        return np.asarray(
            features.opposition_quality * features.pocket_quality,
            dtype=self._np_dtype,
        )

    def _reward_ring_support(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """讓無名指側也要出力支撐，避免小拇指/無名指完全沒作用。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        ring_contact = features.contacts[:, 2]
        ring_proximity = features.pad_proximity[:, 2]
        ring_alignment = features.pad_alignment[:, 2]
        return np.asarray(
            (0.60 * ring_contact + 0.40 * ring_proximity)
            * ring_alignment
            * features.ring_side_quality
            * features.pocket_quality,
            dtype=self._np_dtype,
        )

    def _reward_wrap_progress(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """比 finger_wrap 更軟的階梯分，讓 policy 慢慢學到四指包覆。"""
        del info, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        posture_quality, _ = _finger_posture_quality(dof_pos)
        pad_readiness = features.pad_proximity * (0.20 + 0.80 * features.pad_alignment)
        staged_progress = _staged_wrap_progress(
            features.contacts * posture_quality,
            pad_readiness,
            features.soft_enclosure_quality,
        )
        capture_region = self._palm_or_pocket_gate(ball_pos, features)
        return np.asarray(
            staged_progress
            * (0.20 + 0.80 * capture_region)
            * (0.25 + 0.75 * features.pocket_quality),
            dtype=self._np_dtype,
        )

    def _reward_finger_wrap(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """核心 reward：球要被手指方向包覆，並且位於 pocket。"""
        del info, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        posture_quality, _ = _finger_posture_quality(dof_pos)
        wrap_quality = np.maximum(features.enclosure_quality, features.soft_enclosure_quality)
        all_pad_contact = _four_pad_contact_quality(
            features.contacts,
            features.pad_proximity,
            features.pad_alignment,
            posture_quality,
        )
        return np.asarray(
            wrap_quality * all_pad_contact * features.opposition_quality * features.pocket_quality,
            dtype=self._np_dtype,
        )

    def _reward_pre_bounce_catch(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵第一次進 pocket 時就由手指吸收速度，而不是先彈跳再撿。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        contact_progress = np.clip(np.sum(features.contacts, axis=1) / 4.0, 0.0, 1.0)
        speed_quality = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        downward_gate = np.clip((-ball_linvel[:, 2] + 0.05) / 0.80, 0.0, 1.0)
        palm_gate = self._palm_or_pocket_gate(ball_pos, features)
        return np.asarray(
            self._capture_window(ball_pos, ball_linvel)
            * contact_progress
            * self._finger_support_quality(features)
            * palm_gate
            * speed_quality
            * downward_gate,
            dtype=self._np_dtype,
        )

    def _reward_hold(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """球已被包覆後，鼓勵低線速度/低角速度地穩定留在 pocket。"""
        del info, dof_vel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        posture_quality, _ = _finger_posture_quality(dof_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        all_pad_contact = _four_pad_contact_quality(
            features.contacts,
            features.pad_proximity,
            features.pad_alignment,
            posture_quality,
        )
        linear_stability = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        angular_stability = np.exp(-0.15 * np.sum(np.square(ball_angvel), axis=1))
        return np.asarray(
            features.pocket_quality
            * wrap_quality
            * all_pad_contact
            * features.opposition_quality
            * linear_stability
            * angular_stability,
            dtype=self._np_dtype,
        )

    def _reward_touch_without_wrap(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰只有碰到球、但沒有形成包覆的假成功。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        any_contact = np.any(features.contacts > 0.5, axis=1).astype(self._np_dtype)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        return np.asarray(
            any_contact * (1.0 - wrap_quality) * features.pocket_quality,
            dtype=self._np_dtype,
        )

    def _reward_palm_only(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰球只是穩穩躺在掌心/平面上，手指沒有包住。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        finger_support = self._finger_support_quality(features)
        slow_ball = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        return np.asarray(
            features.pocket_quality
            * slow_ball
            * (1.0 - wrap_quality)
            * (1.0 - finger_support)
            * (0.40 + 0.60 * features.palm_contact),
            dtype=self._np_dtype,
        )

    def _reward_palm_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """掌心接觸但沒有手指支撐時扣分；掌心參與正常握球時不扣。"""
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        finger_support = self._finger_support_quality(features)
        return np.asarray(
            features.palm_contact * (1.0 - wrap_quality) * (1.0 - finger_support),
            dtype=self._np_dtype,
        )

    def _reward_palm_guidance(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """早期 shaping：把球導向掌根/指根，而不是只用三指在外側擋球。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        palm_near = self._loose_pocket_quality(ball_pos, features)
        downward_gate = np.clip((-ball_linvel[:, 2] + 0.05) / 0.80, 0.0, 1.0)
        return np.asarray(
            palm_near
            * (0.30 + 0.70 * features.palm_contact)
            * (0.35 + 0.65 * self._finger_near_quality(features))
            * downward_gate,
            dtype=self._np_dtype,
        )

    def _reward_palm_assist(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵球貼近掌根/指根，同時由手指支撐形成真握持。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        finger_support = self._finger_support_quality(features)
        slow_ball = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        return np.asarray(
            features.palm_contact
            * features.pocket_quality
            * finger_support
            * (0.50 + 0.50 * wrap_quality)
            * slow_ball,
            dtype=self._np_dtype,
        )

    def _reward_ball_bounce(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰球在手附近任何向上彈跳。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        finger_support = self._finger_support_quality(features)
        # 只要球的 z 速度轉正，就代表它正在往上彈。這裡不留 dead-zone，
        # 因為接球任務不希望 policy 學到「先彈一下再抓」。
        upward_bounce = np.clip(ball_linvel[:, 2], 0.0, 2.0)
        palm_gate = self._palm_or_pocket_gate(ball_pos, features)
        any_finger_contact = np.clip(np.sum(features.contacts, axis=1), 0.0, 1.0)
        near_hand = np.maximum(palm_gate, any_finger_contact)
        fake_bounce_gate = 1.0 - palm_gate * finger_support
        return np.asarray(
            upward_bounce
            * near_hand
            * (0.25 + 0.75 * (1.0 - wrap_quality))
            * (0.25 + 0.75 * fake_bounce_gate)
            * (0.50 + 0.50 * features.palm_contact),
            dtype=self._np_dtype,
        )

    def _reward_pocket_without_finger_contact(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰球已經在 pocket 附近、但手指沒有真正支撐球的策略。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        finger_support = self._finger_support_quality(features)
        slow_ball = np.exp(-3.0 * np.sum(np.square(ball_linvel), axis=1))
        return np.asarray(
            features.pocket_quality * slow_ball * (1.0 - finger_support),
            dtype=self._np_dtype,
        )

    def _reward_pocket_depth(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """鼓勵球心下降到掌根/指根 pocket 高度，而不是停在三指上方。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        vertical_error = self._vertical_pocket_error(ball_pos, features)
        horizontal_error = ball_pos[:, :2] - features.pocket_center[:, :2]
        horizontal_quality = np.exp(-90.0 * np.sum(np.square(horizontal_error), axis=1))
        # 目標不是讓球掉到掌心下面，而是讓球心接近或略低於 pocket 高度。
        depth_quality = np.exp(-900.0 * np.square(np.maximum(vertical_error + 0.005, 0.0)))
        too_low_guard = np.exp(-700.0 * np.square(np.maximum(-0.060 - vertical_error, 0.0)))
        controlled_descent = np.clip((-ball_linvel[:, 2] + 0.10) / 0.90, 0.0, 1.0)
        return np.asarray(
            horizontal_quality
            * depth_quality
            * too_low_guard
            * (0.40 + 0.60 * self._finger_near_quality(features))
            * controlled_descent,
            dtype=self._np_dtype,
        )

    def _reward_high_ball(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰球心停在 pocket 上方，典型就是卡在三指上面。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        vertical_error = self._vertical_pocket_error(ball_pos, features)
        horizontal_error = ball_pos[:, :2] - features.pocket_center[:, :2]
        horizontal_near = np.exp(-70.0 * np.sum(np.square(horizontal_error), axis=1))
        high_amount = np.clip((vertical_error - 0.025) / 0.080, 0.0, 1.0)
        contact_gate = np.clip(np.sum(features.contacts, axis=1), 0.0, 1.0)
        return np.asarray(
            high_amount
            * horizontal_near
            * (0.35 + 0.65 * contact_gate)
            * (1.0 - features.palm_contact),
            dtype=self._np_dtype,
        )

    def _reward_drop_risk(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """球已在掌根附近但手指貼合不足、仍有下滑速度時扣分。"""
        del info, dof_pos, dof_vel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        downward_speed = np.clip(-ball_linvel[:, 2], 0.0, 2.0)
        support_gap = 1.0 - self._finger_support_quality(features)
        tight_gap = 1.0 - np.minimum(
            np.min(np.maximum(features.per_finger[:, :3], features.contacts[:, :3]), axis=1),
            np.maximum(features.per_finger[:, 3], features.contacts[:, 3]),
        )
        return np.asarray(
            downward_speed
            * self._palm_or_pocket_gate(ball_pos, features)
            * (0.55 * support_gap + 0.45 * tight_gap),
            dtype=self._np_dtype,
        )

    def _reward_grasp_stability(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """球已穩定被包住時，懲罰關節高速抖動。"""
        del info, dof_pos, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        wrap_quality = np.maximum(features.wrap, features.soft_enclosure_quality)
        slow_ball = np.exp(-4.0 * np.sum(np.square(ball_linvel), axis=1))
        stable_gate = features.pocket_quality * wrap_quality * slow_ball
        motion_penalty = np.mean(np.square(dof_vel), axis=1)
        return np.asarray(stable_gate * motion_penalty, dtype=self._np_dtype)

    def _settled_grasp_gate(
        self,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        features: _CatchRewardFeatures,
    ) -> np.ndarray:
        """Activate stabilization only after the ball is supported near the pocket."""
        contact_progress = np.mean(features.contacts, axis=1)
        finger_support = self._finger_support_quality(features)
        support_gate = np.maximum(contact_progress, finger_support)
        slow_ball = np.exp(-4.0 * np.sum(np.square(ball_linvel), axis=1))
        return np.asarray(
            self._palm_or_pocket_gate(ball_pos, features) * support_gate * slow_ball,
            dtype=self._np_dtype,
        )

    def _reward_finger_jitter(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize finger velocity after the ball becomes supported near the pocket."""
        del info, dof_pos, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        normalized_motion = np.clip(
            np.mean(np.square(dof_vel / 3.0), axis=1),
            0.0,
            4.0,
        )
        return np.asarray(
            normalized_motion * self._settled_grasp_gate(ball_pos, ball_linvel, features),
            dtype=self._np_dtype,
        )

    def _reward_action_rate(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Penalize step-to-step policy command oscillation."""
        del dof_pos, dof_vel, ball_angvel, torques, terminated
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._NUM_HAND_DOF))),
            dtype=self._np_dtype,
        )
        last_actions = np.asarray(
            info.get("last_actions", np.zeros_like(current_actions)),
            dtype=self._np_dtype,
        )
        features = self._reward_features or self._compute_catch_features(ball_pos)
        settled_gate = self._settled_grasp_gate(ball_pos, ball_linvel, features)
        return np.asarray(
            np.mean(np.square(current_actions - last_actions), axis=1) * settled_gate,
            dtype=self._np_dtype,
        )

    def _reward_settled_action(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Drive cumulative-control actions back to zero once the grasp is stable."""
        del dof_pos, dof_vel, ball_angvel, torques, terminated
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._NUM_HAND_DOF))),
            dtype=self._np_dtype,
        )
        features = self._reward_features or self._compute_catch_features(ball_pos)
        return np.asarray(
            np.mean(np.square(current_actions), axis=1)
            * self._settled_grasp_gate(ball_pos, ball_linvel, features),
            dtype=self._np_dtype,
        )

    def _reward_thumb_crossing(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰大拇指穿過球前方、和其他手指交叉的姿勢。"""
        del info, dof_pos, dof_vel, ball_linvel, ball_angvel, torques, terminated
        features = self._reward_features or self._compute_catch_features(ball_pos)
        any_contact = np.any(features.contacts > 0.5, axis=1).astype(self._np_dtype)
        thumb_index_gap = np.linalg.norm(
            features.local_tip_vectors[:, 3, :] - features.local_tip_vectors[:, 0, :], axis=1
        )
        index_collision_penalty = np.square(np.maximum(0.055 - thumb_index_gap, 0.0)) * 500.0
        return np.asarray(
            (features.crossing_penalty + index_collision_penalty) * (0.25 + 0.75 * any_contact),
            dtype=self._np_dtype,
        )

    def _reward_unnatural_joint_pose(
        self,
        info: dict[str, Any],
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_linvel: np.ndarray,
        ball_angvel: np.ndarray,
        torques: np.ndarray,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """懲罰不自然/超限的手指姿態，避免為了 reward 把手指扭壞。"""
        del info, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        _, posture_penalty = _finger_posture_quality(dof_pos)
        limit_center = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)
        limit_violation = np.maximum(np.abs(limit_center) - 0.90, 0.0)
        return np.asarray(
            posture_penalty + 0.25 * np.sum(np.square(limit_violation), axis=1),
            dtype=self._np_dtype,
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """每步更新 state.info，把 contact 資訊交給 reward 和 observation 使用。"""
        contacts = self._finger_contacts()
        palm_contact = self._palm_contact()
        state.info["curr_contacts"] = contacts
        state.info["curr_palm_contact"] = palm_contact
        updated = super().update_state(state)
        updated.info["prev_contacts"] = contacts.copy()
        updated.info["prev_palm_contact"] = palm_contact.copy()
        return updated

    def _build_current_obs(
        self, info: dict[str, Any], dof_pos: np.ndarray, ball_pos: np.ndarray
    ) -> np.ndarray:
        """組 observation。

        在 base rotation obs 後面補上：
        - ball linear velocity：policy 要知道球正在怎麼掉。
        - four finger contacts：policy 要知道哪些手指已碰到。
        - palm contact：policy 要知道是不是只靠掌心托住。
        - normalized time-to-contact：policy 要知道什麼時候該快速閉合。
        """
        base_rotation_obs = super()._build_current_obs(info, dof_pos, ball_pos)
        # Rotation now intentionally keeps Allegro's 35-D policy contract.
        # Catch owns the extra object orientation/rotation signals it needs.
        ball_quat = np.asarray(
            info.get("curr_ball_quat", info["prev_ball_quat"]),
            dtype=get_global_dtype(),
        )
        ball_angvel = np.asarray(
            info.get("curr_ball_angvel", info["prev_ball_angvel"]),
            dtype=get_global_dtype(),
        )
        rotation_obs = np.concatenate(
            [base_rotation_obs, ball_quat, ball_angvel],
            axis=1,
            dtype=get_global_dtype(),
        )
        ball_linvel = np.asarray(
            info.get("curr_ball_linvel", info["prev_ball_linvel"]),
            dtype=get_global_dtype(),
        )
        contacts = np.asarray(
            info.get(
                "curr_contacts",
                info.get(
                    "prev_contacts",
                    np.zeros((dof_pos.shape[0], len(self._CONTACT_SENSORS))),
                ),
            ),
            dtype=get_global_dtype(),
        )
        palm_contact = np.asarray(
            info.get(
                "curr_palm_contact",
                info.get("prev_palm_contact", np.zeros((dof_pos.shape[0],))),
            ),
            dtype=get_global_dtype(),
        )[:, None]
        normalized_time_to_contact = np.clip(
            self._time_to_contact(ball_pos, ball_linvel) / 0.3,
            0.0,
            1.0,
        )[:, None]
        return np.concatenate(
            [rotation_obs, ball_linvel, contacts, palm_contact, normalized_time_to_contact],
            axis=1,
            dtype=get_global_dtype(),
        )
