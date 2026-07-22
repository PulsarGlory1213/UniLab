"""LEAP Hand 球體指尖旋轉任務的完整獨立實作。

包含設定、cache reset、domain randomization、reward、observation 與 state update。
此模組不繼承或 import Allegro 任務。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_common_reset_randomization,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.utils.geometry import (
    np_normalize_axis,
    np_quat_angular_velocity_from_pair,
)

from .base import LeapHandBaseCfg, LeapHandBaseEnv


def resolve_grasp_cache_path(cache_path: str) -> epath.Path:
    """依照 UniLab asset root 規則解析 LEAP grasp cache 路徑。"""
    path = epath.Path(cache_path)
    if path.is_absolute() or path.exists():
        return path
    return epath.Path(ASSETS_ROOT_PATH / cache_path)


def normalize_rotation_axis(rotation_axis: tuple[float, float, float]) -> np.ndarray:
    # Cast to the training dtype first so the norm and division happen at that
    # precision, matching the pre-refactor bit-exact behavior for float32 runs.
    axis = np.asarray(rotation_axis, dtype=get_global_dtype())
    return np.asarray(np_normalize_axis(axis), dtype=get_global_dtype())


def compute_ball_angvel(
    ball_quat: np.ndarray, prev_ball_quat: np.ndarray, ctrl_dt: float
) -> np.ndarray:
    return np.asarray(
        np_quat_angular_velocity_from_pair(ball_quat, prev_ball_quat, ctrl_dt),
        dtype=get_global_dtype(),
    )


def compute_pd_torques(
    targets: np.ndarray, dof_pos: np.ndarray, dof_vel: np.ndarray, kp: float, kd: float
) -> np.ndarray:
    torques = kp * (targets - dof_pos) - kd * dof_vel
    return np.asarray(np.clip(torques, -0.5, 0.5), dtype=get_global_dtype())


def build_obs_lag_history(
    init_obs: np.ndarray, num_lag_steps: int, num_obs_per_step: int
) -> np.ndarray:
    num_envs = init_obs.shape[0]
    history = np.broadcast_to(
        init_obs[:, None, :],
        (num_envs, num_lag_steps, num_obs_per_step),
    ).copy()
    return np.asarray(history, dtype=init_obs.dtype)


def sample_cached_grasps(
    grasp_cache: np.ndarray, num_reset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.randint(0, len(grasp_cache), size=num_reset)
    sampled = grasp_cache[idx]
    return sampled[:, :16], sampled[:, 16:19], sampled[:, 19:23]


def validate_grasp_cache(grasp_cache: np.ndarray, source: str = "<memory>") -> np.ndarray:
    """Validate the LEAP cache contract: 16 hand joints + 3 position + 4 quaternion."""
    cache = np.asarray(grasp_cache)
    if cache.ndim != 2 or cache.shape[1] != 23:
        raise ValueError(
            f"Invalid LEAP grasp cache {source}: expected shape (N, 23), got {cache.shape}"
        )
    if cache.shape[0] == 0:
        raise ValueError(f"Invalid LEAP grasp cache {source}: the cache is empty")
    if not np.all(np.isfinite(cache)):
        raise ValueError(f"Invalid LEAP grasp cache {source}: values must all be finite")
    return cache.astype(np.float64, copy=False)


@dataclass
class RewardConfigPPO:
    scales: dict[str, float]
    angvel_clip_min: float
    angvel_clip_max: float
    reset_z_threshold: float
    rotation_warmup_seconds: float = 0.0
    reverse_angvel_clip_max: float = 2.0
    minimum_positive_angvel: float = 0.15
    rotation_window_seconds: float = 1.0
    rotation_streak_target_seconds: float = 2.0
    stall_grace_seconds: float = 1.0
    contact_rotation_min_contacts: int = 2


@dataclass
class DomainRandConfig:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [0.0, 0.0])
    randomize_gravity: bool = False
    gravity_range: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]]
    )
    push_robots: bool = False
    push_interval: int = 750
    max_force: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
    push_body_name: str | None = None
    joint_noise: float = 0.0
    ball_vel_noise: float = 0.0
    ball_z_offset: float = 0.0


@registry.envcfg("LeapInhandRotation")
@dataclass
class LeapInhandRotationCfg(LeapHandBaseCfg):
    # LEAP robot XML 與 rotation task fragment 分離，backend 由 Hydra owner YAML 決定。
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "leap_hand.xml"),
            fragment_files=[str(ASSETS_ROOT_PATH / "robots" / "leap_hand" / "scene.xml")],
        )
    )
    base_body_name: str = "palm_lower"
    max_episode_seconds: float = 20.0
    reward_config: RewardConfigPPO | None = None
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_cycle_seconds: float = 2.0
    grasp_cache_path: str = "caches/leap_hand_seed_bank.npy"
    use_grasp_cache: bool = True
    grasp_cache_sample_mode: str = "random"
    grasp_cache_start_index: int = 0
    gen_grasp: bool = False


class LeapRotationDomainRandomizationProvider(DomainRandomizationProvider):
    """負責 reset/cache sampling 與 backend-independent domain randomization。"""

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(env, capabilities)
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return build_interval_push_plan(env, step_counter)

    def _load_grasp_cache(self, env: Any) -> np.ndarray | None:
        if env._grasp_cache_loaded:
            return cast(np.ndarray | None, env._grasp_cache)
        if env.cfg.gen_grasp or not env.cfg.use_grasp_cache:
            env._grasp_cache = None
            env._grasp_cache_loaded = True
            return None

        cache_path = resolve_grasp_cache_path(env.cfg.grasp_cache_path)
        if not cache_path.exists():
            print(
                "[leap_inhand] Grasp cache is missing; no Hugging Face download will be "
                f"attempted. Expected local cache: {cache_path}. Generate one with "
                "`uv run train --algo ppo --task leap_inhand_grasp --sim motrix "
                "training.no_play=true`, or point `env.grasp_cache_path` at an existing "
                "local cache."
            )
            env._grasp_cache = None
            env._grasp_cache_loaded = True
            return None
        env._grasp_cache = validate_grasp_cache(np.load(cache_path), str(cache_path))
        env._grasp_cache_loaded = True
        print(
            "[leap_inhand] Loaded grasp cache: "
            f"{cache_path}, shape={env._grasp_cache.shape}, dtype={env._grasp_cache.dtype}"
        )
        return cast(np.ndarray | None, env._grasp_cache)

    def _sample_reset_state(
        self, env: Any, num_reset: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dr = env.cfg.domain_rand
        grasp_cache = self._load_grasp_cache(env)
        if grasp_cache is not None:
            sample_mode = str(env.cfg.grasp_cache_sample_mode)
            if sample_mode == "random":
                hand_qpos, ball_pos, ball_quat = sample_cached_grasps(grasp_cache, num_reset)
            elif sample_mode == "sequential":
                start = int(env._grasp_cache_cursor)
                indices = (start + np.arange(num_reset, dtype=np.int64)) % len(grasp_cache)
                sampled = grasp_cache[indices]
                hand_qpos, ball_pos, ball_quat = (
                    sampled[:, :16],
                    sampled[:, 16:19],
                    sampled[:, 19:23],
                )
                env._grasp_cache_cursor = (start + num_reset) % len(grasp_cache)
            else:
                raise ValueError(
                    f"grasp_cache_sample_mode must be 'random' or 'sequential', got {sample_mode!r}"
                )
        else:
            hand_qpos = np.broadcast_to(env.default_angles, (num_reset, env._NUM_HAND_DOF)).copy()
            hand_qpos += np.random.uniform(-dr.joint_noise, dr.joint_noise, hand_qpos.shape).astype(
                np.float64
            )
            hand_qpos = np.clip(
                hand_qpos,
                env._ctrl_lower.astype(np.float64),
                env._ctrl_upper.astype(np.float64),
            )
            ball_init_pos = env._init_qpos[env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3]
            ball_pos = np.broadcast_to(ball_init_pos, (num_reset, 3)).copy()
            ball_pos[:, 2] += dr.ball_z_offset
            ball_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_reset, 1))

        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)
        qvel[:, env._NUM_HAND_DOF : env._NUM_HAND_DOF + 3] = np.random.uniform(
            -dr.ball_vel_noise,
            dr.ball_vel_noise,
            (num_reset, 3),
        )
        return hand_qpos, ball_pos, ball_quat, qvel

    def _build_info_updates(
        self,
        env: Any,
        hand_qpos: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        num_reset = hand_qpos.shape[0]
        dtype = get_global_dtype()

        init_ctrl = np.asarray(hand_qpos, dtype=dtype)
        init_ball_pos = np.asarray(ball_pos, dtype=dtype)
        info_updates = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "prev_ctrl": init_ctrl,
            "init_pose": init_ctrl.copy(),
            "prev_dof_pos": init_ctrl.copy(),
            "prev_ball_pos": init_ball_pos.copy(),
            "prev_ball_quat": np.asarray(ball_quat, dtype=dtype).copy(),
            "prev_ball_linvel": np.zeros((num_reset, 3), dtype=dtype),
            "prev_ball_angvel": np.zeros((num_reset, 3), dtype=dtype),
            "curr_ball_angvel": np.zeros((num_reset, 3), dtype=dtype),
            "curr_fingertip_contacts": np.zeros(
                (num_reset, len(env._CONTACT_SENSORS)), dtype=dtype
            ),
            "contact_duration_steps": np.zeros(
                (num_reset, len(env._CONTACT_SENSORS)), dtype=dtype
            ),
            "rotation_streak_steps": np.zeros(num_reset, dtype=np.uint32),
            "rotation_window": np.zeros(
                (num_reset, env._rotation_window_steps), dtype=dtype
            ),
            "rotation_window_sum": np.zeros(num_reset, dtype=dtype),
        }
        init_obs = env._build_current_obs(info_updates, init_ctrl, init_ball_pos)
        info_updates["obs_lag_history"] = build_obs_lag_history(
            init_obs, env._NUM_LAG_STEPS, env._NUM_OBS_PER_STEP
        )
        return info_updates

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        hand_qpos, ball_pos, ball_quat, qvel = self._sample_reset_state(env, num_reset)
        qpos = np.concatenate([hand_qpos, ball_pos, ball_quat], axis=1, dtype=np.float64)
        info_updates = self._build_info_updates(env, hand_qpos, ball_pos, ball_quat)

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(env, num_reset),
        )

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        info_updates["curr_fingertip_rel"] = np.asarray(
            env.get_fingertip_pos()[env_ids]
            - info_updates["prev_ball_pos"][:, None, :],
            dtype=get_global_dtype(),
        )
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                info_updates["prev_ctrl"],
                info_updates["prev_ball_pos"],
            ),
        )


# ─────────────────────────── Environment ──────────────────────────────


@registry.env("LeapInhandRotation", sim_backend="mujoco")
@registry.env("LeapInhandRotation", sim_backend="motrix")
class LeapInhandRotationEnv(LeapHandBaseEnv):
    """以 16 個 LEAP position targets 控制四指旋轉球體。"""

    _cfg: LeapInhandRotationCfg
    _FINGERTIP_BODY_NAMES = (
        "fingertip",
        "fingertip_2",
        "fingertip_3",
        "thumb_fingertip",
    )
    _PALM_CONTACT_SENSOR = "leap_rotation_palm_contact"
    _CONTACT_SENSORS = (
        "leap_ff_contact",
        "leap_mf_contact",
        "leap_rf_contact",
        "leap_th_contact",
    )
    _reward_cfg: RewardConfigPPO

    # 每幀 = 16 joint positions + 16 targets + 3 ball position。
    _NUM_OBS_PER_STEP = 60
    _INCLUDE_FINGERTIP_CONTACT_OBS = True
    _INCLUDE_BALL_ANGVEL_OBS = True
    _INCLUDE_ROTATION_PHASE_OBS = True
    _INCLUDE_FINGERTIP_REL_OBS = True
    _INCLUDE_CONTACT_DURATION_OBS = True
    # 疊三幀，讓 policy 能由歷史差分推斷運動趨勢。
    _NUM_LAG_STEPS = 3

    def __init__(
        self, cfg: LeapInhandRotationCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ) -> None:
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_body_name,
            push_body_name=cfg.domain_rand.push_body_name,
            add_body_sensors=True,
            position_actuator_gains={
                "kp": cfg.control_config.kp,
                "kd": cfg.control_config.kd,
                "actuator_ids": slice(0, 16),
            },
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._rotation_window_steps = max(
            1,
            int(
                round(
                    float(getattr(self._reward_cfg, "rotation_window_seconds", 1.0))
                    / float(self._cfg.ctrl_dt)
                )
            ),
        )

        self._dof_range = self._ctrl_upper - self._ctrl_lower
        self._dof_mid = (self._ctrl_upper + self._ctrl_lower) / 2.0
        self._rot_axis = normalize_rotation_axis(cfg.rotation_axis)
        self._grasp_cache: np.ndarray | None = None
        self._grasp_cache_loaded = False
        self._grasp_cache_cursor = int(cfg.grasp_cache_start_index)

        self._init_reward_functions()
        self._init_domain_randomization(self._domain_randomization_provider())

    def _domain_randomization_provider(self) -> DomainRandomizationProvider:
        return LeapRotationDomainRandomizationProvider()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._NUM_OBS_PER_STEP * self._NUM_LAG_STEPS}

    def _init_reward_functions(self) -> None:
        """Register the reward terms selected by the owner YAML."""
        self._reward_fns = {
            "rotate": self._reward_rotate,
            "reverse_rotate": self._reward_reverse_rotate,
            "rotation_streak": self._reward_rotation_streak,
            "stall": self._reward_stall,
            "contact_rotation": self._reward_contact_rotation,
            "window_rotation": self._reward_window_rotation,
            "palm_contact": self._reward_palm_contact,
            "obj_linvel": self._reward_obj_linvel,
            "pose_diff": self._reward_pose_diff,
            "torque": self._reward_torque,
            "work": self._reward_work,
            "drop": self._reward_drop,
        }

    def _rotation_reward_active(self, info: dict[str, Any], values: np.ndarray) -> np.ndarray:
        warmup_steps = int(
            round(
                float(getattr(self._reward_cfg, "rotation_warmup_seconds", 0.0))
                / float(self._cfg.ctrl_dt)
            )
        )
        if warmup_steps <= 0:
            return values
        step_count = np.asarray(info.get("steps", np.zeros(values.shape[0], dtype=np.uint32)))
        return np.where(step_count >= warmup_steps, values, 0.0)

    def _reward_rotate(
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
        del dof_pos, dof_vel, ball_pos, ball_linvel, torques, terminated
        signed_angvel = ball_angvel @ self._rot_axis
        reward = np.clip(signed_angvel, 0.0, self._reward_cfg.angvel_clip_max)
        return self._rotation_reward_active(info, reward)

    def _reward_reverse_rotate(
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
        """Return only angular speed opposite to the configured rotation axis."""
        del dof_pos, dof_vel, ball_pos, ball_linvel, torques, terminated
        penalty = np.clip(
            -(ball_angvel @ self._rot_axis),
            0.0,
            float(
                getattr(
                    self._reward_cfg,
                    "reverse_angvel_clip_max",
                    max(0.0, -float(self._reward_cfg.angvel_clip_min)),
                )
            ),
        )
        return np.asarray(self._rotation_reward_active(info, penalty), dtype=get_global_dtype())

    def _reward_rotation_streak(
        self, info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
    ) -> np.ndarray:
        del dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        seconds = np.asarray(info["rotation_streak_steps"]) * float(self._cfg.ctrl_dt)
        reward = np.clip(
            seconds / float(self._reward_cfg.rotation_streak_target_seconds),
            0.0,
            1.0,
        )
        return self._rotation_reward_active(info, reward)

    def _reward_stall(
        self, info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
    ) -> np.ndarray:
        del dof_pos, dof_vel, ball_pos, ball_linvel, torques
        signed_angvel = ball_angvel @ self._rot_axis
        grace_steps = int(
            round(float(self._reward_cfg.stall_grace_seconds) / float(self._cfg.ctrl_dt))
        )
        steps = np.asarray(info.get("steps", np.zeros(signed_angvel.shape[0])))
        stalled = (steps >= grace_steps) & (
            signed_angvel < float(getattr(self._reward_cfg, "minimum_positive_angvel", 0.15))
        )
        return np.asarray(stalled & ~terminated, dtype=get_global_dtype())

    def _reward_contact_rotation(
        self, info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
    ) -> np.ndarray:
        del dof_pos, dof_vel, ball_pos, ball_linvel, torques, terminated
        signed_angvel = np.clip(ball_angvel @ self._rot_axis, 0.0, self._reward_cfg.angvel_clip_max)
        contacts = np.asarray(info["curr_fingertip_contacts"])
        contact_gate = np.asarray(
            np.sum(contacts, axis=1)
            >= float(self._reward_cfg.contact_rotation_min_contacts),
            dtype=get_global_dtype(),
        )
        return self._rotation_reward_active(info, signed_angvel * contact_gate)

    def _reward_window_rotation(
        self, info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
    ) -> np.ndarray:
        del dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        average_angvel = np.asarray(info["rotation_window_sum"]) / float(
            self._reward_cfg.rotation_window_seconds
        )
        reward = np.clip(average_angvel, 0.0, self._reward_cfg.angvel_clip_max)
        return self._rotation_reward_active(info, reward)

    @staticmethod
    def _contact_flag(sensor_data: np.ndarray) -> np.ndarray:
        values = np.asarray(sensor_data)
        return np.asarray(
            values.reshape(values.shape[0], -1)[:, 0] > 0.5,
            dtype=get_global_dtype(),
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
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        return self._contact_flag(self.get_sensor_data(self._PALM_CONTACT_SENSOR))

    def _reward_obj_linvel(
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
        del info, dof_pos, dof_vel, ball_pos, ball_angvel, torques, terminated
        penalty: np.ndarray = np.sum(np.abs(ball_linvel), axis=1)
        return penalty

    def _reward_pose_diff(
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
        del dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        diff = dof_pos - info["init_pose"]
        penalty: np.ndarray = np.sum(np.square(diff), axis=1)
        return penalty

    def _reward_torque(
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
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, terminated
        penalty: np.ndarray = np.sum(np.square(torques), axis=1)
        return penalty

    def _reward_work(
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
        del info, dof_pos, ball_pos, ball_linvel, ball_angvel, terminated
        work = np.sum(torques * dof_vel, axis=1)
        penalty: np.ndarray = np.square(work)
        return penalty

    def _reward_drop(
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
        del info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques
        return np.asarray(terminated, dtype=get_global_dtype())

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """讀取物理狀態、估算速度、計算 reward/termination/observation。"""
        dof_pos = self.get_hand_dof_pos()
        ball_pos = self.get_ball_pos()
        ball_quat = self.get_ball_quat()

        dof_vel = (dof_pos - state.info.get("prev_dof_pos", dof_pos)) / self._cfg.ctrl_dt
        ball_linvel = (ball_pos - state.info.get("prev_ball_pos", ball_pos)) / self._cfg.ctrl_dt

        prev_ball_quat = state.info.get("prev_ball_quat", ball_quat)
        ball_angvel = compute_ball_angvel(ball_quat, prev_ball_quat, self._cfg.ctrl_dt)

        state.info["curr_dof_pos"] = dof_pos.copy()
        state.info["curr_ball_pos"] = ball_pos.copy()
        state.info["curr_ball_quat"] = ball_quat.copy()
        state.info["curr_ball_linvel"] = ball_linvel.copy()
        state.info["curr_ball_angvel"] = ball_angvel.copy()
        contacts = np.stack(
            [self._contact_flag(self.get_sensor_data(name)) for name in self._CONTACT_SENSORS],
            axis=1,
        ).astype(get_global_dtype())
        state.info["curr_fingertip_contacts"] = contacts
        state.info["curr_fingertip_rel"] = np.asarray(
            self.get_fingertip_pos() - ball_pos[:, None, :],
            dtype=get_global_dtype(),
        )
        previous_contact_steps = np.asarray(
            state.info.get("contact_duration_steps", np.zeros_like(contacts))
        )
        state.info["contact_duration_steps"] = np.where(
            contacts > 0.5, previous_contact_steps + 1.0, 0.0
        ).astype(get_global_dtype())

        signed_angvel = np.asarray(ball_angvel @ self._rot_axis)
        previous_streak = np.asarray(
            state.info.get(
                "rotation_streak_steps",
                np.zeros(self._num_envs, dtype=np.uint32),
            )
        )
        state.info["rotation_streak_steps"] = np.where(
            signed_angvel >= float(getattr(self._reward_cfg, "minimum_positive_angvel", 0.15)),
            previous_streak + 1,
            0,
        ).astype(np.uint32)

        rotation_window = np.asarray(
            state.info.get(
                "rotation_window",
                np.zeros(
                    (self._num_envs, self._rotation_window_steps),
                    dtype=get_global_dtype(),
                ),
            )
        )
        rotation_window_sum = np.asarray(
            state.info.get(
                "rotation_window_sum",
                np.zeros(self._num_envs, dtype=get_global_dtype()),
            )
        )
        steps = np.asarray(state.info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))
        cursor = np.mod(steps, self._rotation_window_steps).astype(np.int64)
        env_indices = np.arange(self._num_envs)
        old_angle = rotation_window[env_indices, cursor].copy()
        new_angle = signed_angvel * float(self._cfg.ctrl_dt)
        rotation_window[env_indices, cursor] = new_angle
        state.info["rotation_window"] = rotation_window
        state.info["rotation_window_sum"] = np.asarray(
            rotation_window_sum - old_angle + new_angle,
            dtype=get_global_dtype(),
        )
        state.info["prev_dof_pos"] = dof_pos.copy()
        state.info["prev_ball_pos"] = ball_pos.copy()
        state.info["prev_ball_quat"] = ball_quat.copy()

        targets = state.info["prev_ctrl"]
        torques = compute_pd_torques(
            targets=targets,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            kp=self._cfg.control_config.kp,
            kd=self._cfg.control_config.kd,
        )
        terminated = ball_pos[:, 2] < self._reward_cfg.reset_z_threshold

        reward = self._compute_reward(
            state.info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
        )
        obs = self._compute_obs(state.info, dof_pos, ball_pos)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

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
        # 所有權重由 backend owner YAML 注入；Python 只負責 term 定義。
        dtype = get_global_dtype()
        reward = np.zeros(self._num_envs, dtype=dtype)
        step_count = info.get("steps", np.zeros(self._num_envs, dtype=np.uint32))
        should_log = self._enable_reward_log and (int(step_count[0]) % 4 == 0)
        log = {} if should_log else info.get("log", {})

        for name, scale in self._reward_cfg.scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](
                info, dof_pos, dof_vel, ball_pos, ball_linvel, ball_angvel, torques, terminated
            )
            weighted_rew = rew * scale
            reward += weighted_rew
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted_rew))

        if should_log:
            log["reward/total"] = float(np.mean(reward))

        info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _build_current_obs(
        self, info: dict[str, Any], dof_pos: np.ndarray, ball_pos: np.ndarray
    ) -> np.ndarray:
        # observation 不含 backend 私有資料，因此 MuJoCo/Motrix policy contract 一致。
        dtype = get_global_dtype()
        targets = info["prev_ctrl"]
        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1e-8)

        noise_cfg = self._cfg.noise_config
        if noise_cfg.level > 0.0:
            dof_pos_norm += (
                np.random.uniform(-1.0, 1.0, dof_pos_norm.shape).astype(dtype)
                * noise_cfg.level
                * noise_cfg.scale_joint_angle
            )

        obs_parts = [dof_pos_norm, targets, ball_pos.astype(dtype)]
        if self._INCLUDE_FINGERTIP_CONTACT_OBS:
            obs_parts.append(
                np.asarray(
                    info.get(
                        "curr_fingertip_contacts",
                        np.zeros((dof_pos.shape[0], len(self._CONTACT_SENSORS))),
                    ),
                    dtype=dtype,
                )
            )
        if self._INCLUDE_BALL_ANGVEL_OBS:
            obs_parts.append(
                np.asarray(
                    info.get("curr_ball_angvel", np.zeros_like(ball_pos)),
                    dtype=dtype,
                )
            )
        if self._INCLUDE_ROTATION_PHASE_OBS:
            cycle_seconds = float(self._cfg.rotation_cycle_seconds)
            if cycle_seconds <= 0.0:
                raise ValueError("rotation_cycle_seconds must be positive")
            step_count = np.asarray(
                info.get("steps", np.zeros(dof_pos.shape[0], dtype=np.float32)),
                dtype=dtype,
            )
            phase = 2.0 * np.pi * step_count * float(self._cfg.ctrl_dt) / cycle_seconds
            obs_parts.append(np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(dtype))
        if self._INCLUDE_FINGERTIP_REL_OBS:
            obs_parts.append(
                np.asarray(
                    info.get(
                        "curr_fingertip_rel",
                        np.zeros((dof_pos.shape[0], len(self._CONTACT_SENSORS), 3)),
                    ),
                    dtype=dtype,
                ).reshape(dof_pos.shape[0], -1)
            )
        if self._INCLUDE_CONTACT_DURATION_OBS:
            contact_duration = np.asarray(
                info.get(
                    "contact_duration_steps",
                    np.zeros((dof_pos.shape[0], len(self._CONTACT_SENSORS))),
                ),
                dtype=dtype,
            )
            obs_parts.append(np.clip(contact_duration * float(self._cfg.ctrl_dt), 0.0, 1.0))
        return np.concatenate(obs_parts, axis=1, dtype=dtype)

    def _compute_obs(
        self, info: dict[str, Any], dof_pos: np.ndarray, ball_pos: np.ndarray
    ) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        current_obs = self._build_current_obs(info, dof_pos, ball_pos)

        num_envs = dof_pos.shape[0]
        obs_lag_history = info.get(
            "obs_lag_history",
            np.zeros(
                (num_envs, self._NUM_LAG_STEPS, self._NUM_OBS_PER_STEP),
                dtype=dtype,
            ),
        )
        obs_lag_history[:, :-1] = obs_lag_history[:, 1:]
        obs_lag_history[:, -1] = current_obs
        info["obs_lag_history"] = obs_lag_history

        return {
            "obs": np.asarray(obs_lag_history.reshape(num_envs, -1), dtype=dtype),
        }


RewardConfig = RewardConfigPPO
Domain_Rand = DomainRandConfig
LeapInhandRotation = LeapInhandRotationEnv
