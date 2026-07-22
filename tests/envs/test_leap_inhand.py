from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir


def test_leap_ball_catch_wrap_requires_contacts_around_ball_center() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _planar_enclosure

    around = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=np.float32,
    )
    same_side = np.array(
        [[[1.0, -0.3, 0.0], [1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.3, 0.0]]],
        dtype=np.float32,
    )
    all_contacts = np.ones((1, 4), dtype=np.float32)
    missing_contact = all_contacts.copy()
    missing_contact[:, 2] = 0.0

    wrapped, quality = _planar_enclosure(around, all_contacts)
    same_side_wrapped, same_side_quality = _planar_enclosure(same_side, all_contacts)
    missing_wrapped, missing_quality = _planar_enclosure(around, missing_contact)

    assert wrapped[0] == 1.0
    assert quality[0] == pytest.approx(1.0)
    assert same_side_wrapped[0] == 0.0
    assert same_side_quality[0] == 0.0
    assert missing_wrapped[0] == 0.0
    assert missing_quality[0] == 0.0


def test_leap_ball_catch_opposition_geometry_rejects_crossed_thumb() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _opposition_geometry

    natural_cup = np.array(
        [[[0.06, 0.04, 0.0], [0.06, 0.0, 0.0], [0.06, -0.04, 0.0], [-0.06, 0.0, 0.0]]],
        dtype=np.float32,
    )
    crossed_thumb = natural_cup.copy()
    crossed_thumb[:, 3, 0] = 0.02
    inactive_ring = natural_cup.copy()
    inactive_ring[:, 2, 1] = 0.02

    quality, ring_side, penalty = _opposition_geometry(natural_cup)
    crossed_quality, _, crossed_penalty = _opposition_geometry(crossed_thumb)
    inactive_quality, inactive_ring_side, _ = _opposition_geometry(inactive_ring)

    assert quality[0] == pytest.approx(1.0)
    assert ring_side[0] == pytest.approx(1.0)
    assert penalty[0] == pytest.approx(0.0)
    assert crossed_quality[0] == 0.0
    assert crossed_penalty[0] > 0.0
    assert inactive_quality[0] == 0.0
    assert inactive_ring_side[0] == 0.0


def test_leap_ball_catch_soft_wrap_gives_gradient_before_all_contacts() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _soft_planar_enclosure

    around = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=np.float32,
    )
    same_side = np.array(
        [[[1.0, -0.3, 0.0], [1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.3, 0.0]]],
        dtype=np.float32,
    )
    partial_contacts = np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    near_fingers = np.array([[0.9, 0.8, 0.55, 0.75]], dtype=np.float32)

    soft = _soft_planar_enclosure(around, partial_contacts, near_fingers)
    poor = _soft_planar_enclosure(same_side, partial_contacts, near_fingers)

    assert soft[0] > 0.0
    assert soft[0] > poor[0]


def test_leap_ball_catch_high_value_pad_contact_requires_all_four_pads() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _four_pad_contact_quality

    contacts = np.ones((1, 4), dtype=np.float32)
    proximity = np.ones((1, 4), dtype=np.float32)
    alignment = np.ones((1, 4), dtype=np.float32)
    posture = np.ones((1, 4), dtype=np.float32)

    assert _four_pad_contact_quality(contacts, proximity, alignment, posture)[0] == pytest.approx(
        1.0
    )

    missing_thumb = contacts.copy()
    missing_thumb[:, 3] = 0.0
    assert _four_pad_contact_quality(missing_thumb, proximity, alignment, posture)[0] == 0.0

    wrong_thumb_surface = alignment.copy()
    wrong_thumb_surface[:, 3] = 0.0
    assert _four_pad_contact_quality(contacts, proximity, wrong_thumb_surface, posture)[0] < 0.35

    reversed_thumb = posture.copy()
    reversed_thumb[:, 3] = 0.05
    assert _four_pad_contact_quality(contacts, proximity, alignment, reversed_thumb)[0] < 0.1


def test_leap_ball_catch_posture_envelope_rejects_telemetry_exploit() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _finger_posture_quality

    natural = np.array(
        [
            [
                0.62,
                -0.14,
                0.78,
                0.38,
                0.74,
                0.0,
                0.88,
                0.46,
                0.84,
                0.06,
                1.04,
                0.58,
                0.72,
                0.48,
                0.62,
                0.32,
            ]
        ],
        dtype=np.float32,
    )
    exploit = np.array(
        [
            [
                -0.025,
                0.813,
                1.656,
                0.153,
                0.319,
                -0.014,
                1.279,
                0.127,
                1.283,
                -0.317,
                0.792,
                -0.334,
                1.823,
                -0.322,
                1.656,
                -0.94,
            ]
        ],
        dtype=np.float32,
    )

    natural_quality, natural_penalty = _finger_posture_quality(natural)
    exploit_quality, exploit_penalty = _finger_posture_quality(exploit)

    assert np.min(natural_quality) > 0.99
    assert exploit_quality[0, 0] < 0.4
    assert exploit_quality[0, 2] < 0.4
    assert exploit_quality[0, 3] < 0.01
    assert exploit_penalty[0] > natural_penalty[0] + 1.0


def test_leap_ball_catch_surface_layout_prefers_fanned_opposed_pads() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _grasp_surface_layout_quality

    natural = np.array(
        [[[0.78, 0.50, -0.35], [0.90, 0.0, -0.43], [0.78, -0.50, -0.35], [-0.62, 0.0, -0.78]]],
        dtype=np.float32,
    )
    lined_up = np.repeat(natural[:, 1:2, :], 4, axis=1)

    natural_quality = _grasp_surface_layout_quality(natural)
    lined_up_quality = _grasp_surface_layout_quality(lined_up)

    assert np.min(natural_quality) > 0.99
    assert np.mean(natural_quality) > np.mean(lined_up_quality)


def test_leap_ball_catch_staged_wrap_progress_is_monotonic() -> None:
    from unilab.envs.manipulation.leap_inhand.catch import _staged_wrap_progress

    readiness = np.full((5, 4), 0.5, dtype=np.float32)
    enclosure = np.zeros(5, dtype=np.float32)
    contacts = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    progress = _staged_wrap_progress(contacts, readiness, enclosure)

    assert np.all(np.diff(progress) > 0.0)


def test_leap_scene_preserves_fixed_hand_object_layout() -> None:
    mujoco = pytest.importorskip("mujoco")
    asset_dir = (
        Path(__file__).resolve().parents[2] / "src" / "unilab" / "assets" / "robots" / "leap_hand"
    )
    from unilab.base.backend.mujoco.xml import materialize_scene_fragments

    materialized = Path(
        materialize_scene_fragments(
            str(asset_dir / "leap_hand.xml"),
            fragment_files=[str(asset_dir / "scene.xml")],
        )
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(materialized))
    finally:
        materialized.unlink(missing_ok=True)

    assert (model.nq, model.nv, model.nu) == (23, 22, 16)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower")
    assert palm_id >= 0
    assert model.body_pos[palm_id].tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert model.body_quat[palm_id].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base") == -1
    actuator_qpos_addresses = [
        int(model.jnt_qposadr[int(model.actuator_trnid[actuator_id, 0])])
        for actuator_id in range(model.nu)
    ]
    assert actuator_qpos_addresses == list(range(16))
    expected_hand_qpos = [
        0.675558466,
        -0.311245401,
        1.28696011,
        0.481546183,
        1.1454015,
        0.0429715889,
        0.0624296905,
        0.440274799,
        1.60076238,
        -0.184116801,
        0.00624721579,
        0.743162204,
        1.40099437,
        1.32970657,
        0.528728303,
        0.312199124,
    ]
    assert model.key_qpos[0, :16].tolist() == pytest.approx(expected_hand_qpos)
    assert model.key_qpos[0, 16:19].tolist() == pytest.approx(
        [-0.0225964729, 0.021169898, 0.639558165]
    )
    assert model.key_qpos[0, 19:23].tolist() == pytest.approx(
        [0.999573468, -0.0156558521, -0.0229829881, 0.00891954799]
    )
    home_qpos = model.key_qpos[0, :16]
    lower_clearance = home_qpos - model.actuator_ctrlrange[:, 0]
    upper_clearance = model.actuator_ctrlrange[:, 1] - home_qpos
    assert np.min(np.minimum(lower_clearance, upper_clearance)) > 0.025
    # Ring joints are qpos 8:12; none may be parked at its lower limit.
    assert np.all(lower_clearance[8:12] > 0.08)
    scene_source = (asset_dir / "scene.xml").read_text(encoding="utf-8")
    assert "ball_orientation_marker" not in scene_source
    assert "rotation_ground_tex" in scene_source
    assert 'type="sphere" size="0.0335"' in scene_source
    assert "rotation_ball_tex" in scene_source
    assert 'builtin="checker"' in scene_source
    assert "leap_th_contact" in scene_source
    assert "leap_rotation_palm_contact" in scene_source


def test_leap_grasp_cache_contract_rejects_invalid_arrays() -> None:
    from unilab.envs.manipulation.leap_inhand.rotation import validate_grasp_cache

    valid = validate_grasp_cache(np.zeros((2, 23), dtype=np.float32), "test-cache")
    assert valid.shape == (2, 23)
    assert valid.dtype == np.float64

    with pytest.raises(ValueError, match=r"expected shape \(N, 23\)"):
        validate_grasp_cache(np.zeros((2, 22), dtype=np.float32), "bad-shape")
    with pytest.raises(ValueError, match="empty"):
        validate_grasp_cache(np.zeros((0, 23), dtype=np.float32), "empty")
    non_finite = np.zeros((1, 23), dtype=np.float32)
    non_finite[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_grasp_cache(non_finite, "nan-cache")


def test_leap_grasp_warmup_is_half_a_second_at_control_rate() -> None:
    from unilab.envs.manipulation.leap_inhand.grasp_gen import grasp_warmup_steps

    assert grasp_warmup_steps(0.5, 0.02) == 25
    assert grasp_warmup_steps(0.001, 0.02) == 1


def test_leap_task_owns_robot_specific_names() -> None:
    from unilab.envs.manipulation.leap_inhand.rotation import (
        LeapInhandRotationCfg,
        LeapInhandRotationEnv,
    )

    cfg = LeapInhandRotationCfg()

    assert cfg.base_body_name == "palm_lower"
    assert Path(cfg.scene.model_file).parts[-3:] == ("robots", "leap_hand", "leap_hand.xml")
    assert tuple(Path(path).name for path in cfg.scene.fragment_files) == ("scene.xml",)
    assert LeapInhandRotationEnv._FINGERTIP_BODY_NAMES == (
        "fingertip",
        "fingertip_2",
        "fingertip_3",
        "thumb_fingertip",
    )
    assert LeapInhandRotationEnv._NUM_OBS_PER_STEP == 60


def test_leap_default_offset_actions_stay_centered_on_reset_grasp() -> None:
    from types import SimpleNamespace

    from unilab.envs.manipulation.leap_inhand.base import ControlConfig, LeapHandBaseEnv

    class TestLeapEnv(LeapHandBaseEnv):
        def update_state(self, state):  # type: ignore[no-untyped-def]
            return state

    env = object.__new__(TestLeapEnv)
    env._np_dtype = np.dtype(np.float32)
    env._num_action = 2
    env.default_angles = np.array([0.5, -0.5], dtype=np.float32)
    env._ctrl_lower = np.array([-1.0, -1.0], dtype=np.float32)
    env._ctrl_upper = np.array([1.0, 1.0], dtype=np.float32)
    env._cfg = SimpleNamespace(
        control_config=ControlConfig(action_scale=0.25, target_mode="default_offset")
    )
    state = SimpleNamespace(
        info={
            "init_pose": np.array([[0.5, -0.5]], dtype=np.float32),
            "prev_ctrl": np.array([[0.9, 0.9]], dtype=np.float32),
        }
    )

    moved = env.apply_action(np.array([[1.0, -1.0]], dtype=np.float32), state)
    centered = env.apply_action(np.zeros((1, 2), dtype=np.float32), state)

    np.testing.assert_allclose(moved, [[0.75, -0.75]])
    np.testing.assert_allclose(centered, [[0.5, -0.5]])


def test_leap_bounded_incremental_actions_accumulate_without_leaving_grasp_window() -> None:
    from types import SimpleNamespace

    from unilab.envs.manipulation.leap_inhand.base import ControlConfig, LeapHandBaseEnv

    class TestLeapEnv(LeapHandBaseEnv):
        def update_state(self, state):  # type: ignore[no-untyped-def]
            return state

    env = object.__new__(TestLeapEnv)
    env._np_dtype = np.dtype(np.float32)
    env._num_action = 2
    env.default_angles = np.array([0.5, -0.5], dtype=np.float32)
    env._ctrl_lower = np.array([-1.0, -1.0], dtype=np.float32)
    env._ctrl_upper = np.array([1.0, 1.0], dtype=np.float32)
    env._cfg = SimpleNamespace(
        control_config=ControlConfig(
            action_scale=0.1,
            target_mode="bounded_incremental",
            target_offset_limit=0.2,
        )
    )
    state = SimpleNamespace(
        info={
            "init_pose": np.array([[0.5, -0.5]], dtype=np.float32),
            "prev_ctrl": np.array([[0.65, -0.65]], dtype=np.float32),
        }
    )

    first = env.apply_action(np.array([[1.0, -1.0]], dtype=np.float32), state)
    second = env.apply_action(np.array([[1.0, -1.0]], dtype=np.float32), state)

    np.testing.assert_allclose(first, [[0.7, -0.7]])
    np.testing.assert_allclose(second, [[0.7, -0.7]])


def test_leap_rotation_reward_ignores_reset_contact_relaxation() -> None:
    from types import SimpleNamespace

    from unilab.envs.manipulation.leap_inhand.rotation import LeapInhandRotationEnv

    env = object.__new__(LeapInhandRotationEnv)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._reward_cfg = SimpleNamespace(
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        rotation_warmup_seconds=1.0,
    )
    zeros3 = np.zeros((2, 3), dtype=np.float32)
    zeros16 = np.zeros((2, 16), dtype=np.float32)
    reward = env._reward_rotate(
        {"steps": np.array([49, 50], dtype=np.uint32)},
        zeros16,
        zeros16,
        zeros3,
        zeros3,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        zeros16,
        np.zeros(2, dtype=bool),
    )

    np.testing.assert_allclose(reward, [0.0, 0.5])


def test_leap_reverse_rotation_reward_penalizes_only_wrong_direction() -> None:
    from types import SimpleNamespace

    from unilab.envs.manipulation.leap_inhand.rotation import LeapInhandRotationEnv

    env = object.__new__(LeapInhandRotationEnv)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._reward_cfg = SimpleNamespace(
        angvel_clip_min=-0.5,
        rotation_warmup_seconds=1.0,
    )
    zeros3 = np.zeros((2, 3), dtype=np.float32)
    zeros16 = np.zeros((2, 16), dtype=np.float32)
    reward = env._reward_reverse_rotate(
        {"steps": np.array([50, 50], dtype=np.uint32)},
        zeros16,
        zeros16,
        zeros3,
        zeros3,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        zeros16,
        np.zeros(2, dtype=bool),
    )

    np.testing.assert_allclose(reward, [0.0, 0.5])


def test_leap_sustained_rotation_terms_reward_contacted_progress() -> None:
    from types import SimpleNamespace

    from unilab.envs.manipulation.leap_inhand.rotation import LeapInhandRotationEnv

    env = object.__new__(LeapInhandRotationEnv)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._reward_cfg = SimpleNamespace(
        angvel_clip_max=1.5,
        rotation_warmup_seconds=0.0,
        minimum_positive_angvel=0.15,
        rotation_streak_target_seconds=2.0,
        rotation_window_seconds=1.0,
        stall_grace_seconds=1.0,
        contact_rotation_min_contacts=2,
    )
    zeros3 = np.zeros((2, 3), dtype=np.float32)
    zeros16 = np.zeros((2, 16), dtype=np.float32)
    angvel = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    info = {
        "steps": np.array([100, 100], dtype=np.uint32),
        "rotation_streak_steps": np.array([50, 0], dtype=np.uint32),
        "rotation_window_sum": np.array([0.8, -0.2], dtype=np.float32),
        "curr_fingertip_contacts": np.array(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=np.float32
        ),
    }
    terminated = np.zeros(2, dtype=bool)

    rotate = env._reward_rotate(info, zeros16, zeros16, zeros3, zeros3, angvel, zeros16, terminated)
    contact = env._reward_contact_rotation(
        info, zeros16, zeros16, zeros3, zeros3, angvel, zeros16, terminated
    )
    streak = env._reward_rotation_streak(
        info, zeros16, zeros16, zeros3, zeros3, angvel, zeros16, terminated
    )
    window = env._reward_window_rotation(
        info, zeros16, zeros16, zeros3, zeros3, angvel, zeros16, terminated
    )
    stall = env._reward_stall(info, zeros16, zeros16, zeros3, zeros3, angvel, zeros16, terminated)

    np.testing.assert_allclose(rotate, [1.0, 0.0])
    np.testing.assert_allclose(contact, [1.0, 0.0])
    np.testing.assert_allclose(streak, [0.5, 0.0])
    np.testing.assert_allclose(window, [0.8, 0.0])
    np.testing.assert_allclose(stall, [0.0, 1.0])


def test_leap_appo_owner_config_composes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "appo"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=leap_inhand/mujoco"])

    assert cfg.training.task_name == "LeapInhandRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.algo == "appo"
    assert cfg.algo.steps_per_env == 8
    assert cfg.algo.algorithm.learning_rate == 0.001
    assert cfg.algo.algorithm.desired_kl == 0.02
    assert cfg.algo.algorithm.entropy_coef == 0.002
    assert cfg.algo.actor.distribution_cfg.init_std == 0.5
    assert cfg.env.rotation_axis == [0.0, 0.0, 1.0]
    assert cfg.env.rotation_cycle_seconds == pytest.approx(2.0)
    assert cfg.env.gen_grasp is False
    assert cfg.env.grasp_cache_path == "caches/leap_hand_allegro_style_20k.npy"
    assert cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)
    assert cfg.env.control_config.target_mode == "incremental"
    assert cfg.env.control_config.kp == 20.0
    assert cfg.env.control_config.kd == 1.5
    assert cfg.reward.reset_z_threshold == pytest.approx(0.54)
    assert cfg.reward.rotation_warmup_seconds == pytest.approx(0.0)
    assert dict(cfg.reward.scales) == {
        "rotate": 3.0,
        "reverse_rotate": -8.0,
        "rotation_streak": 1.0,
        "stall": -0.5,
        "contact_rotation": 2.0,
        "window_rotation": 2.0,
        "palm_contact": -0.5,
        "obj_linvel": -0.3,
        "pose_diff": 0.0,
        "torque": -0.001,
        "work": -0.0001,
        "drop": -10.0,
    }
    assert cfg.training.checkpoint_video.enabled is True
    assert cfg.training.play_env_num == 1
    assert cfg.training.render_width == 640
    assert cfg.training.render_height == 352
    assert cfg.training.render_num_processes == 1
    assert cfg.training.checkpoint_video.play_env_num == 1
    assert cfg.training.checkpoint_video.play_steps == 200


def test_leap_appo_motrix_owner_preserves_cross_backend_contract() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "appo"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=leap_inhand/motrix"])

    assert cfg.training.task_name == "LeapInhandRotation"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.actor.hidden_dims == [512, 256, 128]
    assert cfg.algo.algorithm.entropy_coef == 0.002
    assert cfg.algo.actor.distribution_cfg.init_std == 0.5
    assert cfg.env.control_config.action_scale == pytest.approx(1.0 / 24.0)
    assert cfg.env.control_config.target_mode == "incremental"
    assert cfg.env.use_grasp_cache is True
    assert cfg.env.rotation_axis == [0.0, 0.0, 1.0]
    assert cfg.env.rotation_cycle_seconds == pytest.approx(2.0)
    assert cfg.reward.reset_z_threshold == pytest.approx(0.54)
    assert cfg.reward.rotation_warmup_seconds == pytest.approx(0.0)
    owner_text = (config_dir / "task" / "leap_inhand" / "motrix.yaml").read_text(encoding="utf-8")
    assert "defaults:" not in owner_text
    assert "/task/leap_inhand/mujoco" not in owner_text
    assert dict(cfg.reward.scales) == {
        "rotate": 3.0,
        "reverse_rotate": -8.0,
        "rotation_streak": 1.0,
        "stall": -0.5,
        "contact_rotation": 2.0,
        "window_rotation": 2.0,
        "palm_contact": -0.5,
        "obj_linvel": -0.3,
        "pose_diff": 0.0,
        "torque": -0.001,
        "work": -0.0001,
        "drop": -10.0,
    }


def test_leap_grasp_motrix_owner_is_standalone_and_targets_20k() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "ppo"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=leap_inhand_grasp/motrix"])

    assert cfg.training.task_name == "LeapInhandRotationGrasp"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.training.no_play is True
    assert cfg.env.gen_grasp is True
    assert cfg.env.grasp_collection_target == 20_000
    assert cfg.env.grasp_cache_path == "caches/leap_hand_allegro_style_20k.npy"
    assert cfg.env.grasp_quality_check is True
    assert cfg.reward.reset_z_threshold == pytest.approx(0.54)

    assert cfg.env.grasp_min_contacts == 2
    assert cfg.env.grasp_contact_mask_quota == 2500
    assert cfg.env.grasp_warmup_seconds == pytest.approx(0.5)
    assert cfg.env.grasp_pad_target_distances == [0.036, 0.039, 0.050, 0.042]
    assert cfg.env.grasp_pad_surface_tolerance == pytest.approx(0.010)
    assert cfg.env.grasp_pad_alignment_minimums == [0.20, 0.20, 0.00, 0.20]
    assert cfg.env.grasp_opposition_dot_max == pytest.approx(0.0)
    assert cfg.env.domain_rand.joint_noise == pytest.approx(0.03)
    owner_text = (config_dir / "task" / "leap_inhand_grasp" / "motrix.yaml").read_text(
        encoding="utf-8"
    )
    assert "defaults:" not in owner_text
    assert "/task/leap_inhand/mujoco" not in owner_text

    from unilab.envs.manipulation.leap_inhand.grasp_gen import (
        LeapInhandRotationGrasp,
    )

    assert LeapInhandRotationGrasp._CONTACT_SENSORS == (
        "leap_ff_contact",
        "leap_mf_contact",
        "leap_rf_contact",
        "leap_th_contact",
    )


def test_leap_ball_catch_motrix_owner_composes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "conf" / "appo"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=leap_ball_catch/motrix"])

    assert cfg.training.task_name == "LeapBallCatch"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.training.play_env_num == 16
    assert cfg.algo.save_interval == 0
    assert cfg.training.checkpoint_video.enabled is False
    assert cfg.algo.algorithm.entropy_coef == 0.003
    assert cfg.algo.actor.distribution_cfg.init_std == 0.45
    assert "rotate" not in cfg.reward.scales
    assert "obj_linvel" not in cfg.reward.scales
    assert "pose_diff" not in cfg.reward.scales
    assert "work" not in cfg.reward.scales
    assert set(cfg.reward.scales) == {
        "reference_pose_tracking",
        "early_closure",
        "finger_pad_progress",
        "grasp_surface_layout",
        "finger_closing",
        "pocket_depth",
        "palm_assist",
        "pocket_without_finger_contact",
        "high_ball",
        "wrap_progress",
        "finger_wrap",
        "hold",
        "finger_gap",
        "finger_synergy",
        "bad_pad_contact",
        "finger_jitter",
        "palm_only",
        "ball_bounce",
        "action_rate",
        "settled_action",
        "torque",
        "drop",
    }
    assert cfg.reward.scales.reference_pose_tracking == 8.0
    assert cfg.reward.scales.early_closure == -14.0
    assert cfg.reward.scales.finger_pad_progress == 14.0
    assert cfg.reward.scales.grasp_surface_layout == 8.0
    assert cfg.reward.scales.finger_closing == 4.0
    assert cfg.reward.scales.pocket_depth == 12.0
    assert cfg.reward.scales.high_ball == -12.0
    assert cfg.reward.scales.wrap_progress == 12.0
    assert cfg.reward.scales.finger_wrap == 34.0
    assert cfg.reward.scales.hold == 18.0
    assert cfg.reward.scales.finger_gap == -12.0
    assert cfg.reward.scales.finger_synergy == -10.0
    assert cfg.reward.scales.bad_pad_contact == -8.0
    assert cfg.reward.scales.finger_jitter == -6.0
    assert cfg.reward.scales.palm_assist == 5.0
    assert cfg.reward.scales.palm_only == -8.0
    assert cfg.reward.scales.pocket_without_finger_contact == -14.0
    assert cfg.reward.scales.ball_bounce == -18.0
    assert cfg.reward.scales.action_rate == -3.0
    assert cfg.reward.scales.settled_action == -5.0
    assert cfg.reward.scales.torque == -0.0005
    assert cfg.reward.scales.drop == -150.0
    assert cfg.env.grasp_pocket_offset == [0.005, 0.012, -0.045]
    assert cfg.env.initial_pose == [
        0.35,
        -0.25,
        0.60,
        0.45,
        0.30,
        0.0,
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
    ]
    assert cfg.env.catch_pose == [
        0.65,
        -0.50,
        1.35,
        0.90,
        0.6,
        0.0,
        1.36,
        0.70,
        0.8,
        0.50,
        1.52,
        1.10,
        1.0,
        1.0,
        0.7,
        1.0,
    ]
    assert cfg.env.catch_close_trigger_height == 0.11
    assert cfg.env.catch_full_close_height == 0.015
    assert cfg.env.early_closure_tolerance == 0.12
    assert cfg.env.reference_pose_sigma == 4.0
    assert cfg.env.reference_pose_joint_weights == [
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
    ]
    assert cfg.env.catch_spread_tolerance == 0.02
    assert cfg.env.catch_spread_limit == 0.08
    assert cfg.env.locked_catch_dof_indices == [12]
    assert cfg.env.pose_preview == "none"
    assert cfg.env.thumb_pregrasp_pose == [0.8, 0.6, 0.9, 0.6]
    assert cfg.env.natural_grasp_pose == [
        0.62,
        -0.14,
        0.78,
        0.38,
        0.74,
        0.0,
        0.88,
        0.46,
        0.84,
        0.06,
        1.04,
        0.58,
        0.72,
        0.48,
        0.62,
        0.32,
    ]
    assert cfg.env.closing_grasp_pose == [
        1.0,
        -0.14,
        1.15,
        0.72,
        1.12,
        0.0,
        1.28,
        0.82,
        1.0,
        0.0,
        1.6,
        0.5,
        1.4,
        0.6,
        0.5,
        0.5,
    ]
    assert cfg.env.ball_spawn_height_range == [0.82, 0.88]
    assert cfg.env.ball_horizontal_velocity_range == [-0.08, 0.08]
    assert cfg.env.ball_radius == 0.0335
    assert cfg.env.fingertip_surface_margin == 0.006
    assert cfg.env.ctrl_dt == 0.01
    assert cfg.env.control_config.action_scale == 0.14
    assert cfg.env.control_config.kp == 34.0
    assert cfg.env.control_config.kd == 2.80


def test_leap_observation_exposes_object_rotation_state() -> None:
    pytest.importorskip("mujoco")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.rotation import (
        LeapInhandRotationCfg,
        LeapInhandRotationEnv,
    )

    reward = RewardConfigPPO(
        scales={"rotate": 1.5},
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        reset_z_threshold=0.045,
    )
    env = LeapInhandRotationEnv(
        LeapInhandRotationCfg(reward_config=reward),
        num_envs=1,
        backend_type="mujoco",
    )

    obs, _ = env.reset(np.array([0], dtype=np.int32))
    state = env.step(np.zeros((1, 16), dtype=np.float32))

    assert obs["obs"].shape == (1, 180)
    assert state.obs["obs"].shape == (1, 180)
    assert env.obs_groups_spec == {"obs": 180}
    contact_flags = state.obs["obs"][0, -25:-21]
    assert np.logical_or(contact_flags == 0.0, contact_flags == 1.0).all()
    np.testing.assert_allclose(state.obs["obs"][0, -21:-18], state.info["curr_ball_angvel"][0])
    assert np.linalg.norm(state.obs["obs"][0, -18:-16]) == pytest.approx(1.0)
    assert np.isfinite(state.obs["obs"]).all()


def test_leap_rotation_partial_reset_preserves_observation_batch() -> None:
    pytest.importorskip("mujoco")
    from unilab.envs.manipulation.leap_inhand.rotation import (
        LeapInhandRotationCfg,
        LeapInhandRotationEnv,
        RewardConfigPPO,
    )

    reward = RewardConfigPPO(
        scales={"rotate": 1.0},
        angvel_clip_min=0.0,
        angvel_clip_max=1.5,
        reset_z_threshold=0.54,
    )
    env = LeapInhandRotationEnv(
        LeapInhandRotationCfg(reward_config=reward),
        num_envs=4,
        backend_type="mujoco",
    )

    env.init_state()
    obs, _ = env.reset(np.array([2], dtype=np.int32))

    assert obs["obs"].shape == (1, 180)
    assert env.state is not None
    assert env.state.obs["obs"].shape == (4, 180)
    assert np.isfinite(obs["obs"]).all()

def test_leap_rotation_uses_owned_grasp_and_allegro_reward_terms() -> None:
    pytest.importorskip("mujoco")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.rotation import (
        LeapInhandRotationCfg,
        LeapInhandRotationEnv,
    )

    scales = {
        "rotate": 2.0,
        "palm_contact": -0.25,
        "obj_linvel": -0.3,
        "pose_diff": -0.03,
        "torque": -0.01,
        "work": -0.001,
    }
    reward = RewardConfigPPO(
        scales=scales,
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        reset_z_threshold=0.48,
    )
    cfg = LeapInhandRotationCfg(reward_config=reward)
    cfg.use_grasp_cache = False
    cfg.domain_rand.joint_noise = 0.0
    cfg.domain_rand.ball_vel_noise = 0.0
    env = LeapInhandRotationEnv(cfg, num_envs=1, backend_type="mujoco")

    env.reset(np.array([0], dtype=np.int32))
    assert env.get_hand_dof_pos()[0] == pytest.approx(env.default_angles, abs=1e-5)
    assert env.get_ball_pos()[0] == pytest.approx(env._init_qpos[16:19], abs=1e-5)
    assert set(scales) <= set(env._reward_fns)

    state = env.step(np.zeros((1, 16), dtype=np.float32))
    assert np.isfinite(state.reward).all()
    assert not state.terminated[0]


def test_leap_motrix_reset_step_has_finite_observation() -> None:
    pytest.importorskip("motrixsim")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.rotation import (
        LeapInhandRotationCfg,
        LeapInhandRotationEnv,
    )

    reward = RewardConfigPPO(
        scales={"rotate": 1.5},
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        reset_z_threshold=0.045,
    )
    env = LeapInhandRotationEnv(
        LeapInhandRotationCfg(reward_config=reward),
        num_envs=2,
        backend_type="motrix",
    )

    obs, _ = env.reset(np.array([0, 1], dtype=np.int32))
    state = env.step(np.zeros((2, 16), dtype=np.float32))

    assert obs["obs"].shape == (2, 180)
    assert state.obs["obs"].shape == (2, 180)
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(env._ctrl_lower).all()
    assert np.isfinite(env._ctrl_upper).all()


def test_leap_ball_catch_passive_palm_support_is_not_opposing_grasp() -> None:
    pytest.importorskip("motrixsim")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.catch import (
        LeapBallCatchCfg,
        LeapBallCatchEnv,
    )

    reward = RewardConfigPPO(
        scales={"catch": 5.0, "drop": -10.0},
        angvel_clip_min=-0.5,
        angvel_clip_max=0.5,
        reset_z_threshold=0.4,
    )
    env = LeapBallCatchEnv(
        LeapBallCatchCfg(
            reward_config=reward,
            gen_grasp=True,
            ball_spawn_x_range=(-0.03, -0.03),
            ball_spawn_y_range=(0.04, 0.04),
            ball_spawn_height_range=(0.85, 0.85),
            ball_horizontal_velocity_range=(0.0, 0.0),
            ball_vertical_velocity_range=(0.0, 0.0),
        ),
        num_envs=1,
        backend_type="motrix",
    )
    obs, _ = env.reset(np.array([0], dtype=np.int32))

    state = None
    for _ in range(100):
        state = env.step(np.zeros((1, 16), dtype=np.float32))
        assert not state.terminated[0]

    assert state is not None
    assert obs["obs"].shape == (1, 153)
    assert state.obs["obs"].shape == (1, 153)
    assert env.obs_groups_spec == {"obs": 153}
    assert env._finger_contacts().shape == (1, 4)
    assert env._palm_contact().shape == (1,)
    pad_positions, pad_normals, pad_proximity, pad_alignment, pad_to_ball = env._read_pad_geometry(
        env.get_ball_pos()
    )
    assert pad_positions.shape == (1, 4, 3)
    assert pad_normals.shape == (1, 4, 3)
    assert pad_proximity.shape == (1, 4)
    assert pad_alignment.shape == (1, 4)
    assert pad_to_ball.shape == (1, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(pad_normals, axis=2), 1.0, atol=1e-5)
    assert env.get_ball_pos()[0, 2] > 0.4
    _, wrap, _ = env._finger_geometry(env.get_ball_pos())
    assert wrap[0] < 0.8


@pytest.mark.parametrize(
    ("backend_type", "dependency"),
    [("mujoco", "mujoco"), ("motrix", "motrixsim")],
)
def test_leap_ball_catch_pocket_is_at_main_finger_roots(backend_type: str, dependency: str) -> None:
    pytest.importorskip(dependency)
    from unilab.envs.manipulation.allegro_inhand.base import ControlConfig
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.catch import (
        LeapBallCatchCfg,
        LeapBallCatchEnv,
    )

    env = LeapBallCatchEnv(
        LeapBallCatchCfg(
            reward_config=RewardConfigPPO(
                scales={"hold": 1.0},
                angvel_clip_min=-0.5,
                angvel_clip_max=0.5,
                reset_z_threshold=0.4,
            ),
            gen_grasp=True,
            control_config=ControlConfig(action_scale=0.08, kp=8.0, kd=0.35),
        ),
        num_envs=1,
        backend_type=backend_type,
    )
    geom_names = env._backend.get_geom_names()
    geom_contype, geom_conaffinity = env._backend.get_geom_contact_masks()
    expected_masks = {
        "fingertip_collision": (2, 29),
        "fingertip_2_collision": (4, 27),
        "fingertip_3_collision": (8, 23),
        "thumb_fingertip_collision": (16, 15),
    }
    resolved_masks = {
        name: (
            int(geom_contype[geom_names.index(name)]),
            int(geom_conaffinity[geom_names.index(name)]),
        )
        for name in expected_masks
    }
    assert resolved_masks == expected_masks

    def can_collide(first: str, second: str) -> bool:
        first_type, first_affinity = resolved_masks[first]
        second_type, second_affinity = resolved_masks[second]
        return bool((first_type & second_affinity) != 0 or (second_type & first_affinity) != 0)

    finger_tips = tuple(expected_masks)
    assert all(
        can_collide(first, second)
        for index, first in enumerate(finger_tips)
        for second in finger_tips[index + 1 :]
    )

    pocket = env._grasp_pocket_center()
    early_ball = pocket.copy()
    early_ball[:, 2] += env.cfg.catch_close_trigger_height + 0.01
    early_contacts = np.ones((1, 4), dtype=np.float32)
    assert env._catch_close_phase(early_ball, early_contacts)[0] == pytest.approx(0.0)

    fully_lowered_ball = pocket.copy()
    fully_lowered_ball[:, 2] += env.cfg.catch_full_close_height
    assert env._catch_close_phase(fully_lowered_ball, early_contacts)[0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        env._catch_reference_pose(early_ball, early_contacts)[0],
        env._initial_pose,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        env._catch_reference_pose(fully_lowered_ball, early_contacts)[0],
        env._catch_pose,
        atol=1e-6,
    )

    zeros = np.zeros((1, 3), dtype=np.float32)
    zero_torque = np.zeros((1, 16), dtype=np.float32)
    not_done = np.zeros((1,), dtype=bool)
    exact_reward = env._reward_reference_pose_tracking(
        {"curr_contacts": early_contacts},
        env._catch_pose[None, :],
        zero_torque,
        fully_lowered_ball,
        zeros,
        zeros,
        zero_torque,
        not_done,
    )
    offset_reward = env._reward_reference_pose_tracking(
        {"curr_contacts": early_contacts},
        env._catch_pose[None, :] + 0.2,
        zero_torque,
        fully_lowered_ball,
        zeros,
        zeros,
        zero_torque,
        not_done,
    )
    assert exact_reward[0] == pytest.approx(1.0)
    assert 0.0 < offset_reward[0] < exact_reward[0]
    early_pose = env._initial_pose[None, :] + 0.75 * (env._catch_pose - env._initial_pose)[None, :]
    early_penalty = env._reward_early_closure(
        {"curr_contacts": np.zeros_like(early_contacts)},
        early_pose,
        zero_torque,
        early_ball,
        zeros,
        zeros,
        zero_torque,
        not_done,
    )
    on_time_penalty = env._reward_early_closure(
        {"curr_contacts": early_contacts},
        env._catch_pose[None, :],
        zero_torque,
        fully_lowered_ball,
        zeros,
        zeros,
        zero_torque,
        not_done,
    )
    assert early_penalty[0] > 0.0
    assert on_time_penalty[0] == pytest.approx(0.0)

    env.reset(np.array([0], dtype=np.int32))

    pocket = env._grasp_pocket_center()[0]
    mcp_center = np.mean(
        env._backend.get_body_pos_w(env._main_mcp_body_ids)[0],
        axis=0,
    )

    palm_center = env._backend.get_body_pos_w(env._palm_body_ids)[0, 0]
    assert pocket.tolist() == pytest.approx([-0.002, 0.010, 0.564], abs=0.004)
    assert pocket[0] > mcp_center[0]
    assert pocket[2] > mcp_center[2]
    assert abs(pocket[1] - palm_center[1]) < abs(mcp_center[1] - palm_center[1])
    kp, kd = env._backend.get_actuator_gains()
    np.testing.assert_allclose(kp, 8.0)
    np.testing.assert_allclose(kd, 0.35)


@pytest.mark.parametrize(
    ("backend_type", "dependency"),
    [("mujoco", "mujoco"), ("motrix", "motrixsim")],
)
def test_leap_ball_catch_rl_control_is_independent_with_explicit_safety_limits(
    backend_type: str,
    dependency: str,
) -> None:
    pytest.importorskip(dependency)
    from unilab.envs.manipulation.allegro_inhand.base import ControlConfig
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.catch import (
        LeapBallCatchCfg,
        LeapBallCatchEnv,
    )

    env = LeapBallCatchEnv(
        LeapBallCatchCfg(
            reward_config=RewardConfigPPO(
                scales={"hold": 1.0},
                angvel_clip_min=-0.5,
                angvel_clip_max=0.5,
                reset_z_threshold=0.4,
            ),
            gen_grasp=True,
            ctrl_dt=0.01,
            control_config=ControlConfig(action_scale=0.08, kp=8.0, kd=0.35),
            ball_spawn_x_range=(0.018, 0.018),
            ball_spawn_y_range=(0.022, 0.022),
            ball_spawn_height_range=(0.85, 0.85),
            ball_horizontal_velocity_range=(0.0, 0.0),
            ball_vertical_velocity_range=(0.0, 0.0),
        ),
        num_envs=1,
        backend_type=backend_type,
    )
    env.reset(np.array([0], dtype=np.int32))
    np.testing.assert_allclose(env.get_hand_dof_pos()[0], env._initial_pose, atol=1e-6)

    state = env.step(np.zeros((1, 16), dtype=np.float32))
    np.testing.assert_allclose(state.info["prev_ctrl"][0], env._initial_pose, atol=1e-6)

    independent_actions = np.zeros((1, 16), dtype=np.float32)
    independent_actions[:, [1, 5, 9, 12]] = 1.0
    independent_actions[:, [2, 6, 10]] = 1.0
    independent_actions[:, [3, 7, 11]] = -1.0
    state = env.step(independent_actions)

    reference = state.info["catch_reference_pose"][0]
    spread = state.info["prev_ctrl"][0, [1, 5, 9]]
    assert np.all(spread <= reference[[1, 5, 9]] + env.cfg.catch_spread_limit + 1e-6)
    assert np.all(spread >= reference[[1, 5, 9]] - env.cfg.catch_spread_limit - 1e-6)
    assert state.info["prev_ctrl"][0, 12] == pytest.approx(env._initial_pose[12])
    for middle_id, tip_id in ((2, 3), (6, 7), (10, 11)):
        assert state.info["prev_ctrl"][0, middle_id] > env._initial_pose[middle_id]
        assert state.info["prev_ctrl"][0, tip_id] < env._initial_pose[tip_id]


def test_leap_ball_catch_reset_randomization_stays_in_configured_ranges() -> None:
    pytest.importorskip("motrixsim")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.catch import (
        LeapBallCatchCfg,
        LeapBallCatchEnv,
    )

    cfg = LeapBallCatchCfg(
        reward_config=RewardConfigPPO(
            scales={"hold": 5.0},
            angvel_clip_min=-0.5,
            angvel_clip_max=0.5,
            reset_z_threshold=0.4,
        ),
        gen_grasp=True,
    )
    env = LeapBallCatchEnv(cfg, num_envs=4, backend_type="motrix")
    _, ball_pos, _, qvel = env._domain_randomization_provider()._sample_reset_state(env, 512)

    assert np.all((0.008 <= ball_pos[:, 0]) & (ball_pos[:, 0] <= 0.024))
    assert np.all((0.016 <= ball_pos[:, 1]) & (ball_pos[:, 1] <= 0.028))
    assert np.all((0.82 <= ball_pos[:, 2]) & (ball_pos[:, 2] <= 0.88))
    assert np.all((-0.08 <= qvel[:, 16:18]) & (qvel[:, 16:18] <= 0.08))
    assert np.all((-0.08 <= qvel[:, 18]) & (qvel[:, 18] <= 0.0))


def test_leap_ball_catch_partial_reset_preserves_observation_batch() -> None:
    pytest.importorskip("motrixsim")
    from unilab.envs.manipulation.allegro_inhand.rotation import RewardConfigPPO
    from unilab.envs.manipulation.leap_inhand.catch import (
        LeapBallCatchCfg,
        LeapBallCatchEnv,
    )

    env = LeapBallCatchEnv(
        LeapBallCatchCfg(
            reward_config=RewardConfigPPO(
                scales={"hold": 1.0},
                angvel_clip_min=-0.5,
                angvel_clip_max=0.5,
                reset_z_threshold=0.4,
            ),
            gen_grasp=True,
        ),
        num_envs=4,
        backend_type="motrix",
    )
    env.init_state()

    obs, _ = env.reset(np.asarray([2], dtype=np.int32))

    assert obs["obs"].shape == (1, 153)
