"""View every pose in the LEAP Hand seed bank with MuJoCo.

By default, the viewer cycles through all seeds without advancing physics.
Use --seed-index to freeze the viewer on one specific seed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Use the native GLFW viewer on Windows.
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
ASSET_ROOT = SRC_DIR / "unilab" / "assets"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.base.backend.mujoco.xml import materialize_scene_fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically inspect every LEAP Hand seed-bank pose."
    )
    parser.add_argument(
        "--seed-bank",
        type=str,
        default="caches/leap_hand_seed_bank.npy",
        help=(
            "Seed-bank path. Relative paths are first resolved under "
            "src/unilab/assets, then under the repository root."
        ),
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help=(
            "Show only one seed and keep it frozen. "
            "Omit this option to cycle through every seed."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First seed shown when cycling through the bank.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds to display each seed when cycling.",
    )
    parser.add_argument(
        "--show-contacts",
        action="store_true",
        help="Display MuJoCo contact points and contact-force arrows.",
    )
    return parser.parse_args()


def resolve_seed_bank(path_text: str) -> Path:
    path = Path(path_text).expanduser()

    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                ASSET_ROOT / path,
                ROOT_DIR / path,
                Path.cwd() / path,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Seed bank was not found. Checked:\n{checked}"
    )


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    asset_dir = ASSET_ROOT / "robots" / "leap_hand"

    materialized_path = Path(
        materialize_scene_fragments(
            str(asset_dir / "leap_hand.xml"),
            fragment_files=[str(asset_dir / "scene.xml")],
        )
    )

    try:
        model = mujoco.MjModel.from_xml_path(str(materialized_path))
    finally:
        materialized_path.unlink(missing_ok=True)

    return model, mujoco.MjData(model)


def validate_seed_bank(
    seeds: np.ndarray,
    model: mujoco.MjModel,
) -> np.ndarray:
    seeds = np.asarray(seeds, dtype=np.float64)

    if seeds.ndim == 1:
        seeds = seeds.reshape(1, -1)

    if seeds.ndim != 2:
        raise ValueError(
            f"Seed bank must be a 2D array, but shape is {seeds.shape}."
        )

    if seeds.shape[0] == 0:
        raise ValueError("Seed bank contains no seeds.")

    if seeds.shape[1] < model.nq:
        raise ValueError(
            f"Each seed has {seeds.shape[1]} values, but the MuJoCo "
            f"model requires {model.nq} qpos values."
        )

    if not np.all(np.isfinite(seeds[:, : model.nq])):
        raise ValueError("Seed bank contains NaN or infinite qpos values.")

    return seeds


def apply_seed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seeds: np.ndarray,
    seed_index: int,
) -> None:
    state = seeds[seed_index]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = state[: model.nq]

    # Hold the hand actuator targets at the stored joint pose. No mj_step() is
    # called, so the pose remains completely static.
    control_count = min(model.nu, 16, state.size)
    if control_count:
        data.ctrl[:control_count] = state[:control_count]

    # Normalize the free-joint ball quaternion when the expected 23-value
    # LEAP state layout is present:
    # [16 hand qpos, 3 ball xyz, 4 ball quaternion].
    if model.nq >= 23:
        quaternion = data.qpos[19:23]
        norm = float(np.linalg.norm(quaternion))
        if norm > 1e-12:
            data.qpos[19:23] = quaternion / norm
        else:
            data.qpos[19:23] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

    mujoco.mj_forward(model, data)

    print()
    print(f"Showing seed {seed_index} / {len(seeds) - 1}")
    print(
        "hand qpos:",
        np.round(data.qpos[:16], 4).tolist(),
    )
    if model.nq >= 23:
        print(
            "ball xyz:",
            np.round(data.qpos[16:19], 5).tolist(),
        )


def main() -> int:
    args = parse_args()

    if args.interval <= 0.0:
        raise ValueError("--interval must be positive.")

    seed_bank_path = resolve_seed_bank(args.seed_bank)
    model, data = load_model()
    seeds = validate_seed_bank(np.load(seed_bank_path), model)

    print(f"Loaded seed bank: {seed_bank_path}")
    print(f"Seed-bank shape: {seeds.shape}")
    print("No physics is running; every displayed pose is frozen.")

    if args.seed_index is not None:
        if not 0 <= args.seed_index < len(seeds):
            raise IndexError(
                f"--seed-index must be between 0 and {len(seeds) - 1}."
            )
        current_index = args.seed_index
        automatic_cycle = False
    else:
        current_index = args.start_index % len(seeds)
        automatic_cycle = True
        print(
            f"Automatically cycling every {args.interval:g} seconds."
        )

    apply_seed(model, data, seeds, current_index)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        if args.show_contacts:
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            ] = True
            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
            ] = True
            viewer.opt.geomgroup[:] = 1

        next_switch_time = time.monotonic() + args.interval

        while viewer.is_running():
            now = time.monotonic()

            if automatic_cycle and now >= next_switch_time:
                current_index = (current_index + 1) % len(seeds)
                apply_seed(model, data, seeds, current_index)
                next_switch_time = now + args.interval

            # Intentionally no mj_step(): this is a static pose viewer.
            viewer.sync()
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())