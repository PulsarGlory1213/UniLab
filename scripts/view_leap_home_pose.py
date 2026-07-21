"""Open the LEAP Hand home keyframe in MuJoCo's native interactive viewer.

This viewer is static: it only displays the pose stored in scene.xml and does
not run gravity or training.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Use the normal Windows OpenGL/GLFW viewer.
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.base.backend.mujoco.xml import materialize_scene_fragments


def main() -> int:
    asset_dir = (
        ROOT_DIR
        / "src"
        / "unilab"
        / "assets"
        / "robots"
        / "leap_hand"
    )

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

    data = mujoco.MjData(model)

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )
    if home_key_id < 0:
        raise RuntimeError(
            'Keyframe "home" was not found in scene.xml.'
        )

    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)

    print("MuJoCo viewer opened.")
    print("This is the static 'home' pose from scene.xml.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        while viewer.is_running():
            # No mj_step(): keep the pose frozen.
            viewer.sync()
            time.sleep(0.02)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())