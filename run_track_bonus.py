#!/usr/bin/env python3
"""Evaluate a HW1 Go2 checkpoint on the 200 m track bonus task.

This benchmark is hierarchical:

    high-level planner: track coordinates -> [vx, vy, yaw_rate]
    low-level policy:  proprioception + command -> 12 joint actions

The starter planner is intentionally weak. It exists to make the interface
concrete and to give students a baseline to improve.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from competition.track_scene import render_track_video
from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    set_runtime_env,
)
from go2_pg_env.track import StandardOvalTrack
from test_policy import load_policy_with_workaround
from track_bonus.planner import StarterTrackPlanner
from track_bonus.scoring import compute_track_bonus_metrics, score_track_bonus


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Low-level Brax PPO best_checkpoint directory.")
    parser.add_argument("--planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="HW1 course config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--start-s-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--render-fps", type=int, default=5)
    parser.add_argument("--render-camera-profile", choices=["showcase", "close", "overview"], default="showcase")
    return parser.parse_args()


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_checkpoint(checkpoint_dir: Path) -> None:
    config_path = checkpoint_dir / "ppo_network_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = payload.get("network_factory_kwargs", {})
    obs_key = kwargs.get("policy_obs_key")
    if obs_key != "state":
        raise ValueError(f"Expected actor policy_obs_key='state', got {obs_key!r}.")


def _make_env(stack: dict[str, Any], course_cfg: dict[str, Any], stage_name: str, episode_steps: int) -> Any:
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    env_name = course_cfg["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, course_cfg["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, course_cfg, stage_name)
    env_cfg.episode_length = int(episode_steps)
    env_cfg.noise_config.level = 0.0
    env_cfg.pert_config.enable = False
    return registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(course_cfg))


def _reset_lowlevel_on_track(
    *,
    stack: dict[str, Any],
    env: Any,
    rng: Any,
    track: StandardOvalTrack,
    start_s: float,
) -> Any:
    jax = stack["jax"]
    jp = jax.numpy
    from mujoco import mjx
    from mujoco.mjx._src import math as mjmath
    from mujoco_playground._src import mjx_env

    state = env.reset(rng)
    qpos = env._init_q
    qvel = jp.zeros(env.mjx_model.nv)
    xy, heading, _ = track.centerline_pose(float(start_s))
    quat = mjmath.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), jp.asarray(heading, dtype=jp.float32))
    qpos = qpos.at[0:2].set(jp.asarray(xy, dtype=jp.float32))
    qpos = qpos.at[3:7].set(quat)
    data = mjx_env.make_data(
        env.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=env.mjx_model.impl.value,
        naconmax=env._config.naconmax,
        njmax=env._config.njmax,
    )
    data = mjx.forward(env.mjx_model, data)
    state.info["command"] = jp.zeros(3, dtype=jp.float32)
    state.info["steps_until_next_cmd"] = jp.asarray(10**9, dtype=jp.int32)
    obs = env._get_obs(data, state.info)
    return state.replace(data=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(()))


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = jax.numpy.asarray(10**9, dtype=jax.numpy.int32)
    return state


def rollout(
    *,
    stack: dict[str, Any],
    env: Any,
    policy: Any,
    planner: StarterTrackPlanner,
    num_steps: int,
    seed: int,
    start_s: float,
    force_cpu: bool,
) -> dict[str, Any]:
    jax = stack["jax"]
    rng = jax.random.PRNGKey(int(seed))
    rng, reset_key = jax.random.split(rng)
    state = _reset_lowlevel_on_track(stack=stack, env=env, rng=reset_key, track=planner.track, start_s=start_s)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    qpos = []
    commands = []
    done = []
    fall = []
    joint_torques = []
    joint_velocities = []
    foot_slip_speed = []
    frozen = False
    frozen_snapshot: dict[str, Any] = {}

    for step_idx in range(num_steps):
        if frozen:
            snap = frozen_snapshot
        else:
            qpos_now = np.asarray(state.data.qpos, dtype=np.float32)
            command = planner.command(qpos_now, t=step_idx * env.dt)
            state = _force_command(state, command, jax)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, command, jax)

            feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], dtype=np.float32)
            snap = {
                "qpos": np.asarray(state.data.qpos, dtype=np.float32),
                "command": command,
                "done": bool(np.asarray(state.done)),
                "fall": bool(np.asarray(state.done)),
                "joint_torques": np.asarray(state.data.actuator_force, dtype=np.float32),
                "joint_velocities": np.asarray(state.data.qvel[6:], dtype=np.float32),
                "foot_slip_speed": np.linalg.norm(feet_vel[:, :2], axis=-1).astype(np.float32),
            }

            projection = planner.track.project_xy_to_track(snap["qpos"][:2])
            terminal = snap["done"] or projection.out_of_bounds
            if terminal:
                frozen = True
                frozen_snapshot = snap

        qpos.append(snap["qpos"])
        commands.append(snap["command"])
        done.append(snap["done"])
        fall.append(snap["fall"])
        joint_torques.append(snap["joint_torques"])
        joint_velocities.append(snap["joint_velocities"])
        foot_slip_speed.append(snap["foot_slip_speed"])

    return {
        "dt": float(env.dt),
        "qpos": np.asarray(qpos, dtype=np.float32),
        "command": np.asarray(commands, dtype=np.float32),
        "done": np.asarray(done, dtype=bool),
        "fall": np.asarray(fall, dtype=bool),
        "joint_torques": np.asarray(joint_torques, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "foot_slip_speed": np.asarray(foot_slip_speed, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    _validate_checkpoint(args.checkpoint_dir)
    force_cpu = bool(args.force_cpu)
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    course_cfg = load_json(args.config)
    course_cfg["runtime_overrides"] = {}
    num_steps = int(round(float(args.duration_seconds) / float(course_cfg["control"]["ctrl_dt"])))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = StarterTrackPlanner.load(args.planner_config.resolve())
    stack = lazy_import_stack()
    env = _make_env(stack, course_cfg, args.stage_name, episode_steps=num_steps)
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = stack["jax"].jit(policy)

    result = rollout(
        stack=stack,
        env=env,
        policy=policy,
        planner=planner,
        num_steps=num_steps,
        seed=int(args.seed),
        start_s=float(args.start_s_m),
        force_cpu=force_cpu,
    )
    metrics = compute_track_bonus_metrics(result, planner.track)
    scores = score_track_bonus(metrics)

    np.savez_compressed(
        output_dir / "race_rollouts.npz",
        policy_names=np.asarray(["starter"]),
        qpos=result["qpos"][None, ...],
        dt=np.asarray([result["dt"]], dtype=np.float32),
        command=result["command"],
    )
    (output_dir / "leaderboard.csv").write_text(
        "rank,name,composite_score,lap_completion,finish_time,mean_progress_speed,fall,boundary_violation,rms_lateral_error,max_lateral_error\n"
        f"1,starter,{scores['composite_score']},{metrics['lap_completion']},"
        f"{'' if metrics['finish_time'] is None else metrics['finish_time']},{metrics['mean_progress_speed']},"
        f"{metrics['fall']},{metrics['boundary_violation']},{metrics['rms_lateral_error']},{metrics['max_lateral_error']}\n",
        encoding="utf-8",
    )

    video_path = None
    if not args.no_render:
        video_path = render_track_video(
            trajectories_qpos=result["qpos"][None, ...],
            output_path=output_dir / "race.mp4",
            colors=["#2563EB"],
            fps=int(args.render_fps),
            render_every=int(args.render_every),
            width=int(args.render_width),
            height=int(args.render_height),
            camera_profile=str(args.render_camera_profile),
            track_config=planner.config.to_dict(),
        )

    payload = {
        "challenge": "track_bonus",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "planner_config": str(args.planner_config.resolve()),
        "metrics": metrics,
        "scores": scores,
        "artifacts": {
            "leaderboard_csv": str(output_dir / "leaderboard.csv"),
            "rollouts_npz": str(output_dir / "race_rollouts.npz"),
            "video_path": None if video_path is None else str(video_path),
        },
    }
    _save_json(output_dir / "results.json", payload)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics, "scores": scores}, indent=2), flush=True)


if __name__ == "__main__":
    main()
