#!/usr/bin/env python3
"""Train a learned MLP high-level planner with CEM (replaces the scalar starter search).

The planner reads the 5D track observation and a small MLP (5 -> hidden -> 3)
outputs [vx, vy, yaw_rate]. We search the MLP weights with the Cross-Entropy
Method, scoring each candidate by a full lap rollout.

Speed trick: build the env + low-level policy ONCE and jit env.step ONCE, then
only swap the planner's numpy weights per candidate -> no JAX recompilation.

Usage:
  !python train_highlevel_starter.py \
      --checkpoint-dir artifacts/low_level_train/best_checkpoint \
      --planner-config configs/learned_planner.json \
      --teacher-config configs/starter_planner.json \
      --out-weights    configs/learned_planner_weights.npz \
      --iterations 15 --population 16 --elite 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from course_common import DEFAULT_CONFIG_PATH, lazy_import_stack, load_json, set_runtime_env
from run_track_bonus import _make_env, _reset_lowlevel_on_track, _force_command, _validate_checkpoint
from test_policy import load_policy_with_workaround
from track_bonus.controller_interface import build_track_controller_observation, validate_high_level_command
from track_bonus.official_track import official_track
from track_bonus.planner import N_IN, N_OUT, StarterPlannerConfig, StarterTrackPlanner, num_mlp_params
from track_bonus.scoring import compute_track_bonus_metrics

# ---- fixed settings (tweak here if you want) -------------------------------
SEED = 20260527          # same seed the evaluator uses -> search score predicts final time
SEARCH_SECONDS = 130.0   # rollout length during search (your baseline laps in ~105 s)
INIT_SIGMA = 0.3         # CEM starting exploration spread
SIGMA_FLOOR = 0.02       # stops CEM from collapsing to a point too early
CLONE_STEPS = 600        # Adam steps for the warm-start imitation fit


# ---- 1. rollout: run one deterministic lap with a given planner ------------
def run_rollout(stack, env, policy, step_fn, planner, track, num_steps):
    jax = stack["jax"]
    rng = jax.random.PRNGKey(SEED)
    rng, reset_key = jax.random.split(rng)
    state = _reset_lowlevel_on_track(stack=stack, env=env, rng=reset_key, track=track, start_s=0.0)

    qpos, cmds, tobs, done, fall, jt, jv, slip = [], [], [], [], [], [], [], []
    frozen, snap = False, {}
    for i in range(num_steps):
        if not frozen:
            q = np.asarray(state.data.qpos, np.float32)
            obs = build_track_controller_observation(qpos=q, track=track)
            cmd = validate_high_level_command(planner.command(obs, t=i * env.dt))
            state = _force_command(state, cmd, jax)              # tell low-level what to track
            rng, ak = jax.random.split(rng)
            action, _ = policy(state.obs, ak)                    # low-level picks joint targets
            state = step_fn(state, action)                       # advance the sim (reused jit)
            state = _force_command(state, cmd, jax)
            fv = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], np.float32)
            snap = {
                "qpos": np.asarray(state.data.qpos, np.float32),
                "command": cmd,
                "track_observation": obs.as_array(),
                "done": bool(np.asarray(state.done)),
                "fall": bool(np.asarray(state.done)),
                "joint_torques": np.asarray(state.data.actuator_force, np.float32),
                "joint_velocities": np.asarray(state.data.qvel[6:], np.float32),
                "foot_slip_speed": np.linalg.norm(fv[:, :2], axis=-1).astype(np.float32),
            }
            if snap["done"] or track.project_xy_to_track(snap["qpos"][:2]).out_of_bounds:
                frozen = True                                    # crashed / off track -> freeze
        qpos.append(snap["qpos"]); cmds.append(snap["command"]); tobs.append(snap["track_observation"])
        done.append(snap["done"]); fall.append(snap["fall"]); jt.append(snap["joint_torques"])
        jv.append(snap["joint_velocities"]); slip.append(snap["foot_slip_speed"])

    return {
        "dt": float(env.dt),
        "qpos": np.asarray(qpos, np.float32),
        "command": np.asarray(cmds, np.float32),
        "track_observation": np.asarray(tobs, np.float32),
        "done": np.asarray(done, bool),
        "fall": np.asarray(fall, bool),
        "joint_torques": np.asarray(jt, np.float32),
        "joint_velocities": np.asarray(jv, np.float32),
        "foot_slip_speed": np.asarray(slip, np.float32),
    }


# ---- 2. objective: turn the lap metrics into one number to maximize --------
def objective(metrics):
    # complete-then-fast: a finished lap always beats an unfinished one,
    # and among finished laps, faster (smaller finish_time) scores higher.
    if metrics["lap_completion"] >= 1.0 and metrics["finish_time"] is not None:
        return 1000.0 - float(metrics["finish_time"])
    return float(metrics["valid_distance_m"])


# ---- 3. inverse bounds: command -> the raw MLP outputs that produce it ------
def inverse_bounds(cmd, cfg):
    eps = 1e-4
    p = min(max((cmd[0] - cfg.vx_min) / max(cfg.vx_max - cfg.vx_min, 1e-6), eps), 1 - eps)
    t1 = min(max(cmd[1] / max(cfg.vy_max, 1e-6), -1 + eps), 1 - eps)
    t2 = min(max(cmd[2] / max(cfg.yaw_rate_max, 1e-6), -1 + eps), 1 - eps)
    return np.array([math.log(p / (1 - p)), math.atanh(t1), math.atanh(t2)])


# ---- 4. warm start: fit the MLP to imitate the teacher (Adam) --------------
def clone(X, Y, hidden):
    rng = np.random.default_rng(SEED)
    w1 = rng.normal(0, 0.3, (hidden, N_IN)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.3, (N_OUT, hidden)); b2 = np.zeros(N_OUT)
    P = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    n = len(X)
    for t in range(1, CLONE_STEPS + 1):
        h = np.tanh(X @ w1.T + b1)            # (n, hidden)
        pred = h @ w2.T + b2                  # (n, 3)
        d = (2.0 / n) * (pred - Y)            # dLoss/dpred
        dpre = (d @ w2) * (1.0 - h ** 2)      # backprop through tanh
        grads = [dpre.T @ X, dpre.sum(0), d.T @ h, d.sum(0)]
        for i, g in enumerate(grads):         # Adam update
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            P[i] -= 5e-2 * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        w1, b1, w2, b2 = P
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])  # flat theta (planner's layout)


# ---- 5. CEM: search the weight vector ---------------------------------------
def cem(score, mean, dim, iters, population, elite):
    rng = np.random.default_rng(SEED)
    sigma = np.full(dim, INIT_SIGMA)
    best_theta, best = mean.copy(), score(mean)
    print(f"[cem] warm-start score = {best:.2f}", flush=True)
    for it in range(iters):
        pop = mean + sigma * rng.standard_normal((population, dim))   # sample candidates
        scores = np.array([score(th) for th in pop])                 # score each by a lap
        order = np.argsort(scores)[::-1]                             # best first
        elites = pop[order[:elite]]                                  # keep the top few
        mean, sigma = elites.mean(0), elites.std(0) + SIGMA_FLOOR    # refit the Gaussian
        if scores[order[0]] > best:
            best, best_theta = float(scores[order[0]]), pop[order[0]].copy()
        print(f"[cem] iter {it+1}/{iters}  best={best:.2f}  this_iter={scores[order[0]]:.2f}", flush=True)
    return best_theta, best


# ---- 6. wire it together ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--planner-config", type=Path, required=True, help="learned_mlp config")
    p.add_argument("--teacher-config", type=Path, default=Path("configs/starter_planner.json"))
    p.add_argument("--out-weights", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--elite", type=int, default=4)
    return p.parse_args()


def main():
    a = parse_args()
    _validate_checkpoint(a.checkpoint_dir)
    set_runtime_env(force_cpu=False)

    course = load_json(a.config)
    course["runtime_overrides"] = {}
    ctrl_dt = float(course["control"]["ctrl_dt"])
    steps = int(round(SEARCH_SECONDS / ctrl_dt))

    track = official_track()
    cfg = StarterPlannerConfig.load(a.planner_config)
    if cfg.planner_type != "learned_mlp":
        raise SystemExit("--planner-config must have planner_type='learned_mlp'.")
    hidden = int(cfg.hidden_size)
    dim = num_mlp_params(hidden)

    # build env + policy ONCE; jit step ONCE
    stack = lazy_import_stack()
    jax = stack["jax"]
    env = _make_env(stack, course, "stage_2", episode_steps=steps)
    policy = jax.jit(load_policy_with_workaround(a.checkpoint_dir.resolve(), deterministic=True))
    step_fn = jax.jit(env.step)
    planner = StarterTrackPlanner(cfg, weights=np.zeros(dim))

    def score(theta):
        planner.set_weights(theta)
        result = run_rollout(stack, env, policy, step_fn, planner, track, steps)
        return objective(compute_track_bonus_metrics(result, track))

    # warm start: clone the starter so iteration 0 already laps
    print("[clone] rolling out teacher to collect imitation data...", flush=True)
    teacher = StarterTrackPlanner.load(a.teacher_config)
    demo = run_rollout(stack, env, policy, step_fn, teacher, track, steps)
    skip = int(round(cfg.stand_seconds / ctrl_dt))          # drop the start stand
    X = demo["track_observation"][skip:].astype(np.float64)
    Y = np.array([inverse_bounds(c, cfg) for c in demo["command"][skip:]])
    theta0 = clone(X, Y, hidden)

    best_theta, best = cem(score, theta0, dim, a.iterations, a.population, a.elite)

    a.out_weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out_weights, theta=best_theta.astype(np.float64), hidden_size=hidden)
    print(json.dumps({
        "saved": str(a.out_weights),
        "best_objective": best,
        "approx_finish_time_s": round(1000.0 - best, 2) if best > 200 else None,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# """Tiny black-box search scaffold for the track bonus high-level planner.

# This is intentionally simple and intentionally not a solved planner. It searches
# over a small JSON controller by repeatedly running `run_track_bonus.py --no-render`.
# Use it to debug the evaluation loop. For a leaderboard submission, replace the
# planner internals with a learned policy that maps the official 5D observation
# to `[vx, vy, yaw_rate]`.
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# import subprocess
# import sys
# from typing import Any

# import numpy as np

# from track_bonus.planner import StarterPlannerConfig


# ROOT = Path(__file__).resolve().parent


# SEARCH_KEYS = [
#     "speed_mps",
#     "max_lateral_speed_mps",
#     "max_yaw_rate_radps",
#     "k_heading",
#     "k_lateral",
#     "heading_slowdown",
# ]


# BOUNDS = {
#     "speed_mps": (0.20, 0.90),
#     "max_lateral_speed_mps": (0.03, 0.22),
#     "max_yaw_rate_radps": (0.12, 0.75),
#     "k_heading": (0.20, 1.40),
#     "k_lateral": (0.02, 0.24),
#     "heading_slowdown": (0.0, 0.80),
# }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--checkpoint-dir", type=Path, required=True)
#     parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
#     parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
#     parser.add_argument("--output-dir", type=Path, required=True)
#     parser.add_argument("--iterations", type=int, default=8)
#     parser.add_argument("--population", type=int, default=12)
#     parser.add_argument("--eval-seconds", type=float, default=60.0)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--force-cpu", action="store_true")
#     return parser.parse_args()


# def _write_json(path: Path, payload: Any) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# def _clip_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
#     clipped = dict(candidate)
#     for key, (low, high) in BOUNDS.items():
#         clipped[key] = float(np.clip(float(clipped[key]), low, high))
#     clipped["min_speed_mps"] = min(float(clipped.get("min_speed_mps", 0.12)), float(clipped["speed_mps"]))
#     return clipped


# def _sample_candidate(center: dict[str, Any], scale: float, rng: np.random.Generator) -> dict[str, Any]:
#     candidate = dict(center)
#     for key in SEARCH_KEYS:
#         low, high = BOUNDS[key]
#         sigma = scale * (high - low)
#         candidate[key] = float(center[key] + rng.normal(0.0, sigma))
#     return _clip_candidate(candidate)


# def _run_eval(
#     *,
#     checkpoint_dir: Path,
#     planner_path: Path,
#     config: Path,
#     output_dir: Path,
#     eval_seconds: float,
#     force_cpu: bool,
# ) -> float:
#     cmd = [
#         sys.executable,
#         "run_track_bonus.py",
#         "--checkpoint-dir",
#         str(checkpoint_dir),
#         "--planner-config",
#         str(planner_path),
#         "--config",
#         str(config),
#         "--output-dir",
#         str(output_dir),
#         "--duration-seconds",
#         str(eval_seconds),
#         "--no-render",
#     ]
#     if force_cpu:
#         cmd.append("--force-cpu")
#     try:
#         subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
#         payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
#         return float(payload["scores"]["composite_score"])
#     except Exception:
#         return -1.0


# def main() -> None:
#     args = parse_args()
#     rng = np.random.default_rng(int(args.seed))
#     output_dir = args.output_dir.resolve()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     base = StarterPlannerConfig.load(args.base_planner_config).to_dict()
#     center = _clip_candidate(base)
#     best = dict(center)
#     best_score = -1.0
#     history = []

#     for iteration in range(int(args.iterations)):
#         scale = max(0.04, 0.18 * (0.72**iteration))
#         candidates = [dict(best if best_score >= 0.0 else center)]
#         while len(candidates) < int(args.population):
#             candidates.append(_sample_candidate(best if best_score >= 0.0 else center, scale, rng))

#         for candidate_idx, candidate in enumerate(candidates):
#             candidate_dir = output_dir / "candidates" / f"iter_{iteration:02d}_cand_{candidate_idx:02d}"
#             planner_path = candidate_dir / "planner_config.json"
#             _write_json(planner_path, candidate)
#             score = _run_eval(
#                 checkpoint_dir=args.checkpoint_dir,
#                 planner_path=planner_path,
#                 config=args.config,
#                 output_dir=candidate_dir / "eval",
#                 eval_seconds=float(args.eval_seconds),
#                 force_cpu=bool(args.force_cpu),
#             )
#             record = {"iteration": iteration, "candidate": candidate_idx, "score": score, "planner": candidate}
#             history.append(record)
#             if score > best_score:
#                 best_score = score
#                 best = dict(candidate)
#                 _write_json(output_dir / "best_planner_config.json", best)
#                 _write_json(output_dir / "best_score.json", {"score": best_score, "iteration": iteration, "candidate": candidate_idx})
#             print(f"iter={iteration} cand={candidate_idx} score={score:.3f} best={best_score:.3f}", flush=True)

#         _write_json(output_dir / "search_summary.json", {"best_score": best_score, "best_planner": best, "history": history})

#     print(json.dumps({"best_score": best_score, "best_planner_config": str(output_dir / "best_planner_config.json")}, indent=2))


# if __name__ == "__main__":
#     main()
