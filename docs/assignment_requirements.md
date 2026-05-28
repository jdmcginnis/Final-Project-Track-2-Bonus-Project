# Track 2 Bonus Project Requirements: Go2 200 m Oval

## 1. Project Goal

Your goal is to make Unitree Go2 run as far and as cleanly as possible around a
200 m standard oval track in MuJoCo.

This starter repo uses a hierarchical design:

```text
high-level planner:
  track coordinates -> local command [vx, vy, yaw_rate]

low-level locomotion policy:
  proprioception + command -> 12 joint position target offsets
```

The low-level policy is the same Brax PPO checkpoint format used in HW1. The
new part of this bonus project is the track-aware high-level control problem:
using track coordinates and geometry to decide how the robot should move around
the oval without leaving the lane.

## 2. What Is Provided

You are given:

- HW1-style Go2 locomotion training code.
- A 200 m oval track geometry module.
- A weak starter high-level planner in `configs/starter_planner.json`.
- `run_track_bonus.py`, which evaluates one low-level checkpoint plus one
  high-level planner on the oval.
- `train_highlevel_starter.py`, a small parameter-search scaffold.
- A Colab notebook that installs dependencies, clones this repo, trains or
  restores a checkpoint, runs evaluation, and renders a video.

The starter planner is intentionally conservative and is not expected to solve
the full project by itself.

## 3. Required Student Work

You must improve at least one of the following:

- Low-level command tracking: improve the PPO locomotion policy so it can track
  forward, lateral, and yaw-rate commands more robustly.
- High-level planning: replace or tune the starter planner so it uses track
  coordinates to keep the robot inside the oval lane.
- Training curriculum: design a curriculum that makes the low-level and
  high-level parts work together better.

The final controller must run through the provided evaluation command and must
output the same required artifacts.

## 4. Allowed Approaches

Allowed:

- Reusing your HW1 low-level checkpoint.
- Training a new low-level Brax PPO checkpoint from this repo.
- Editing `go2_pg_env/joystick.py` and `configs/course_config.json`.
- Replacing `track_bonus/planner.py` with your own high-level controller.
- Training a high-level MLP, RL policy, residual controller, black-box searched
  controller, or hand-designed baseline.
- Using global track features in the high-level controller.
- Adding new config files, scripts, or planner modules.

Not allowed:

- Changing benchmark metric names or deleting required output fields.
- Hard-coding benchmark results instead of running the policy.
- Submitting evaluator-only hacks that bypass the low-level policy.
- Using privileged simulator information inside the low-level actor beyond the
  normal `state` observation.
- Submitting generated artifacts without code/configs that explain how they
  were produced.

## 5. Observation And Action Design

The low-level actor should remain compatible with HW1:

```text
state =
  local linear velocity
  gyro
  gravity vector
  joint position error
  joint velocity
  last action
  command [vx, vy, yaw_rate]
```

The high-level planner may use track-coordinate features such as:

- robot global `x, y`
- progress along the centerline
- lap fraction
- lateral error from centerline
- distance to boundary
- heading error relative to track tangent
- local curvature
- lookahead heading or curvature

The high-level output must be:

```text
[vx, vy, yaw_rate]
```

in the same command convention used by the low-level joystick policy.

## 6. Starter Commands

Low-level training:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

Track evaluation:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

Quick no-render evaluation:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval_smoke \
  --duration-seconds 5 \
  --no-render
```

Optional high-level parameter search:

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

## 7. Evaluation Metrics

The evaluator reports:

- `lap_completion`: fraction of one 200 m lap completed.
- `finish_time`: time to complete one lap, or `null` if not completed.
- `mean_progress_speed`: average speed along the centerline.
- `rms_lateral_error`: RMS distance from the centerline.
- `max_lateral_error`: maximum absolute distance from the centerline.
- `min_boundary_margin_m`: closest distance to the lane boundary.
- `fall`: whether the robot fell before finishing.
- `boundary_violation`: whether the robot left the lane.
- `energy_proxy`: mean actuator power proxy.
- `foot_slip_proxy`: mean foot slip speed proxy.

The score rewards lap completion, speed, line keeping, stability, and
efficiency. Completing more of the lap is the most important objective.

## 8. Deliverables

Submit a folder or zip containing:

- `best_checkpoint/`
- `planner_config.json` or your planner file
- `submission.json`
- `track_eval/results.json`
- optional `track_eval/race.mp4`
- `short_report.pdf`

Your `submission.json` should include:

```json
{
  "team_name": "your_team_name",
  "checkpoint_dir": "best_checkpoint",
  "planner": "planner_config.json",
  "training_steps": 15000000,
  "notes": "Brief summary of your method"
}
```

## 9. Short Report Requirements

Your report should answer:

1. What does your low-level policy observe and output?
2. What did you change in the low-level training, reward, or command
   curriculum?
3. What track features does your high-level planner use?
4. Is your high-level controller hand-designed, searched, supervised, or RL
   trained?
5. How does your controller avoid leaving the track?
6. What are your final `lap_completion`, `finish_time`, lateral error, fall,
   energy, and slip metrics?
7. What failed idea did you try, and what did you learn?
8. What extra localization would be needed for a real robot to use this track
   coordinate controller?

## 10. Suggested Development Path

1. Run the notebook setup and inspect the starter files.
2. Reuse a HW1 checkpoint or train the default low-level baseline.
3. Run `run_track_bonus.py --no-render` for a short smoke test.
4. Render one evaluation video and inspect where the robot fails.
5. Improve yaw-rate and curved-command tracking in the low-level policy.
6. Tune or replace the high-level planner.
7. Repeat evaluation and compare metrics.
8. Save final artifacts and write the report.

## 11. Grading Rubric

Suggested grading:

- 35% lap performance: lap completion, finish time, and progress speed.
- 20% track safety: boundary margin, lateral error, and avoiding lane exit.
- 15% locomotion stability: no fall, smooth gait, robust turning.
- 10% efficiency: energy and foot slip proxies.
- 10% method quality: clear control design, justified training choices.
- 10% reproducibility: submitted configs, checkpoint, planner, results, and
  report can be rerun.

The bonus project is not graded only by fastest speed. A slower policy that
stays upright and inside the track can be better than a brief sprint that falls
or exits the lane.

## 12. Reproducibility Expectations

Your submission should be reproducible from the provided files. Include the
config used for training and planner evaluation when it differs from the
starter config. Do not rely on Colab temporary files that are not included in
your submission.

Recommended final verification:

```bash
python run_track_bonus.py \
  --checkpoint-dir best_checkpoint \
  --planner-config planner_config.json \
  --output-dir track_eval
```

Then confirm that `track_eval/results.json` exists and contains all required
metrics.
