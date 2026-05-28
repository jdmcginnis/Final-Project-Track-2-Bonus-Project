# Go2 Track 2 Bonus Project Starter

This repository is the starter baseline for the final-project track bonus. It
extends the HW1 Go2 MuJoCo Playground assignment with a 200 m oval track
evaluation, while keeping the low-level locomotion pipeline aligned with the
original homework.

The intended controller is hierarchical:

```text
high-level planner:
  track coordinates -> [vx, vy, yaw_rate]

low-level policy:
  proprioception + command -> 12 Go2 joint actions
```

The starter high-level planner is intentionally weak. It is only an interface
example so students can run the benchmark and then improve it.

## Important

This repo does not include a trained solution checkpoint, a successful planner,
teacher rollouts, rendered solution videos, or tuned full-lap artifacts.
Students should reuse their HW1 checkpoint or train a new low-level policy.

## Colab Workflow

In Colab, set:

```python
COURSE_REPO_URL = "https://github.com/WeijieLai1024/Final-Project-Track-2-Bonus-Project.git"
COURSE_REPO_BRANCH = "main"
COURSE_REPO_DIR = Path("/content/go2_track_bonus_repo")
```

Then follow the notebook:

```text
notebooks/track_bonus_colab_template.ipynb
```

The notebook clones this repo, installs MuJoCo Playground dependencies, copies
Go2 assets, optionally trains a low-level policy, runs the starter track
evaluation, and renders a video.

Read the full assignment requirements before changing code:

```text
docs/assignment_requirements.md
```

## Low-Level Training

The low-level policy uses the same Brax PPO checkpoint format as HW1:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

The default baseline remains simple. A good project should improve command
tracking for curved running, especially yaw-rate commands.

## Starter Track Evaluation

Run the track bonus evaluation with a low-level checkpoint:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

For a quick non-rendered check:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval_smoke \
  --duration-seconds 5 \
  --no-render
```

Outputs:

- `results.json`
- `leaderboard.csv`
- `race_rollouts.npz`
- `race.mp4` unless `--no-render` is used

## Optional High-Level Search

The included high-level trainer is a small black-box parameter search:

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This is a scaffold, not the expected final method. You may replace it with an
MLP, RL controller, residual controller, or another learned high-level policy,
as long as the runtime outputs `[vx, vy, yaw_rate]`.

## Track Metrics

The benchmark reports:

- `lap_completion`
- `finish_time`
- `mean_progress_speed`
- `rms_lateral_error`
- `max_lateral_error`
- `min_boundary_margin_m`
- `fall`
- `boundary_violation`
- `energy_proxy`
- `foot_slip_proxy`

## Recommended Files To Read

```text
go2_pg_env/joystick.py
go2_pg_env/track.py
track_bonus/planner.py
run_track_bonus.py
train_highlevel_starter.py
docs/assignment_requirements.md
```

## Submission

Submit:

- `best_checkpoint/`
- `planner_config.json` or your planner file
- `submission.json`
- `track_eval/results.json`
- optional `track_eval/race.mp4`
- a short report explaining the low-level policy, high-level planner, training
  method, final metrics, and at least one failed idea.

## Student Modification Boundary

Students should mostly modify:

- `go2_pg_env/joystick.py`
- `configs/course_config.json`
- `track_bonus/planner.py`
- `train_highlevel_starter.py` or a new high-level training script
- planner config files

Students should usually not modify:

- metric names
- checkpoint restore logic
- rollout bundle field names
- renderer-only code
