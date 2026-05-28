import numpy as np

from go2_pg_env.track import StandardOvalTrack
from track_bonus.scoring import compute_track_bonus_metrics, score_track_bonus


def _synthetic_rollout(track: StandardOvalTrack, progress_end: float, *, lateral: float = 0.0, fall_step: int | None = None):
    steps = 101
    qpos = np.zeros((steps, 19), dtype=np.float32)
    for idx, s in enumerate(np.linspace(0.0, progress_end, steps)):
        xy, _, _ = track.centerline_pose(float(s))
        qpos[idx, :2] = xy
        qpos[idx, 2] = 0.32
        if lateral:
            _, heading, _ = track.centerline_pose(float(s))
            normal = np.asarray([-np.sin(heading), np.cos(heading)])
            qpos[idx, :2] += lateral * normal
        qpos[idx, 3] = 1.0
    done = np.zeros(steps, dtype=bool)
    fall = np.zeros(steps, dtype=bool)
    if fall_step is not None:
        done[fall_step:] = True
        fall[fall_step:] = True
    return {
        "dt": 0.02,
        "qpos": qpos,
        "done": done,
        "fall": fall,
        "joint_torques": np.zeros((steps, 12), dtype=np.float32),
        "joint_velocities": np.zeros((steps, 12), dtype=np.float32),
        "foot_slip_speed": np.zeros((steps, 4), dtype=np.float32),
    }


def test_completed_rollout_scores_above_partial_rollout() -> None:
    track = StandardOvalTrack()
    complete = score_track_bonus(compute_track_bonus_metrics(_synthetic_rollout(track, 205.0), track))
    partial = score_track_bonus(compute_track_bonus_metrics(_synthetic_rollout(track, 60.0), track))
    assert complete["composite_score"] > partial["composite_score"]


def test_fall_and_boundary_violation_are_penalized() -> None:
    track = StandardOvalTrack()
    stable = score_track_bonus(compute_track_bonus_metrics(_synthetic_rollout(track, 120.0), track))
    fall = score_track_bonus(compute_track_bonus_metrics(_synthetic_rollout(track, 120.0, fall_step=30), track))
    outside = score_track_bonus(compute_track_bonus_metrics(_synthetic_rollout(track, 120.0, lateral=2.2), track))
    assert stable["composite_score"] > fall["composite_score"]
    assert stable["composite_score"] > outside["composite_score"]


def test_startline_projection_does_not_count_as_completed_lap() -> None:
    track = StandardOvalTrack()
    rollout = _synthetic_rollout(track, 2.0)
    metrics = compute_track_bonus_metrics(rollout, track)
    assert 0.005 < metrics["lap_completion"] < 0.02
    assert metrics["finish_time"] is None
