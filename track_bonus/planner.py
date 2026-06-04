"""High-level planner for the 200 m track bonus.

    5D track observation -> [vx, vy, yaw_rate]

Two planner types share one load/command entry point so the evaluator contract
never changes:

  * ``starter_pd``   -- the original hand-written PD baseline (unchanged).
  * ``learned_mlp``  -- a small trained MLP (this is the leaderboard planner).

The evaluator always does ``StarterTrackPlanner.load(planner_config)`` then
``planner.command(track_observation, t)``.  Keep those two signatures intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


# ---------------------------------------------------------------------------
# Shared MLP math.  The trainer imports these so the network used during
# optimization is byte-for-byte the network used at evaluation time.
# ---------------------------------------------------------------------------

N_IN = 5
N_OUT = 3


def num_mlp_params(hidden_size: int, n_in: int = N_IN, n_out: int = N_OUT) -> int:
    return hidden_size * n_in + hidden_size + n_out * hidden_size + n_out


def unpack_weights(theta: np.ndarray, hidden_size: int, n_in: int = N_IN, n_out: int = N_OUT):
    theta = np.asarray(theta, dtype=np.float64).ravel()
    i = 0
    w1 = theta[i:i + hidden_size * n_in].reshape(hidden_size, n_in); i += hidden_size * n_in
    b1 = theta[i:i + hidden_size]; i += hidden_size
    w2 = theta[i:i + n_out * hidden_size].reshape(n_out, hidden_size); i += n_out * hidden_size
    b2 = theta[i:i + n_out]; i += n_out
    return w1, b1, w2, b2


def mlp_raw(obs5: np.ndarray, theta: np.ndarray, hidden_size: int) -> np.ndarray:
    """Raw (unbounded) 3D network output for a single observation."""
    w1, b1, w2, b2 = unpack_weights(theta, hidden_size)
    x = np.asarray(obs5, dtype=np.float64).ravel()
    h = np.tanh(w1 @ x + b1)
    return w2 @ h + b2


def _sigmoid(z: float) -> float:
    # numerically stable
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def bound_command(
    raw: np.ndarray,
    *,
    vx_min: float,
    vx_max: float,
    vy_max: float,
    yaw_rate_max: float,
) -> np.ndarray:
    """Map raw network outputs into the low-level's trainable command ranges.

    vx is one-directional  -> sigmoid into [vx_min, vx_max]
    vy, yaw are zero-centered -> tanh into [-max, +max]
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()
    vx = vx_min + _sigmoid(float(raw[0])) * (vx_max - vx_min)
    vy = vy_max * math.tanh(float(raw[1]))
    yaw = yaw_rate_max * math.tanh(float(raw[2]))
    return np.asarray([vx, vy, yaw], dtype=np.float32)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"

    # starter_pd knobs
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0

    # learned_mlp knobs
    weights_path: str = ""
    hidden_size: int = 16
    vx_min: float = 0.5
    vx_max: float = 2.5
    vy_max: float = 0.3
    yaw_rate_max: float = 0.4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
            "weights_path": self.weights_path,
            "hidden_size": self.hidden_size,
            "vx_min": self.vx_min,
            "vx_max": self.vx_max,
            "vy_max": self.vy_max,
            "yaw_rate_max": self.yaw_rate_max,
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class StarterTrackPlanner:
    """High-level controller. Supports the starter PD law and a learned MLP."""

    SUPPORTED = ("starter_pd", "learned_mlp")

    def __init__(self, config: StarterPlannerConfig, weights: np.ndarray | None = None) -> None:
        if config.planner_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._theta = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
        if config.planner_type == "learned_mlp":
            expected = num_mlp_params(int(config.hidden_size))
            if self._theta is None:
                raise ValueError(
                    "learned_mlp planner requires weights; set 'weights_path' in the planner config."
                )
            if self._theta.size != expected:
                raise ValueError(
                    f"weights have {self._theta.size} params but hidden_size={config.hidden_size} "
                    f"expects {expected}. Did hidden_size change after training?"
                )

    # -- entry points the evaluator calls -----------------------------------

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        path = Path(path)
        config = StarterPlannerConfig.load(path)
        weights = None
        if config.planner_type == "learned_mlp":
            weights = cls._load_weights(path, config)
        return cls(config, weights=weights)

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        if self.config.planner_type == "learned_mlp":
            return self._learned_command(obs)
        return self.command_from_observation(obs)

    # -- weights ------------------------------------------------------------

    @staticmethod
    def _load_weights(config_path: Path, config: StarterPlannerConfig) -> np.ndarray:
        if not config.weights_path:
            raise ValueError("learned_mlp planner config is missing 'weights_path'.")
        wpath = Path(config.weights_path)
        if not wpath.is_absolute():
            wpath = config_path.parent / wpath  # resolve relative to the config file
        data = np.load(wpath)
        theta = np.asarray(data["theta"], dtype=np.float64).ravel()
        if "hidden_size" in data and int(data["hidden_size"]) != int(config.hidden_size):
            raise ValueError(
                f"weights hidden_size={int(data['hidden_size'])} != config hidden_size={config.hidden_size}."
            )
        return theta

    def set_weights(self, theta: np.ndarray) -> None:
        """Used by the trainer to swap candidate weights without reloading."""
        self._theta = np.asarray(theta, dtype=np.float64).ravel()

    # -- learned planner ----------------------------------------------------

    def _learned_command(self, obs: TrackControllerObservation) -> np.ndarray:
        raw = mlp_raw(obs.as_array(), self._theta, int(self.config.hidden_size))
        return bound_command(
            raw,
            vx_min=float(self.config.vx_min),
            vx_max=float(self.config.vx_max),
            vy_max=float(self.config.vy_max),
            yaw_rate_max=float(self.config.yaw_rate_max),
        )

    # -- original starter PD law (unchanged) --------------------------------

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)






# """Starter high-level planner for the 200 m track bonus.

# The evaluator builds the official compact 5D track observation defined in
# `track_bonus/controller_interface.py`. The high-level planner maps it to the
# local joystick command consumed by the HW1 Go2 locomotion policy:

#     5D track observation -> [vx, vy, yaw_rate]

# This file is intentionally small.  It is a weak baseline and an interface
# example, not a solved full-lap controller.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# import json
# import math
# from pathlib import Path
# from typing import Any

# import numpy as np

# from go2_pg_env.track import StandardOvalTrack, wrap_angle
# from track_bonus.controller_interface import TrackControllerObservation
# from track_bonus.official_track import official_track


# @dataclass(frozen=True)
# class StarterPlannerConfig:
#     planner_type: str = "starter_pd"
#     speed_mps: float = 0.45
#     min_speed_mps: float = 0.12
#     max_lateral_speed_mps: float = 0.08
#     max_yaw_rate_radps: float = 0.25
#     k_heading: float = 0.55
#     k_lateral: float = 0.08
#     heading_slowdown: float = 0.45
#     stand_seconds: float = 1.0

#     @classmethod
#     def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
#         valid = set(cls.__dataclass_fields__.keys())
#         values = {key: payload[key] for key in valid if key in payload}
#         return cls(**values)

#     @classmethod
#     def load(cls, path: Path) -> "StarterPlannerConfig":
#         return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "planner_type": self.planner_type,
#             "speed_mps": self.speed_mps,
#             "min_speed_mps": self.min_speed_mps,
#             "max_lateral_speed_mps": self.max_lateral_speed_mps,
#             "max_yaw_rate_radps": self.max_yaw_rate_radps,
#             "k_heading": self.k_heading,
#             "k_lateral": self.k_lateral,
#             "heading_slowdown": self.heading_slowdown,
#             "stand_seconds": self.stand_seconds,
#         }


# class StarterTrackPlanner:
#     """Conservative coordinate-to-command baseline.

#     The policy is deliberately simple and conservative. Students should improve
#     it by changing this controller, replacing it with an MLP, or training a
#     higher-level policy that produces the same command vector.
#     """

#     def __init__(self, config: StarterPlannerConfig) -> None:
#         if config.planner_type != "starter_pd":
#             raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
#         self.config = config
#         self.track: StandardOvalTrack = official_track()

#     @classmethod
#     def load(cls, path: Path) -> "StarterTrackPlanner":
#         return cls(StarterPlannerConfig.load(path))

#     def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
#         if t < self.config.stand_seconds:
#             return np.zeros(3, dtype=np.float32)
#         return self.command_from_observation(obs)





#     # 5D Input (Observation) -> Tells robot what to do in the form of 3D Output
#     # 5D Input: [lap_fraction, lateral_error_norm, boundary_margin_norm, heading_error_rad, curvature_norm]
#     # 3D Output: [vx_mps, vy_mps, yaw_rate_radps]
#     def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
#         lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
#         lateral_bias = math.atan2(
#             float(self.config.k_lateral) * lateral_error,
#             max(float(self.config.speed_mps), 1e-3),
#         )
#         heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

#         speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
#         vx = np.clip(
#             float(self.config.speed_mps) * speed_scale,
#             float(self.config.min_speed_mps),
#             float(self.config.speed_mps),
#         )
#         vy = np.clip(
#             -float(self.config.k_lateral) * lateral_error,
#             -float(self.config.max_lateral_speed_mps),
#             float(self.config.max_lateral_speed_mps),
#         )
#         curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
#         yaw_rate = np.clip(
#             curvature * vx + float(self.config.k_heading) * heading_error,
#             -float(self.config.max_yaw_rate_radps),
#             float(self.config.max_yaw_rate_radps),
#         )
#         return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
