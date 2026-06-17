"""Game-count curricula for the deep-reflection (moonshot) run.

All schedules are piecewise by total self-play games. The learner writes the
current game count to a progress file; workers poll it (like the gate/
checkpoint files) so learner and workers compute identical configs.

Rationale (from the project's findings):
- Q is noise early -> search should not trust q_init until the Q-head matures
  (q_trust ramp), and the Q-loss/consistency weights start low.
- Study fires on value noise too early -> study stays off until the gate,
  then ramps intensity (deeper re-search, more lines) as the net gets good
  enough for deep reflection to actually pay off.
"""

from dataclasses import replace

from .study import StudyConfig
from .train import LossWeights

INF = 10**9


def _stage(games, stages):
    """stages: [(upper_threshold, value), ...] sorted ascending; returns the
    value for the first threshold the game count is below."""
    for thr, val in stages:
        if games < thr:
            return val
    return stages[-1][1]


def q_trust_at(games: int) -> float:
    """How much search trusts the Q-head for unvisited children."""
    return _stage(
        games,
        [(2000, 0.1), (10000, 0.25), (40000, 0.5), (80000, 0.75), (INF, 1.0)],
    )


def study_config_at(games: int, base: StudyConfig) -> StudyConfig:
    """Ramp study intensity: (top_k, deep_sims, branch_plies, n_lines)."""
    tk, ds, bp, nl = _stage(
        games,
        [
            (10000, (4, 256, 24, 2)),
            (40000, (6, 384, 32, 3)),
            (80000, (8, 512, 40, 3)),
            (INF, (8, 768, 48, 4)),
        ],
    )
    return replace(base, top_k=tk, deep_sims=ds, branch_plies=bp, n_lines=nl)


def loss_weights_at(games: int) -> LossWeights:
    """Ramp Q and consistency loss up as the Q-head becomes trustworthy."""
    q, c = _stage(
        games,
        [(2000, (0.25, 0.1)), (10000, (0.5, 0.25)), (INF, (1.0, 0.6))],
    )
    return LossWeights(policy=1.0, value=1.0, q=q, consistency=c, wdl=0.5)
