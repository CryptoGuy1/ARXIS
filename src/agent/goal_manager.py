"""Goal Manager — defines system safety goals.

FIXED (post-audit):
  The v1 GoalManager unpacked the 22-dim state vector into 5 named variables
  (anomaly, gas_class, conf, temp, hum) — but the state is [anomaly | 7 current |
  7 delta | 7 std], so those names were wrong. `gas_class` was actually MQ2,
  `conf` was actually MQ3, etc.

  CORRECT design — the goal manager only needs the normalized anomaly (state[0])
  to evaluate risk. It does NOT unpack the full state vector.
"""


class GoalManager:
    def __init__(self):
        self.goal = "Maintain industrial safety while minimizing false shutdowns"

    def evaluate_risk(self, state):
        """
        Evaluate risk from the state vector.

        Args:
            state: 22-dim state vector (state[0] = normalized anomaly in [0,1]).

        Returns:
            "HIGH_RISK" | "MEDIUM_RISK" | "LOW_RISK"
        """
        # Only use the normalized anomaly (state[0]) — not the full 22-dim vector.
        anomaly = float(state[0]) if state is not None and len(state) > 0 else 0.0

        if anomaly > 0.85:
            return "HIGH_RISK"
        elif anomaly > 0.6:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"

    def get_goal(self):
        return self.goal
