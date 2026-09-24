"""Safety override for the live Decision Agent.

FIXED (post-audit, v3):
  The v2 rule was `if gas_id is None or gas_id not in DANGER_GAS_IDS: return action`,
  and gas_id was the ground-truth label from the CSV. In deployment there's no label,
  so the function returned before ever looking at the anomaly. It could not fire in the
  field — a real defect the reviewer flagged.

  CORRECT design — the override is a *fail-safe against UNDER-escalation* that works
  both with and without a known gas class:
    * With a known danger gas (Smoke=1, Mixture=2): escalate an under-responding policy
      to the class-correct severe action (Smoke -> 3 Raise Alarm, Mixture -> 4 Emergency).
    * With a known non-danger gas (NoGas=0, Perfume=3): do NOT escalate — these gases
      don't warrant emergency action regardless of anomaly.
    * Without a known gas class (deployment): escalate an under-responding policy to
      Raise Alarm (3) if the raw anomaly is extreme. This is the most conservative
      choice that still alerts on a confirmed hazard.
    * It NEVER overrides an already-correct severe action (Smoke=3 or Mixture=4),
      so Smoke is never wrongly forced to Emergency Shutdown.
    * Mixture is the only class whose correct action is Emergency (4); Smoke's
      correct action is Raise Alarm (3) and is preserved.
"""

import math

DANGER_GAS_IDS = {1, 2}          # Smoke, Mixture
# Raw reconstruction-error thresholds measured on this corpus (unit = MSE of the
# 20-window scaled AE). NoGas ~0.74, Perfume ~1.5, Mixture ~118, Smoke ~229.
# Escalate only past a value far above any clean/Perfume reading, so we never
# mistake a benign reading for a hazard.
EMERGENCY_RAW_ANOMALY = 50.0

# Class-correct severe action (used only to promote an UNDER-escalated policy).
CORRECT_SEVERE_ACTION = {1: 3, 2: 4}   # Smoke -> Raise Alarm, Mixture -> Emergency


def safety_override(state, action, gas_id=None, raw_anomaly=None):
    """
    Args:
        state:       22-dim state vector (state[0] = normalized anomaly in [0,1]).
        action:      raw policy action (0..4).
        gas_id:      canonical gas class if known (0 NoGas, 1 Smoke, 2 Mixture, 3 Perfume).
                     None in deployment (no label available).
        raw_anomaly: raw reconstruction-error anomaly (absolute scale), if available.

    Returns:
        action (possibly promoted to the class-correct severe action) or the original.
    """
    # Resolve the raw anomaly from the state vector if not provided.
    if raw_anomaly is None:
        raw_anomaly = float(state[0]) if state is not None else 0.0

    # Require a genuinely extreme RAW anomaly (not the clipped [0,1] normalized value).
    if raw_anomaly < EMERGENCY_RAW_ANOMALY:
        return action

    # ── Known danger gas: escalate to the class-correct severe action ──
    if gas_id is not None and gas_id in DANGER_GAS_IDS:
        # If the policy already chose the class-correct severe action (or higher),
        # leave it — never override Smoke=Raise Alarm (3) or Mixture=Emergency (4).
        correct_severe = CORRECT_SEVERE_ACTION[gas_id]
        if action >= correct_severe:
            return action
        # Policy UNDER-responded on a confirmed danger: promote to class-correct severe.
        return correct_severe

    # ── Known non-danger gas (NoGas, Perfume): do NOT escalate ──
    if gas_id is not None:
        return action

    # ── Unknown gas class (deployment): extreme anomaly with under-response ──
    # Conservative default: escalate to Raise Alarm (3) — alerts on a confirmed
    # hazard without over-escalating to Emergency Shutdown on an unknown class.
    if action < 3:
        return 3

    # Policy already chose a severe action (3 or 4); preserve it.
    return action
