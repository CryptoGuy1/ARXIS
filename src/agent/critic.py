"""Critic Agent — audits decisions after they are made.

FIXED (post-audit):
  The v1 CriticAgent read state[1] as gas_id, state[7] as temperature, and
  state[8] as humidity — but the state vector is [anomaly | 7 current | 7 delta | 7 std],
  so those slots were actually MQ2, MQ135, and dMQ2. The "temperature" and "humidity"
  it fed to the language model were sensor readings mislabeled as °C and %.

  It was also called with one argument against a three-argument signature.

  CORRECT design — the critic receives the same structured context as the explanation
  layer (state, action, q_values, gas_id, anomaly). It does NOT unpack the state
  vector directly; the caller passes the already-resolved values.
"""


class CriticAgent:
    def __init__(self, explanation_tool):
        self.explainer = explanation_tool

    def critique(self, state, action, q_values, gas_id=None, anomaly=None):
        """
        Audit the safety of a decision.

        Args:
            state:     22-dim state vector (state[0] = normalized anomaly).
            action:    raw policy action (0..4).
            q_values:  array of 5 Q-values from the DQN.
            gas_id:    canonical gas class if known.
            anomaly:   raw reconstruction-error anomaly.
        """
        # Parse state vector correctly — NOT by unpacking into named variables.
        norm_anomaly = float(state[0]) if state is not None else 0.0
        raw_anom = anomaly if anomaly is not None else norm_anomaly

        # Resolve gas class label
        gas_label = {0: "NoGas", 1: "Smoke", 2: "Mixture", 3: "Perfume"}.get(gas_id, "Unknown")

        prompt = f"""
You are a safety auditor reviewing an industrial methane monitoring AI.

System readings:
  Normalized anomaly: {norm_anomaly:.4f}
  Raw anomaly (recon error): {raw_anom:.2f}
  Gas class: {gas_label}

RL policy output:
  Action chosen: {action} ({['Monitor', 'Increase Sampling', 'Request Verification', 'Raise Alarm', 'Emergency Shutdown'][action] if 0 <= action <= 4 else 'Unknown'})
  Q-values: {q_values}

Action meanings:
  0 = Monitor
  1 = Increase sampling
  2 = Request verification
  3 = Raise alarm
  4 = Emergency shutdown

Your task:
  1. Determine if the action is SAFE given the anomaly and gas class.
  2. Determine if the action matches the anomaly level.
  3. Suggest a safer action if needed.

Return format:
  Safety Verdict:
  Recommended Action:
  Reasoning:
"""

        critique = self.explainer.explain(prompt)
        return critique
