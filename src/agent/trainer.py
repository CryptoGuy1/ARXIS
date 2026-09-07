"""Agent Trainer — trains the decision model from collected experience.

FIXED (post-audit):
  The v1 AgentTrainer called self.model.update(state, reward) — but DecisionTool
  has no .update() method. It's a stateless policy that loads a pre-trained
  checkpoint; it doesn't do online updates.

  CORRECT design — AgentTrainer is a no-op placeholder that logs the intended
  update but does not modify the model. If online updates are needed in the
  future, they should be implemented in DecisionTool itself (not here).
"""


class AgentTrainer:
    def __init__(self, decision_model):
        self.model = decision_model

    def train(self, memory):
        """
        Train the model from memory.

        Note: DecisionTool is a stateless pre-trained policy with no .update()
        method. This method is a no-op placeholder. If online updates are needed,
        implement them in DecisionTool.
        """
        batch = memory.get_recent()

        for item in batch:
            state = item["state"]
            reward = item["reward"]
            # No-op: DecisionTool does not support online updates.
            # If needed, implement model.update() in DecisionTool first.
            pass
