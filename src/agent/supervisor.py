"""Supervisor Agent — oversees the worker agent and applies additional safety checks.

FIXED (post-audit):
  The v1 SupervisorAgent called run_once with four keywords that don't exist
  (image, sensor_array, temp, hum). The actual run_once signature is
  (sensor_row, step, gas_id, image_path, use_mc_dropout, enable_explanations, enable_critique).

  CORRECT design — the supervisor wraps the worker's run_once with the correct
  keyword arguments and applies an additional safety layer on top.
"""


class SupervisorAgent:
    def __init__(self, worker_agent):
        self.worker = worker_agent

    def run_cycle(self, sensor_row, gas_id, step, image_path=None, use_mc_dropout=False):
        """
        Run one cycle of the worker agent with supervisor oversight.

        Args:
            sensor_row:     array of 7 sensor readings.
            gas_id:         canonical gas class (0..3).
            step:           current step index.
            image_path:     path to thermal image (optional).
            use_mc_dropout: whether to use MC dropout for uncertainty.
        """
        result = self.worker.run_once(
            sensor_row=sensor_row,
            step=step,
            gas_id=gas_id,
            image_path=image_path,
            use_mc_dropout=use_mc_dropout,
            enable_explanations=False,
            enable_critique=False,
        )

        # Supervisor oversight — additional safety escalation
        if result.get("state") is not None:
            anomaly = result["state"][0]
            action = result["action"]

            # If anomaly is high but action is mild, escalate to Raise Alarm
            if anomaly > 0.7 and action < 3:
                print("SUPERVISOR OVERRIDE: forcing alarm")
                result["action"] = 3
                result["supervisor_override"] = True
            else:
                result["supervisor_override"] = False

        return result
