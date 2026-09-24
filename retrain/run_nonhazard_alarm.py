"""Alarm burden counted over every nonhazardous window, not only over clean air.

Reviewer comment 4.3, 14 September. The high-severity false-alarm rate in this
paper is defined on NoGas windows alone. Perfume is also nonhazardous under the
paper's own response map: its target action is 1, and the acceptable set is
{1, 2}, neither of which is alarm-grade. An operator who receives an alarm
because a nuisance odorant crossed the array receives an unnecessary alarm just
the same, and the manuscript itself keeps showing that the NoGas and Perfume
boundary is where the models are least certain. Restricting the denominator to
NoGas therefore reports the cleaner half of the nuisance problem.

This driver adds the missing quantity without retraining anything. The action
matrices in v2_action_matrix.csv already record, for every comparator and every
gas class, the share of windows assigned to each of the five actions, averaged
over the five seed-varying partitions. The alarm-grade share on a class is the
sum of its action-3 and action-4 shares, and the nonhazardous rate is the
window-weighted combination of NoGas and Perfume, which carry equal window counts
here (1,581 each, so 316 each per test partition).

Two rates are written per comparator:

    clean_air_alarm_rate      alarm-grade share on NoGas alone, the quantity the
                              manuscript has been reporting
    nonhazardous_alarm_rate   alarm-grade share on NoGas and Perfume together

Writes results_v2/v3_nonhazard_alarm.csv.

    python3 -m retrain.run_nonhazard_alarm
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain.run_all_v2 import OUT

ALARM_GRADE = 3
NONHAZ = ["NoGas", "Perfume"]


def main():
    am = pd.read_csv(os.path.join(OUT, "v2_action_matrix.csv"))
    alarm = am[am.action >= ALARM_GRADE]
    rows = []
    for model, g in am.groupby("model"):
        a = alarm[alarm.model == model]
        clean = float(a[a.gas == "NoGas"].share.sum())
        perf = float(a[a.gas == "Perfume"].share.sum())
        # equal window counts per class, so the combined rate is the mean
        nonhaz = (clean + perf) / 2.0
        rows.append(dict(model=model,
                         clean_air_alarm_rate=clean,
                         perfume_alarm_rate=perf,
                         nonhazardous_alarm_rate=nonhaz,
                         clean_air_per_1000=1000 * clean,
                         nonhazardous_per_1000=1000 * nonhaz))
    df = pd.DataFrame(rows).sort_values("nonhazardous_alarm_rate", ascending=False)
    df.to_csv(os.path.join(OUT, "v3_nonhazard_alarm.csv"), index=False)

    print(f"{'model':18s} {'NoGas only':>12s} {'Perfume':>10s} {'both':>10s}")
    for _, r in df.iterrows():
        print(f"{r['model']:18s} {r['clean_air_alarm_rate']:12.4f} "
              f"{r['perfume_alarm_rate']:10.4f} {r['nonhazardous_alarm_rate']:10.4f}")
    worst = df.iloc[0]
    print(f"\nlargest nonhazardous alarm-grade rate: {worst['model']} at "
          f"{worst['nonhazardous_alarm_rate']:.4f}, against "
          f"{worst['clean_air_alarm_rate']:.4f} on clean air alone")
    n_hidden = int(((df.nonhazardous_alarm_rate > 0) & (df.clean_air_alarm_rate == 0)).sum())
    print(f"comparators with no clean-air alarm but a nonzero nonhazardous rate: {n_hidden}")
    print("wrote v3_nonhazard_alarm.csv")


if __name__ == "__main__":
    main()
