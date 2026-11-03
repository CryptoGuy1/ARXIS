"""Calibration band carved from a raw training frame, before anything is fitted.

Two analyses read model confidence off a held-in band of the training windows:
the temperature-scaling comparison (run_calib_actions_v2) and the reject option
(run_reject_option). Both previously fitted the input scaler, the anomaly
autoencoder and the anomaly normalization on the whole training partition, band
included, so the band was in-sample for the upstream pipeline although it was
out-of-sample for the classifier. Carving the band first, and fitting every
upstream quantity on the fitting windows alone, removes that overlap.

The band is the last `frac` of each class's training windows in acquisition
order. The `gap` windows before it (in training-window order) are discarded, so
no fitting window shares a raw reading with a band window: windows whose start
positions differ by at least 20 share no reading. When the trailing training
segment of a class is shorter than the band, the band spans the removed test
block and is contiguous in training-window order but not in time; the discarded
windows still separate it from every fitting window.
"""
from retrain import raw_pipeline as rp


def carve_calibration_band(df, frac=0.2, gap=None):
    """Return (fit_df, cal_df), both copies of rows of `df`, which must hold the
    columns gas_id and win_id and must not yet have been scaled or rescored."""
    gap = rp.WINDOW_SIZE if gap is None else gap
    fit_ids, cal_ids = [], []
    for _, g in df.groupby("gas_id"):
        g = g.sort_values("win_id")
        n_cal = max(1, int(round(frac * len(g))))
        cut = len(g) - n_cal
        g_gap = min(gap, max(0, cut))
        fit_ids.extend(g.index[:max(0, cut - g_gap)])
        cal_ids.extend(g.index[cut:])
    return df.loc[fit_ids].copy(), df.loc[cal_ids].copy()
