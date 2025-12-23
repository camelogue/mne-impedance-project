from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf


def load_eegbci_eeg(subject=1, run=3):
    edf_path = eegbci.load_data(subject, [run])[0]
    raw = read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick(picks="eeg")
    return raw


def clean_channel_names(raw):
    mapping = {ch: ch.strip().rstrip(".") for ch in raw.ch_names}
    raw.rename_channels(mapping)
    return raw


def apply_montage(raw):
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def simulate_base_impedance_ohm(n_channels, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=7000, scale=2500, size=n_channels)
    base = np.clip(base, 800, 20000)
    return base.astype(float)


def montage_pos2d_clean(raw, decimals=7):
    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError("No montage found on raw. Did you call raw.set_montage()?")

    ch_pos = montage.get_positions().get("ch_pos", {})
    if not ch_pos:
        raise RuntimeError("Montage has no channel positions (ch_pos).")

    keep_idx = []
    pos2d = []
    seen = set()

    dropped_no_pos = []
    dropped_nan = []
    dropped_overlap = []

    for i, ch in enumerate(raw.ch_names):
        if ch not in ch_pos:
            dropped_no_pos.append(ch)
            continue

        x, y, z = ch_pos[ch]
        xy = np.array([x, y], dtype=float)

        if not np.all(np.isfinite(xy)):
            dropped_nan.append(ch)
            continue

        key = (round(float(xy[0]), decimals), round(float(xy[1]), decimals))
        if key in seen:
            dropped_overlap.append(ch)
            continue

        seen.add(key)
        keep_idx.append(i)
        pos2d.append([xy[0], xy[1]])

    if dropped_no_pos:
        print("Dropped (no position):", dropped_no_pos)
    if dropped_nan:
        print("Dropped (NaN/inf position):", dropped_nan)
    if dropped_overlap:
        print("Dropped (overlapping position):", dropped_overlap)

    pos2d = np.asarray(pos2d, dtype=float)
    keep_idx = np.asarray(keep_idx, dtype=int)

    if pos2d.size == 0:
        raise RuntimeError("No valid electrode positions remain after filtering.")
    if not np.all(np.isfinite(pos2d)):
        raise RuntimeError("pos2d still contains NaN/inf after filtering (unexpected).")

    return pos2d, keep_idx


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "impedance"
    fig_dir = BASE_DIR / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_eegbci_eeg(subject=1, run=3)
    raw = clean_channel_names(raw)
    raw = apply_montage(raw)

    ch_names = raw.ch_names

    imp_ohm = simulate_base_impedance_ohm(len(ch_names), seed=7)

    pos2d, keep_idx = montage_pos2d_clean(raw)

    rng = np.random.default_rng(7)
    bad_local = rng.choice(keep_idx, size=3, replace=False)
    bad_channels = [ch_names[i] for i in bad_local]
    imp_ohm[bad_local] = rng.uniform(25000, 60000, size=3)

    print("Simulated bad channels (visible on topomap):", bad_channels)

    out_csv = data_dir / "impedance_begin_ohm.csv"
    pd.DataFrame({"channel": ch_names, "impedance_ohm": imp_ohm}).to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    data_viz = imp_ohm[keep_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        data_viz,
        pos2d,
        axes=ax,
        show=False,
        contours=6,
        sensors=True,
        cmap="Reds",
    )
    ax.set_title("EEG Electrode Impedance (Begin) [Ohm]")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Impedance (Ω)")

    out_fig = fig_dir / "impedance_topomap_begin_ohm.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
