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


def montage_pos2d_clean(raw, decimals=7):
    montage = raw.get_montage()
    ch_pos = montage.get_positions().get("ch_pos", {}) if montage is not None else {}

    keep_idx, pos2d = [], []
    seen = set()

    dropped_no_pos, dropped_nan, dropped_overlap = [], [], []

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

    if pos2d.size == 0 or not np.all(np.isfinite(pos2d)):
        raise RuntimeError("No valid electrode positions remain after filtering.")

    return pos2d, keep_idx

def load_begin_csv(path):
    df = pd.read_csv(path)
    if "channel" not in df.columns or "impedance_ohm" not in df.columns:
        raise ValueError("CSV must have columns: channel, impedance_ohm")
    df = df.set_index("channel").sort_index()
    return df


def simulate_end_from_begin(begin_ohm, seed=21):
    rng = np.random.default_rng(seed)

    end = begin_ohm.copy().astype(float)

    end += rng.normal(loc=0, scale=350, size=end.shape)

    improve_idx = rng.choice(len(end), size=2, replace=False)
    end[improve_idx] -= rng.uniform(3000, 8000, size=2)

    worsen_idx = rng.choice(len(end), size=2, replace=False)
    end[worsen_idx] += rng.uniform(8000, 25000, size=2)

    end = np.clip(end, 200, None)

    return end


def save_topomap(fig_path, title, values, pos2d):
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        values,
        pos2d,
        axes=ax,
        show=False,
        contours=6,
        sensors=True,
        cmap="Reds",
    )
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Ohm (Ω)")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_delta_topomap(fig_path, title, delta_values, pos2d):
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        delta_values,
        pos2d,
        axes=ax,
        show=False,
        contours=6,
        sensors=True,
    )
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Ohm (Ω)")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "impedance"
    fig_dir = BASE_DIR / "figures"
    out_dir = BASE_DIR / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    begin_csv = data_dir / "impedance_begin_ohm.csv"
    if not begin_csv.exists():
        raise FileNotFoundError(f"Missing begin CSV: {begin_csv}")

    raw = apply_montage(clean_channel_names(load_eegbci_eeg(subject=1, run=3)))
    pos2d, keep_idx = montage_pos2d_clean(raw)

    df_begin = load_begin_csv(begin_csv)

    ch_names = raw.ch_names
    begin_all = np.array([df_begin.loc[ch, "impedance_ohm"] for ch in ch_names], dtype=float)

    end_all = simulate_end_from_begin(begin_all, seed=21)

    df_end = pd.DataFrame({"channel": ch_names, "impedance_ohm": end_all})
    end_csv = data_dir / "impedance_end_ohm.csv"
    df_end.to_csv(end_csv, index=False)
    print(f"Saved end CSV: {end_csv}")

    begin_viz = begin_all[keep_idx]
    end_viz = end_all[keep_idx]
    delta_viz = end_viz - begin_viz

    save_topomap(
        fig_dir / "impedance_topomap_end_ohm.png",
        "EEG Electrode Impedance (End) [Ohm]",
        end_viz,
        pos2d,
    )
    print(f"Saved figure: {fig_dir / 'impedance_topomap_end_ohm.png'}")

    save_delta_topomap(
        fig_dir / "impedance_topomap_delta_ohm.png",
        "EEG Electrode Impedance (End - Begin) [Δ Ohm]",
        delta_viz,
        pos2d,
    )
    print(f"Saved figure: {fig_dir / 'impedance_topomap_delta_ohm.png'}")

    THRESH_BAD = 20000.0  # Ohm
    THRESH_WARN = 10000.0

    qc = pd.DataFrame(
        {
            "channel": ch_names,
            "begin_ohm": begin_all,
            "end_ohm": end_all,
        }
    )
    qc["delta_ohm"] = qc["end_ohm"] - qc["begin_ohm"]
    qc["status"] = "OK"
    qc.loc[qc["end_ohm"] >= THRESH_WARN, "status"] = "WARN"
    qc.loc[qc["end_ohm"] >= THRESH_BAD, "status"] = "BAD"

    bad_channels = qc.loc[qc["status"] == "BAD", "channel"].tolist()
    print("Detected BAD channels (end >= 20000Ω):", bad_channels)

    qc_sorted = qc.sort_values(["status", "end_ohm"], ascending=[True, False])
    out_csv = out_dir / "qc_summary.csv"
    qc_sorted.to_csv(out_csv, index=False)
    print(f"Saved QC summary: {out_csv}")

    md_path = out_dir / "qc_summary.md"
    top_bad = qc_sorted[qc_sorted["status"] == "BAD"].head(10)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# EEG Impedance QC Summary\n\n")
        f.write(f"- Thresholds: WARN ≥ {THRESH_WARN:.0f}Ω, BAD ≥ {THRESH_BAD:.0f}Ω\n")
        f.write(f"- Number of BAD channels: {len(bad_channels)}\n\n")
        f.write("## Top BAD channels\n\n")
        if len(top_bad) == 0:
            f.write("No BAD channels detected.\n")
        else:
            f.write(top_bad.to_markdown(index=False))
            f.write("\n")
    print(f"Saved QC markdown: {md_path}")


if __name__ == "__main__":
    main()
