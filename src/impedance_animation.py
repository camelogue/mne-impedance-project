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
    ch_pos = montage.get_positions().get("ch_pos", {}) if montage else {}

    keep_idx, pos2d = [], []
    seen = set()

    dropped = []

    for i, ch in enumerate(raw.ch_names):
        if ch not in ch_pos:
            dropped.append(ch)
            continue

        x, y, _ = ch_pos[ch]
        if not np.isfinite(x) or not np.isfinite(y):
            dropped.append(ch)
            continue

        key = (round(float(x), decimals), round(float(y), decimals))
        if key in seen:
            dropped.append(ch)
            continue

        seen.add(key)
        keep_idx.append(i)
        pos2d.append([x, y])

    if dropped:
        print(f"Dropped {len(dropped)} channels for visualization")

    return np.asarray(pos2d), np.asarray(keep_idx)


def load_begin_impedance(csv_path, ch_names):
    df = pd.read_csv(csv_path).set_index("channel")
    return np.array([df.loc[ch, "impedance_ohm"] for ch in ch_names], dtype=float)


def simulate_impedance_over_time(
    begin_all,
    n_frames=60,
    dt_sec=5,
    seed=42,
    drift_sigma=250.0,
):
    rng = np.random.default_rng(seed)
    cur = begin_all.copy()
    series = np.zeros((n_frames, len(cur)))

    improve_idx = rng.choice(len(cur), size=3, replace=False)
    worsen_idx = rng.choice(len(cur), size=3, replace=False)

    for t in range(n_frames):
        cur += rng.normal(0, drift_sigma, size=len(cur))

        if t == 10:   # 50. saniye
            cur[improve_idx] -= rng.uniform(5000, 12000, size=len(improve_idx))
        if t == 30:   # 150. saniye
            cur[worsen_idx] += rng.uniform(10000, 25000, size=len(worsen_idx))
        if t == 45:   # 225. saniye
            cur[improve_idx] -= rng.uniform(2000, 6000, size=len(improve_idx))

        cur = np.clip(cur, 200, None)
        series[t] = cur

    return series


def render_frame(values_viz, pos2d, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        values_viz,
        pos2d,
        axes=ax,
        show=False,
        contours=6,
        sensors=True,
        cmap="Reds",
    )
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Impedance (Ω)")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_paths, gif_path, fps=6):
    import imageio.v2 as imageio
    duration = 1.0 / fps
    with imageio.get_writer(gif_path, mode="I", duration=duration) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    data_dir = BASE_DIR / "data" / "impedance"
    fig_dir = BASE_DIR / "figures"
    frame_dir = fig_dir / "animation_frames"

    frame_dir.mkdir(parents=True, exist_ok=True)

    begin_csv = data_dir / "impedance_begin_ohm.csv"
    if not begin_csv.exists():
        raise FileNotFoundError("Begin impedance CSV not found")

    raw = apply_montage(clean_channel_names(load_eegbci_eeg()))
    pos2d, keep_idx = montage_pos2d_clean(raw)

    ch_names = raw.ch_names
    begin_all = load_begin_impedance(begin_csv, ch_names)

    n_frames = 60
    dt_sec = 5    
    fps = 6

    series = simulate_impedance_over_time(
        begin_all,
        n_frames=n_frames,
        dt_sec=dt_sec,
    )

    frame_paths = []
    for t in range(n_frames):
        time_sec = t * dt_sec
        values_viz = series[t][keep_idx]
        frame_path = frame_dir / f"frame_{t:03d}.png"
        title = f"EEG Impedance QC | t = {time_sec:03d} s"
        render_frame(values_viz, pos2d, title, frame_path)
        frame_paths.append(frame_path)

    print(f"Rendered {len(frame_paths)} frames")

    gif_path = fig_dir / "impedance_qc_animation.gif"
    make_gif(frame_paths, gif_path, fps=fps)
    print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
