import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf


def load_eeg(subject=1, run=3):
    raw_file = eegbci.load_data(subject, [run])[0]
    raw = read_raw_edf(raw_file, preload=True, verbose=False)
    raw.pick(picks="eeg")
    return raw


def clean_channel_names(raw):
    mapping = {}
    for ch in raw.ch_names:
        new = ch.strip()
        if new.endswith("."):
            new = new[:-1]
        mapping[ch] = new
    raw.rename_channels(mapping)
    return raw


def apply_standard_montage(raw):
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    return raw


if __name__ == "__main__":
    raw = load_eeg()
    raw = clean_channel_names(raw)
    raw = apply_standard_montage(raw)

    print(raw)
    print("Montage successfully applied (on_missing='ignore').")
    print("First 15 channels:", raw.ch_names[:15])
