
# Impedance-Based EEG Quality Control (MNE-Python)

This project implements a **practical impedance-based quality control (QC) workflow for EEG recordings**, built on top of **MNE-Python**.  
It focuses on **spatial and temporal visualization** of electrode impedance to help identify poor electrode–scalp contact during EEG recordings.

The project is designed as a **lightweight, extensible QC layer**, without relying on complex signal processing or device-specific software.

---

## Project Overview

EEG signal quality is highly dependent on electrode–skin contact.  
High electrode impedance often leads to noisy or unreliable signals, yet impedance information is typically provided only as raw numeric values.

This project provides:
- **Scalp topographic maps** of electrode impedance
- **Begin–end impedance comparison**
- **Delta (change) visualization**
- **Time-resolved impedance animation** (GIF)
- Simple **threshold-based QC summaries**

All visualizations are implemented using **MNE-Python** and standard scientific Python tools.

---

## Key Features

- Impedance visualization on the scalp using standard EEG montages
- Robust handling of missing, invalid, or overlapping electrode positions
- Begin vs. end impedance comparison
- Delta impedance maps (End − Begin)
- Time-resolved impedance animation (5-second sampling)
- CSV and Markdown outputs for reporting
- Designed to be easily extended to real impedance measurements

---

## Technologies Used

- Python 3
- MNE-Python
- NumPy
- Pandas
- Matplotlib
- imageio (for GIF generation)
- EEGBCI dataset (for channel structure and montage reference)

---

## Project Structure

mne-impedance-project/
├── src/
│   ├── impedance_topomap.py
│   ├── impedance_compare.py
│   └── impedance_animation.py
├── data/
│   └── impedance/
├── figures/
├── reports/
├── requirements.txt
└── README.md

---

## Installation

All dependencies are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## How to Run the Scripts

```bash
python src/impedance_topomap.py
python src/impedance_compare.py
python src/impedance_animation.py
```

---

## Interpretation of Results

- High impedance regions indicate poor electrode contact
- Delta maps highlight temporal changes
- Animations reveal when impedance issues emerge

---

## Extensibility

- Real impedance measurements
- Real-time QC dashboards
- Integration with signal quality metrics

---