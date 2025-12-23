# EEG Impedance Quality Control with MNE-Python

This project demonstrates a simple and visual EEG quality control (QC) pipeline
based on electrode impedance values using **MNE-Python**.

## Project Goals
- Visualize EEG electrode impedance using scalp topographic maps
- Detect high-impedance ("bad") channels using threshold-based rules
- Compare impedance at recording start vs end
- Provide a practical QC workflow that can be extended to real EEG systems

## Tools
- Python
- MNE-Python
- NumPy / Pandas
- Matplotlib

## Data
- EEG signals: MNE EEGBCI dataset (EDF)
- Impedance values: simulated CSV files for demonstration purposes

## Structure
- `scripts/` : analysis scripts
- `data/` : input data (not tracked)
- `figures/` : generated visualizations

## Status
Work in progress
