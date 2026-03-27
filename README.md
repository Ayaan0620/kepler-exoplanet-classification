# Exoplanet Classification using Kepler Data

Machine learning project that classifies Kepler Objects of Interest (KOIs) as confirmed exoplanets or false positives. We compare four models (logistic regression from scratch, sklearn LR, random forest, XGBoost) across different feature sets, including an experiment testing whether measurement uncertainty (error bars) alone can classify planets nearly as well as the raw measurements.

## How to Run

```bash
pip install -r requirements.txt
python src/run_experiments.py
```

The cleaning script (`src/cleaner.py`) processes the raw NASA data into `data/kepler_clean_v2.csv`. The main experiment script (`src/run_experiments.py`) runs 10-fold cross-validation across all models and feature sets.

## Project Structure

```
data/           - raw and cleaned datasets
src/            - all python scripts
results/        - experiment output csv and figures
```

## Team Members

[ADD NAMES]

## Course

[ADD COURSE INFO]
