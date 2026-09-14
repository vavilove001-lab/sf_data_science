# Data

Raw data files are not committed to the repository.

## Required files

Place the following files in this directory:

| File | Description |
| :--- | :--- |
| `train.csv` | Main training dataset with trip details |
| `Project5_test_data.csv` | Test dataset for final predictions |
| `osrm_data_train.csv` | OSRM route features for training set |
| `Project5_osrm_data_test.csv` | OSRM route features for test set |
| `holiday_data.csv` | Public holidays calendar |
| `weather_data.csv` | Historical weather conditions |

## Instructions

1. Download the required files from the competition source or your local storage.
2. Place them directly into the `data/` folder.
3. Run `notebooks/trip_duration_prediction.ipynb` to start the pipeline.

> ⚠️ **Note:** Do not commit `.csv` files to Git. They are ignored by `.gitignore`.