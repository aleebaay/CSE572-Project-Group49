
# Airline Delay Prediction Project

## What this is
A leakage-safe binary classification pipeline for predicting whether a flight departure will be delayed by more than 15 minutes.

## Files
- `src/train_flight_delay.py` - main training and evaluation script
- `requirements.txt` - Python dependencies

## Data
Use the Kaggle flight dataset file:
`Combined_Flights_2022.csv`
[Download from Kaggle](https://www.kaggle.com/code/robikscube/flight-delay-exploratory-data-analysis-twitch/input?select=Combined_Flights_2022.csv)

Place the CSV in the same folder where you run the script, or pass its path with `--data`.

## Run
```bash
python src/train_flight_delay.py --data Combined_Flights_2022.csv --sample-size 50000 --outdir outputs
```

## Outputs
The script saves:
- sampled dataset
- model comparison table
- metrics JSON
- target distribution plot
- delay rate by hour plot
- ROC curves
- confusion matrices
- feature importance plot and CSV

## Notes
The pipeline uses only pre-departure variables to avoid data leakage:
- Month
- DayOfWeek
- CRS departure hour
- CRSElapsedTime
- Distance
- Airline / Origin / Dest
