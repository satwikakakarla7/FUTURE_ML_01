# FUTURE_ML_01 — Sales & Demand Forecasting for Businesses

## Objective
Build a machine-learning forecasting model that estimates future daily sales from historical business data.

## Dataset
This repository uses a **synthetic demonstration dataset** created for the internship portfolio. It contains daily sales observations for four products from January 2024 through June 2025. It is not company/customer data.

## Approach
1. Load and validate the historical sales data.
2. Aggregate product-level sales to daily business demand.
3. Create time-based features:
   - day of week
   - month
   - day of year
   - week of year
   - trend
   - 1/7/14/28-day lags
   - 7/28-day rolling averages
4. Split chronologically: 80% train / 20% holdout.
5. Train a Random Forest regression model.
6. Evaluate using MAE, RMSE and MAPE.
7. Produce a recursive 30-day forecast.
8. Visualize actual vs predicted sales and future demand.

## Business interpretation
The forecast can support inventory planning, staffing, purchasing and promotional planning. Forecasts should be refreshed when new sales data arrives.

## Results
Run:

```bash
python src/sales_forecasting.py
```

Metrics are written to `outputs/metrics.json`.

## Important limitation
This is a portfolio demonstration using synthetic data. It should not be described as a production forecasting system or as a forecast of a real company's sales.

## Run the notebook
Open `notebooks/task01_sales_forecasting.ipynb` in Jupyter Notebook or VS Code.
