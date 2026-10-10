import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
OUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

def add_features(df):
    x = df.copy().sort_values("date").reset_index(drop=True)
    x["dayofweek"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["dayofyear"] = x["date"].dt.dayofyear
    x["weekofyear"] = x["date"].dt.isocalendar().week.astype(int)
    x["trend"] = np.arange(len(x))
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = x["sales"].shift(lag)
    x["rolling_7"] = x["sales"].shift(1).rolling(7).mean()
    x["rolling_28"] = x["sales"].shift(1).rolling(28).mean()
    return x

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    daily = df.groupby("date", as_index=False)["sales"].sum()
    featured = add_features(daily).dropna().reset_index(drop=True)

    split = int(len(featured) * 0.80)
    train, test = featured.iloc[:split], featured.iloc[split:]
    features = [
        "dayofweek","month","dayofyear","weekofyear","trend",
        "lag_1","lag_7","lag_14","lag_28","rolling_7","rolling_28"
    ]

    model = RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
    model.fit(train[features], train["sales"])
    pred = model.predict(test[features])

    mae = mean_absolute_error(test["sales"], pred)
    rmse = np.sqrt(mean_squared_error(test["sales"], pred))
    mape = np.mean(np.abs((test["sales"] - pred) / test["sales"])) * 100

    metrics = {
        "model": "RandomForestRegressor",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE_percent": round(float(mape), 2)
    }
    (OUT/"metrics.json").write_text(json.dumps(metrics, indent=2))

    test_out = test[["date","sales"]].copy()
    test_out["predicted_sales"] = pred
    test_out.to_csv(OUT/"test_predictions.csv", index=False)

    plt.figure(figsize=(11,5))
    plt.plot(test_out["date"], test_out["sales"], label="Actual")
    plt.plot(test_out["date"], test_out["predicted_sales"], label="Predicted")
    plt.title("Sales Forecast — Holdout Period")
    plt.xlabel("Date"); plt.ylabel("Daily Sales")
    plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS/"actual_vs_predicted.png", dpi=160)
    plt.close()

    # Recursive 30-day forecast
    history = daily.copy()
    forecasts = []
    for _ in range(30):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        temp = pd.concat(
            [history, pd.DataFrame({"date":[next_date], "sales":[np.nan]})],
            ignore_index=True
        )
        feat = add_features(temp).iloc[-1]
        row = pd.DataFrame([feat])
        yhat = float(model.predict(row[features])[0])
        history = pd.concat(
            [history, pd.DataFrame({"date":[next_date], "sales":[yhat]})],
            ignore_index=True
        )
        forecasts.append([next_date, yhat])

    forecast = pd.DataFrame(forecasts, columns=["date","forecast_sales"])
    forecast.to_csv(OUT/"next_30_days_forecast.csv", index=False)

    plt.figure(figsize=(11,5))
    plt.plot(forecast["date"], forecast["forecast_sales"], marker="o")
    plt.title("Next 30 Days Sales Forecast")
    plt.xlabel("Date"); plt.ylabel("Forecast Sales")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(CHARTS/"30_day_forecast.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
