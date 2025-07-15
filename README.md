# Time-Series-Forecasting-with-Machine-Learning-to-predict-energy-consumption

---

## 🧠 What’s the Goal?

> Forecast **future hourly energy consumption** using XGBoost and time series features.

---

## 📦 Step-by-Step Explanation of the Code (from PDF)

---

### 🔹 Step 1: Import and Load the Data

```python
df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
```

* Loads hourly energy usage data (`PJME_MW`)
* Sets the `Datetime` as the index so we can treat it like a time series

---

### 🔹 Step 2: Outlier Removal

```python
df.query('PJME_MW < 19_000')['PJME_MW'].plot(...)
df = df.query('PJME_MW > 19_000').copy()
```

* Removes values where energy use was **less than 19,000 MW**, considering them outliers (errors or rare low demand)
* Helps improve model quality

---

### 🔹 Step 3: Train-Test Split

```python
train = df[df.index < '2015-01-01']
test = df[df.index >= '2015-01-01']
```

* Uses past data for training, and data from 2015 onwards for testing

---

### 🔹 Step 4: TimeSeriesSplit for Cross-Validation

```python
TimeSeriesSplit(n_splits=5, test_size=8760, gap=24)
```

* Instead of just one train/test split, this splits the data **5 times** (folds)
* Each fold uses one year (8760 hours) of data for testing
* `gap=24` avoids overlap near the split boundary

---

### 🔹 Step 5: Feature Creation

```python
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
...
```

* Extracts time-based features (hour of day, day of week, month, etc.)
* Helps the model learn **seasonal patterns** (e.g., high demand during certain hours/days)

---

### 🔹 Step 6: Add Lag Features

```python
df['lag1'] = df.shift(364 days)
df['lag2'] = df.shift(728 days)
df['lag3'] = df.shift(1092 days)
```

* These features pull energy values from **1, 2, and 3 years ago (same hour)**
* Energy usage often follows seasonal/yearly patterns — this helps the model capture that

---

### 🔹 Step 7: Cross-Validation + Model Training

```python
for train_idx, val_idx in tss.split(df):
    # Prepare data
    # Train XGBoost on X_train, y_train
    # Predict on X_test, calculate RMSE
```

Each loop:

* Trains on one chunk of past data
* Tests on the next year
* Collects predictions + RMSE scores

> ✅ **Score across folds: \~3750 MW RMSE**

This gives more reliable performance estimation.

---

### 🔹 Step 8: Train Final Model on ALL Data

```python
reg.fit(X_all, y_all)
```

* After cross-validation, trains **one final model using all data** up to Aug 2018
* Prepares to **predict the future**

---

### 🔹 Step 9: Forecast the Future

```python
future = pd.date_range('2018-08-03','2019-08-01', freq='1h')
```

* Creates an empty DataFrame of future hourly timestamps
* Adds features (hour, month, lags, etc.)
* Uses the trained model to predict energy use for these times

---

### 🔹 Step 10: Plot Future Predictions

```python
future_w_features['pred'].plot(...)
```

* Shows the predicted energy use from Aug 2018 to Aug 2019

---

### 🔹 Step 11: Save and Reload Model

```python
reg.save_model('model.json')
```

* Saves the trained model to disk
* Can load it again later and reuse it

---

## ✅ What Improvements Were Made Over the First Version?

| Improvement           | Explanation                                           |
| --------------------- | ----------------------------------------------------- |
| ✅ Outlier Removal     | Removes bad values to reduce noise                    |
| ✅ TimeSeriesSplit     | Better model validation, not just one test set        |
| ✅ Lag Features        | Adds memory of energy use in past years               |
| ✅ Retrain on all data | Uses full dataset for final model                     |
| ✅ Forecasts future    | Not just test prediction, but real forward prediction |

---

## 📏 Final Model Accuracy?

* **Cross-validated RMSE** across 5 folds = \~**3750 MW**
* That means your model is **reasonably accurate**, with an average prediction error of about **3750 megawatts** per hour

---

## 🔧 Want to Improve It Even More?

You could try:

* Adding weather data (temp, humidity)
* Including public holidays
* Using rolling averages (24-hour, 7-day trends)
* Trying LightGBM or deep learning (LSTM)


