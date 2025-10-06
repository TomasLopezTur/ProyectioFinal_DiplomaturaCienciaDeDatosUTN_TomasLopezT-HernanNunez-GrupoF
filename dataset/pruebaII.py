# train_models.py
# Mejoras sobre el baseline: features de series temporales + modelos (LinearRegression y RandomForest)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# 1) CARGA Y PREPARACIÓN
# ---------------------------
CSV_PATH = os.path.join("dataset", "Renewable_energy_dataset.csv")
df = pd.read_csv(CSV_PATH)

# Parseo de tiempo y orden temporal
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Target
TARGET = "total_renewable_energy"

# ---------------------------
# 2) FEATURE ENGINEERING
# ---------------------------
# a) Variables cíclicas de tiempo
#   - Si ya tenés hour_of_day y day_of_week en el CSV, los usamos; si no, los generamos
if "hour_of_day" not in df.columns:
    df["hour_of_day"] = df["timestamp"].dt.hour
if "day_of_week" not in df.columns:
    df["day_of_week"] = df["timestamp"].dt.dayofweek

# Codificación cíclica (mejor que one-hot para ciclos)
df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

# b) Lags y ventanas para el target (dependencia temporal)
def add_lags_rolling(data, col, lags=(1, 2, 24), roll_windows=(3, 6)):
    for L in lags:
        data[f"{col}_lag{L}"] = data[col].shift(L)
    for W in roll_windows:
        data[f"{col}_rollmean{W}"] = data[col].shift(1).rolling(W).mean()  # rolling pasado
    return data

df = add_lags_rolling(df, TARGET, lags=(1, 2, 24), roll_windows=(3, 6))

# c) Lags básicos de predictores clave (opcional pero suele ayudar)
for col in ["solar_irradiance", "wind_speed", "temperature", "humidity"]:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)

# Eliminamos filas iniciales con NaN generados por lags
df = df.dropna().reset_index(drop=True)

# ---------------------------
# 3) TRAIN / TEST SPLIT (temporal)
# ---------------------------
# Usamos última porción como test (ej. 20%)
test_size_ratio = 0.2
split_idx = int((1 - test_size_ratio) * len(df))

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

# Columnas numéricas candidatas (todas menos timestamp/target obvios)
drop_cols = {"timestamp", TARGET}
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]

# ---------------------------
# 4) MÉTRICAS + MASE
# ---------------------------
def metrics_report(y_true, y_pred, y_train_for_mase=None, y_train_naive=None):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # compatible con scikit-learn viejo
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    mase = None
    if (y_train_for_mase is not None) and (y_train_naive is not None):
        mae_naive_train = mean_absolute_error(y_train_for_mase, y_train_naive)
        mase = mae / mae_naive_train if mae_naive_train > 0 else np.nan
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MASE": mase}

# Naïve para entreno (para MASE): y_t-1 como predicción
train_naive = train_df[TARGET].shift(1).dropna()
train_truth_for_mase = train_df[TARGET].iloc[1:]  # alineado con lag-1

# ---------------------------
# 5) BASELINE NAÏVE (en test) – referencia
# ---------------------------
test_naive_pred = test_df[TARGET].shift(1)  # OJO: esto usa lag dentro del test
# Para un naive “puro” que no mire el primer punto del test, descartamos primer fila del test
valid_mask = ~test_naive_pred.isna()
naive_metrics = metrics_report(
    y_true=y_test[valid_mask],
    y_pred=test_naive_pred[valid_mask],
    y_train_for_mase=train_truth_for_mase,
    y_train_naive=train_naive
)
print("=== BASELINE NAÏVE (lag-1) ===")
for k, v in naive_metrics.items():
    print(f"{k}: {v:.4f}")
print()

# ---------------------------
# 6) MODELOS
# ---------------------------
# A) Regresión Lineal (con escalado)
lin_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lin_pipeline.fit(X_train, y_train)
lin_pred = lin_pipeline.predict(X_test)
lin_metrics = metrics_report(
    y_true=y_test,
    y_pred=lin_pred,
    y_train_for_mase=train_truth_for_mase,
    y_train_naive=train_naive
)

print("=== REGRESIÓN LINEAL (con escalado) ===")
for k, v in lin_metrics.items():
    print(f"{k}: {v:.4f}")
print()

# B) Random Forest (sin escalado necesario)
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_metrics = metrics_report(
    y_true=y_test,
    y_pred=rf_pred,
    y_train_for_mase=train_truth_for_mase,
    y_train_naive=train_naive
)

print("=== RANDOM FOREST ===")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.4f}")
print()

# ---------------------------
# 7) GRÁFICOS: Pred vs Real (muestra)
# ---------------------------
def plot_pred_vs_real(time_index, y_true, y_pred, title, n=200):
    plt.figure(figsize=(12, 4))
    idx = np.arange(len(y_true))[:n]
    plt.plot(time_index.iloc[idx], y_true.iloc[idx], label="Real")
    plt.plot(time_index.iloc[idx], y_pred[:n], label="Pred", alpha=0.9)
    plt.legend()
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel(TARGET)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("Graficando (primeras 200 observaciones del set de test)...")
plot_pred_vs_real(test_df["timestamp"], y_test, lin_pred, "Regresión Lineal: Real vs Pred (test)", n=200)
plot_pred_vs_real(test_df["timestamp"], y_test, rf_pred,  "Random Forest: Real vs Pred (test)", n=200)

# ---------------------------
# 8) IMPORTANCIA DE FEATURES (RF)
# ---------------------------
# Importancias
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features (Random Forest):")
print(importances.head(15))

plt.figure(figsize=(8,6))
importances.head(15).sort_values().plot(kind="barh")
plt.title("Importancia de variables - Random Forest (Top 15)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()
