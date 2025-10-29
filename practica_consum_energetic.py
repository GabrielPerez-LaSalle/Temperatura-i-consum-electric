# practica_consum_energetic.py
# -------------------------------------------
# 🌍 Projecte integrador — Temperatura i consum elèctric
# Autor: [Tu nombre]
# -------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# -------------------------------------------
# 1️⃣ Càrrega i preparació de les dades
# -------------------------------------------

energy_path = "energy_dataset.csv"
weather_path = "weather_features.csv"

# Cargar datos
energy = pd.read_csv(energy_path)
weather = pd.read_csv(weather_path)

print("\n✅ Archivos cargados correctamente.")
print("\nColumnas de energy_dataset.csv:")
print(energy.columns)
print("\nColumnas de weather_features.csv:")
print(weather.columns)

# Convertir las columnas de fecha
energy['time'] = pd.to_datetime(energy['time'], utc=True)
weather['time'] = pd.to_datetime(weather['dt_iso'], utc=True)

# -------------------------------------------
# Crear DataFrames diarios
# -------------------------------------------

# Consum elèctric (sumat per dia)
energy_daily = energy.groupby(energy['time'].dt.date)['total load actual'].sum().reset_index()

# Temperatura mitjana diària (mitjana per dia)
weather_daily = weather.groupby(weather['time'].dt.date)['temp'].mean().reset_index()

# Fusionar dades per data
data = pd.merge(energy_daily, weather_daily, left_on='time', right_on='time')
data.columns = ['Date', 'EnergyConsumption', 'Temperature']

print("\n✅ Dades processades correctament:")
print(data.head())

# -------------------------------------------
# 2️⃣ Exploració inicial
# -------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(data['Temperature'], data['EnergyConsumption'], alpha=0.6)
plt.title("Temperatura vs Consum elèctric")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.grid(True)
plt.show()

# -------------------------------------------
# 3️⃣ Regressió lineal
# -------------------------------------------
X = data[['Temperature']]
y = data['EnergyConsumption']

model = LinearRegression()
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_

print(f"\nCoeficient (pendent w): {w:.2f}")
print(f"Intercept (b): {b:.2f}")

# -------------------------------------------
# 4️⃣ Avaluació del model
# -------------------------------------------
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# -------------------------------------------
# 5️⃣ Visualització del model
# -------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(X, y, label='Dades reals', alpha=0.6)
plt.plot(X, y_pred, color='red', label='Model lineal')
plt.title("Ajust de regressió lineal")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.legend()
plt.show()

# -------------------------------------------
# 6️⃣ EXTRA — Regressió polinòmica (grau 2)
# -------------------------------------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

r2_poly = r2_score(y, y_poly_pred)

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6)
plt.plot(np.sort(X.values, axis=0),
         y_poly_pred[np.argsort(X.values[:, 0])],
         color='orange', label='Model polinòmic (grau 2)')
plt.title("Regressió polinòmica (grau 2)")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.legend()
plt.show()

print(f"\nR² model polinòmic: {r2_poly:.4f}")

print("\n✅ Anàlisi completada correctament.")

# ---------- Análisis adicional y diagnóstico (pegar al final del script) ----------

from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
import math

# RMSE (más interpretable que MSE)
rmse = math.sqrt(mse)
print(f"\nRMSE: {rmse:.2f} MWh (Error medio cuadrático raíz)")

# Predicción ejemplo: ¿qué predice el modelo a 0°C?
pred_0 = model.predict([[0]])[0]
print(f"Predicción del consumo si Temperatura = 0°C: {pred_0:.2f} MWh")

# Residuales
residuals = y - y_pred
print(f"\nEstadísticas residuales:\n  media: {residuals.mean():.2f}\n  std: {residuals.std():.2f}\n  min: {residuals.min():.2f}\n  max: {residuals.max():.2f}")

# Gráfico 1: residuales vs predicho
plt.figure(figsize=(8,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuales vs Predicciones")
plt.xlabel("Predicho (MWh)")
plt.ylabel("Residuales (MWh)")
plt.grid(True)
plt.show()

# Gráfico 2: histograma de residuales (normalidad)
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=40, edgecolor='k', alpha=0.7)
plt.title("Histograma residuales")
plt.xlabel("Residual (MWh)")
plt.ylabel("Frecuencia")
plt.show()

# Test visual: residuales vs temperatura (para ver no-linealidad)
plt.figure(figsize=(8,4))
plt.scatter(data['Temperature'], residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuales vs Temperatura")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Residuales (MWh)")
plt.grid(True)
plt.show()

# Validación cruzada (K-Fold) — para estimar desempeño fuera de muestra
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
mse_cv = -neg_mse_scores.mean()
rmse_cv = math.sqrt(mse_cv)
r2_cv_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
r2_cv = r2_cv_scores.mean()

print(f"\nValidación cruzada (5-fold):\n  RMSE_cv: {rmse_cv:.2f} MWh\n  R²_cv: {r2_cv:.4f}")

# Sugerencia automática simple
print("\nSugerencia:")
if r2 < 0.4:
    print(" - R² bajo: considera usar variables adicionales (estacionalidad, día de la semana), features polinómicas o modelos no lineales.")
elif r2 < 0.7:
    print(" - R² moderado: el modelo captura parte de la variabilidad, mejora posible con más features o polinomio.")
else:
    print(" - R² alto: el modelo lineal simple probablemente captura bien la relación principal.")

# Si quieres, guarda un CSV con las predicciones y residuales
out = data.copy()
out['Predicted'] = y_pred
out['Residual'] = residuals
out.to_csv("predictions_and_residuals.csv", index=False)
print("\nSe ha guardado 'predictions_and_residuals.csv' con predicciones y residuales.")

