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
