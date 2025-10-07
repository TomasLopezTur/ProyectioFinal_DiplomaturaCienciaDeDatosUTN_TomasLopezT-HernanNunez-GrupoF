# Tp Final de la diplomatura ciencia de datos y analisis avanzado - UTN BA

### Grupo F
Alumnos: Tomas Lopez Turconi, Hernan Nuñez 

---

## Objetivos
  * Definir claramente el problema a resolver, estableciendo su relevancia para el negocio y los objetivos estratégicos asociados.
  * Seleccionar y analizar un conjunto de datos adecuado, garantizando su calidad mediante procesos de limpieza, tratamiento de valores faltantes e ingeniería de variables.
  * Aplicar una metodología estructurada (CRISP-DM u otra equivalente) para guiar el desarrollo del proyecto de forma coherente y reproducible.
  * Identificar riesgos y limitaciones del proyecto, incluyendo consideraciones éticas y de privacidad de los datos utilizados.

---

### En este Tp se usa una base de datos de [kaggle](https://www.kaggle.com/datasets/programmer3/renewable-energy-microgrid-dataset)
Features:

* timestamp – Date and time of data collection
* solar_pv_output – Actual solar PV power output (kW)
* wind_power_output – Actual wind power output (kW)
* total_renewable_energy – Combined solar and wind power output (kW)
* solar_irradiance – Solar radiation intensity (W/m²)
* wind_speed – Wind speed (m/s)
* temperature – Ambient temperature (°C)
* humidity – Relative humidity (%)
* atmospheric_pressure – Atmospheric pressure (hPa)
* grid_load_demand – Power demand from the grid (kW)
* frequency – Grid frequency (Hz)
* voltage – Grid voltage (V)
* power_exchange – Power exchanged with the grid (kW)
* battery_state_of_charge – Battery charge level (%)
* battery_charging_rate – Battery charging rate (kW)
* battery_discharging_rate – Battery discharging rate (kW)
* hour_of_day – Hour of the day (0–23)
* day_of_week – Day of the week (0=Monday, 6=Sunday)

Target Variables:
* predicted_solar_pv_output – Forecasted solar PV output (kW)
* predicted_wind_power_output – Forecasted wind power output (kW)
* total_predicted_energy – Combined forecasted renewable energy output (kW)

 ---
Parte de la Pre-Entrega
### Resumen Ejecutivo (≤ ½ página)
El presente proyecto se enfoca en el desafío de predecir la generación horaria de energía renovable en una micro-red, utilizando datos de un dataset público de Kaggle. La generación de energía solar y eólica es altamente variable debido a factores climáticos y operativos, lo que dificulta la planificación de consumo y almacenamiento.
El objetivo de negocio es mejorar la confiabilidad y eficiencia energética en comunidades rurales al anticipar cuánta energía estará disponible en cada hora.
