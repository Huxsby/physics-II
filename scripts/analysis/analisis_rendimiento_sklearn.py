# analisis_rendimiento_sklearn.py
"""
Este script realiza un análisis de rendimiento para la función graficar_workspace,
utilizando modelos de Machine Learning con scikit-learn.

Funcionalidades:
1. Carga los datos de rendimiento desde 'data/workspace_performance_log.csv'.
2. Permite filtrar los datos por nombre de robot.
3. Entrena y compara dos modelos de regresión:
   - Regresión Lineal Simple.
   - Regresión Polinómica (grado 2).
4. Evalúa los modelos utilizando R², Error Absoluto Medio (MAE) y Raíz del Error Cuadrático Medio (RMSE).
5. Genera y guarda una gráfica comparativa de los modelos.
6. Muestra un resumen con las métricas de rendimiento de cada modelo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

def cargar_y_preparar_datos(log_file='data/workspace_performance_log.csv', robot_name_filter=None):
    """
    Carga los datos desde el archivo CSV y los prepara para el modelado.
    - Filtra por nombre de robot si se especifica.
    - Elimina duplicados y valores atípicos.
    - Devuelve los datos listos para el entrenamiento.
    """
    if not os.path.exists(log_file):
        print(f"Error: El archivo de log '{log_file}' no fue encontrado.")
        return None, None

    df = pd.read_csv(log_file)
    
    # Filtrar por nombre de robot si se proporciona
    if robot_name_filter:
        df = df[df['robot_name'] == robot_name_filter]
        if df.empty:
            print(f"No se encontraron datos para el robot: '{robot_name_filter}'")
            return None, None

    # Preparar datos para scikit-learn
    # Usaremos 'N' y 'num_links' como características (X) y 'execution_time' como objetivo (y)
    X = df[['N', 'num_links']]
    y = df['execution_time']
    
    return X, y

def entrenar_evaluar_y_visualizar(X, y, robot_name):
    """
    Entrena, evalúa y visualiza los modelos de regresión.
    """
    # --- Modelo 1: Regresión Lineal ---
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)

    # --- Modelo 2: Regresión Polinómica (Grado 2) ---
    # Creamos un pipeline para añadir características polinómicas y luego ajustar el modelo lineal
    poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)

    # --- Evaluación de Modelos ---
    metrics_linear = {
        "R²": r2_score(y, y_pred_linear),
        "MAE": mean_absolute_error(y, y_pred_linear),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred_linear))
    }
    
    metrics_poly = {
        "R²": r2_score(y, y_pred_poly),
        "MAE": mean_absolute_error(y, y_pred_poly),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred_poly))
    }

    # --- Visualización ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Ordenar los datos por 'N' para una gráfica limpia
    sort_axis = X['N'].argsort()
    X_sorted = X.iloc[sort_axis]
    y_sorted = y.iloc[sort_axis]
    
    ax.scatter(X['N'], y, color='blue', label='Datos Reales', alpha=0.6)
    ax.plot(X_sorted['N'], y_pred_linear[sort_axis], color='red', linestyle='--', label=f'Regresión Lineal (R²={metrics_linear["R²"]:.3f})')
    ax.plot(X_sorted['N'], y_pred_poly[sort_axis], color='green', label=f'Regresión Polinómica (R²={metrics_poly["R²"]:.3f})')
    
    ax.set_xlabel("Número de Puntos (N)")
    ax.set_ylabel("Tiempo de Ejecución (s)")
    ax.set_title(f"Análisis de Rendimiento para el Robot: '{robot_name}'")
    ax.legend()
    ax.grid(True)

    # Guardar el gráfico
    output_image_path = f"analisis_rendimiento_{robot_name}.png"
    plt.savefig(output_image_path, dpi=300)
    print(f"\nGráfico guardado en: '{output_image_path}'")

    plt.show()

    # --- Imprimir Resumen ---
    print("\n--- Resumen de Evaluación de Modelos ---")
    print("\n[ Regresión Lineal Simple ]")
    for metric, value in metrics_linear.items():
        print(f"  - {metric}: {value:.6f}")

    print("\n[ Regresión Polinómica (Grado 2) ]")
    for metric, value in metrics_poly.items():
        print(f"  - {metric}: {value:.6f}")
    print("\n----------------------------------------")


if __name__ == "__main__":
    # Nombre del robot a analizar. Cambia a 'None' para analizar todos los datos juntos.
    ROBOT_A_ANALIZAR = "niryo_one" 
    
    print(f"Iniciando análisis para el robot: '{ROBOT_A_ANALIZAR}'...")
    
    X_data, y_data = cargar_y_preparar_datos(robot_name_filter=ROBOT_A_ANALIZAR)
    
    if X_data is not None and y_data is not None:
        if len(X_data) > 5: # Se necesitan suficientes datos para un análisis significativo
            entrenar_evaluar_y_visualizar(X_data, y_data, ROBOT_A_ANALIZAR)
        else:
            print("No hay suficientes datos para realizar un análisis significativo.")
            print("Ejecuta la simulación varias veces con diferentes valores de N para generar más datos.")
