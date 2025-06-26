import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def get_prediction_for_n(target_n, robot_name, log_file='data/workspace_performance_log.csv'):
    """
    Entrena un modelo localmente para predecir el tiempo de ejecución para un N específico.

    Args:
        target_n (int): El número de puntos (N) para el que se quiere la predicción.
        robot_name (str): El nombre del robot para filtrar los datos.
        log_file (str): La ruta al archivo de datos de rendimiento.

    Returns:
        float: El tiempo de ejecución predicho en segundos.
    """
    if not os.path.exists(log_file):
        print(f"Error: El archivo de log '{log_file}' no fue encontrado.")
        return None

    df = pd.read_csv(log_file)
    df = df[df['robot_name'] == robot_name]

    if df.empty:
        print(f"No hay datos para el robot '{robot_name}'.")
        return None

    # --- Lógica de Filtrado para Entrenamiento Local ---
    # Filtramos los datos para entrenar un modelo más preciso para el target_n
    # Usaremos datos hasta 5 veces el valor de N objetivo, con un mínimo de 10000
    max_n_for_training = max(target_n * 5, 10000)
    df_local = df[df['N'] <= max_n_for_training]

    if len(df_local) < 10:
        print(f"Advertencia: Pocos datos locales para N={target_n}. Usando todos los datos disponibles.")
        df_local = df # Si no hay suficientes datos locales, usamos todos

    X = df_local[['N', 'num_links']]
    y = df_local['execution_time']

    # Usamos el modelo polinómico que demostró ser ligeramente mejor
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    model.fit(X, y)

    # Obtenemos el número de links del robot desde el dataframe
    num_links = df_local['num_links'].iloc[0]
    
    # Predecir para el valor N objetivo
    # Crear un DataFrame para la predicción para evitar el UserWarning
    prediction_data = pd.DataFrame([[target_n, num_links]], columns=['N', 'num_links'])
    prediction = model.predict(prediction_data)
    
    return prediction[0]

if __name__ == "__main__":
    ROBOT_NAME = "niryo_one"
    
    # --- Puntos para los que queremos una predicción ---
    puntos_a_predecir = [1000, 5000, 15000, 100000]
    
    print(f"--- Predicciones de Tiempo para el Robot: '{ROBOT_NAME}' ---")
    
    for n_val in puntos_a_predecir:
        tiempo_predicho = get_prediction_for_n(n_val, ROBOT_NAME)
        if tiempo_predicho is not None:
            print(f"Para N = {n_val:6d}, el tiempo de ejecución predicho es: {tiempo_predicho:.4f} segundos")
