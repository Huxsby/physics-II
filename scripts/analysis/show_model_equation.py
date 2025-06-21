import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

def show_model_equations(robot_name, log_file='workspace_performance_log.csv'):
    """
    Entrena los modelos y muestra sus ecuaciones matemáticas.
    """
    if not os.path.exists(log_file):
        print(f"Error: El archivo de log '{log_file}' no fue encontrado.")
        return

    df = pd.read_csv(log_file)
    df = df[df['robot_name'] == robot_name]
    if df.empty:
        print(f"No hay datos para el robot '{robot_name}'.")
        return

    X = df[['N', 'num_links']]
    y = df['execution_time']

    # --- Modelo Lineal ---
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    
    # Coeficientes: El primer valor es para 'N', el segundo para 'num_links'
    coef_N_linear = linear_model.coef_[0]
    # El intercepto es el término independiente
    intercept_linear = linear_model.intercept_

    print("--- Ecuación del Modelo Lineal ---")
    print(f"Tiempo(N) ≈ {intercept_linear:.6f} + ({coef_N_linear:.8f} * N)")
    print("Nota: El coeficiente para 'num_links' se omite por simplicidad, ya que es constante para un robot.")


    # --- Modelo Polinómico ---
    poly_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    poly_pipeline.fit(X, y)
    
    # Extraer los componentes del pipeline
    poly_features = poly_pipeline.named_steps['polynomialfeatures']
    poly_regression = poly_pipeline.named_steps['linearregression']

    # Nombres de las características polinómicas (ej: 'N', 'num_links', 'N^2', 'N * num_links', etc.)
    feature_names = poly_features.get_feature_names_out(['N', 'num_links'])
    coefs_poly = poly_regression.coef_

    intercept_poly = poly_regression.intercept_

    print("\n--- Ecuación del Modelo Polinómico (Grado 2) ---")
    equation = f"Tiempo(N) ≈ {intercept_poly:.6e}"
    for coef, name in zip(coefs_poly, feature_names):
        # Simplificamos para mostrar solo los términos que dependen de N
        if 'num_links' not in name or name == 'N * num_links':
             equation += f" + ({coef:.8e} * {name})"

    print(equation)
    print("Nota: La ecuación completa incluye términos con 'num_links', pero se muestran los más relevantes para la predicción basada en N.")


if __name__ == "__main__":
    ROBOT_NAME = "niryo_one"
    show_model_equations(ROBOT_NAME)
