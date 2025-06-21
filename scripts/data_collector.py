import numpy as np
from src.simulation.class_robot_plotter import graficar_workspace
from src.core import Robot, cargar_robot_desde_yaml
import matplotlib.pyplot as plt

def recopilar_datos_rendimiento(robot: Robot, n_inicial, n_final, n_pasos):
    """
    Ejecuta graficar_workspace para un rango de valores de N y recopila datos.
    Los datos se guardan automáticamente en 'workspace_performance_log.csv'.
    """
    print(f"--- Iniciando Recopilación de Datos ---")
    print(f"Robot: {robot.name}")
    print(f"Rango de N: de {n_inicial} a {n_final} en pasos de {n_pasos}")
    
    valores_N = np.arange(n_inicial, n_final + 1, n_pasos)
    
    for i, n_val in enumerate(valores_N):
        print(f"\n[{i+1}/{len(valores_N)}] Ejecutando para N = {n_val}...")
        try:
            # Llamamos a la función sin mostrar el plot para solo registrar los datos
            fig, ax, anim, tiempo = graficar_workspace(robot, N=n_val, show_plot=False)
            if fig:
                plt.close(fig) # Cerramos la figura para no consumir memoria
        except Exception as e:
            print(f"Error al ejecutar para N={n_val}: {e}")
            
    print("\n--- Recopilación de Datos Finalizada ---")

if __name__ == "__main__":
    # Define el rango de N para la recolección de datos
    N_INICIAL = 100000
    N_FINAL = 1000000
    N_PASOS = 5000  # Se ejecutará para N = 1000, 2000, 3000, ..., 10000

    # --- Ejecución ---
    try:
        robot = cargar_robot_desde_yaml("config/robot.yaml")
        recopilar_datos_rendimiento(robot, N_INICIAL, N_FINAL, N_PASOS)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del robot en '{robot.name}'")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
