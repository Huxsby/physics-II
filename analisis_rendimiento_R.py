# analisis_rendimiento_R.py
# Actualmente este modelo de predicción de tiempo de ejecución se basa en un modelo lineal simple.
# Solo se tiene en cuenta el número de puntos (N) como variable independiente. Y los datos son de un el niryo_one un brazo de 6 links y articulaciones.
# Para usar este modelo, en la terminal de R, ejecutar:
# install.packages("ggplot2")
# install.packages("broom")
# setwd("tu/ruta/al/proyecto")
# source("script_analisis.R")

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from class_robot_plotter import graficar_workspace
from class_robot_structure import cargar_robot_desde_yaml

def generar_datos_rendimiento(robot, N_valores, num_repeticiones=3):
    """
    Genera datos de tiempo de ejecución para graficar_workspace con diferentes N.
    """
    tiempos = []
    for N_val in N_valores:
        tiempos_N = []
        for _ in range(num_repeticiones):
            print(f"Ejecutando graficar_workspace con N={N_val}...")
            # Llamamos a graficar_workspace sin mostrar el plot y obtenemos el tiempo
            _fig, _ax, _anim, tiempo_calc = graficar_workspace(robot, N=N_val, show_plot=False)
            tiempos_N.append(tiempo_calc)
            if _fig: # Solo cerrar si la figura fue creada
                plt.close(_fig) # Cerrar la figura para no acumularlas
        tiempo_promedio = np.mean(tiempos_N)
        tiempos.append(tiempo_promedio)
        print(f"Tiempo promedio para N={N_val}: {tiempo_promedio:.4f}s")
    return pd.DataFrame({'N': N_valores, 'tiempo': tiempos})

def analizar_y_graficar_con_R(df_tiempos, output_image_path="prediccion_tiempo.png"):
    """
    Usa R para ajustar un modelo y graficar la predicción.
    Guarda el gráfico y muestra la ecuación del modelo.
    """
    # Guardar datos en CSV para que R los lea
    csv_path = "tiempos_workspace.csv"
    df_tiempos.to_csv(csv_path, index=False)

    # Script de R
    # Se asume que Rscript está en el PATH y los paquetes 'ggplot2' y 'broom' están instalados.
    r_script_content = f"""
library(ggplot2)
library(broom) # Para tidy()

# Leer los datos
datos <- read.csv('{csv_path}')

# Ajustar un modelo lineal (puedes probar otros, ej. cuadrático: y ~ x + I(x^2))
# Para predecir tiempo (y) en función de N (x)
modelo <- lm(tiempo ~ N, data=datos)

# Obtener resumen y coeficientes
resumen_modelo <- summary(modelo)
coeficientes <- tidy(modelo)

print('Resumen del Modelo Lineal:')
print(resumen_modelo)
print('Coeficientes del Modelo:')
print(coeficientes)

# Extraer la ecuación para el título del gráfico
intercepto <- coef(modelo)[1]
pendiente_N <- coef(modelo)[2]
ecuacion <- sprintf("Tiempo ≈ %.4f + %.4f * N (R² = %.3f)", 
                    intercepto, pendiente_N, resumen_modelo$r.squared)

# Crear el gráfico con ggplot2
grafico <- ggplot(datos, aes(x=N, y=tiempo)) +
  geom_point(color='blue', size=3) +
  geom_smooth(method='lm', se=TRUE, color='red', formula=y ~ x) + # Línea de regresión
  labs(title=paste("Predicción del Tiempo de Cálculo de graficar_workspace", ecuacion, sep="\n"),
       x="Número de Puntos (N)",
       y="Tiempo de Cálculo Promedio (s)") +
  theme_minimal()

# Guardar el gráfico
ggsave('{output_image_path}', plot=grafico, width=10, height=6)

print(paste("Gráfico guardado en: {output_image_path}"))
"""

    r_script_path = "script_analisis.R"
    with open(r_script_path, "w", encoding='utf-8') as f: # Añadido encoding='utf-8'
        f.write(r_script_content)

    print("\nEjecutando script de R para análisis y graficación...")
    try:
        # Asegúrate de que Rscript esté en el PATH o proporciona la ruta completa
        # Usamos shell=True en Windows si Rscript no es directamente ejecutable o está en una ruta con espacios
        # Para mayor portabilidad, es mejor asegurarse que Rscript esté en el PATH.
        # Si se usa powershell, el comando es simplemente "Rscript script_analisis.R"
        # Si se usa cmd.exe, también es "Rscript script_analisis.R"
        # subprocess.run(["Rscript", r_script_path], check=True, capture_output=True, text=True, shell=False)
        # Para Windows, si Rscript está en el PATH, esto debería funcionar:
        result = subprocess.run(["Rscript", r_script_path], check=True, capture_output=True, text=True, encoding='utf-8')
        print("Script de R ejecutado con éxito.")
        print("Salida de R:")
        print(result.stdout)
        if result.stderr:
            print("Errores de R (si los hay):")
            print(result.stderr)
        
        # Mostrar el gráfico generado por R (opcional, si tienes un visor de imágenes)
        # from IPython.display import Image
        # display(Image(filename=output_image_path))

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script de R:")
        print(f"Código de retorno: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
    except FileNotFoundError:
        print("Error: Rscript no encontrado. Asegúrate de que R esté instalado y Rscript en el PATH.")
        print("Puedes intentar ejecutar el script 'script_analisis.R' manualmente en tu consola de R.")

if __name__ == "__main__":
    print("Cargando robot...")
    # Asegúrate de que este archivo exista y sea correcto en la raíz del proyecto
    robot_yaml_path = "robot.yaml" 
    try:
        robot = cargar_robot_desde_yaml(robot_yaml_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo YAML del robot en '{robot_yaml_path}'.")
        print("Asegúrate de que el archivo está en la ubicación correcta o proporciona la ruta completa.")
        exit()
    except Exception as e:
        print(f"Error al cargar el robot desde YAML: {e}")
        exit()

    # Definir los valores de N para probar
    # Puedes ajustar estos valores según sea necesario
    # Valores más pequeños para una prueba rápida:
    # valores_N = [100, 200, 300, 400, 500] 
    valores_N = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000] 
    
    print("\nGenerando datos de rendimiento...")
    df_tiempos_calculados = generar_datos_rendimiento(robot, valores_N, num_repeticiones=3)
    
    if not df_tiempos_calculados.empty:
        analizar_y_graficar_con_R(df_tiempos_calculados, output_image_path="c:\\\\Users\\\\Huxsby\\\\Documents\\\\repgit\\\\physics-II\\\\prediccion_tiempo_workspace.png")
    else:
        print("No se generaron datos de rendimiento.")

    print("\\nAnálisis de rendimiento completado.")
    print(f"Puedes encontrar el gráfico de predicción en: c:\\\\Users\\\\Huxsby\\\\Documents\\\\repgit\\\\physics-II\\\\prediccion_tiempo_workspace.png")
    print(f"Y los datos crudos en: c:\\\\Users\\\\Huxsby\\\\Documents\\\\repgit\\\\physics-II\\\\tiempos_workspace.csv")
