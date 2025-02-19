# graficar_holonomicas_y_pfaffianas_sin_optimizar (GraHolPafUnOpt)

# Dado un brazo (2D) con 2 dof, de dos eslabones (L1, angulo1 y L2, angulo2), con un punto fijo en el origen.
# Restringido a una recta y = mx + b.
# El objetivo es graficar las restricciones holonomicas y pfaffianas de un brazo de dos eslabones.

# g(01,02) = 0 ; y - mx -b = 0
# 01 = [...] de 30º a 30º -> Salida en RADINS for 1 a 360
# 02 => g(01,02) = 0, puede dar dos soluciones para cada angulo de 01
# Arrays de 01 y 02 a representar

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Params de la simulación
L1 = 1
L2 = 0.2
m = 0.5
b = 0.2

# Devuelve n angulos espaciados entre 1º y 360º
def angulo1_generar(n=36, min_ang=30, max_ang=55):
    "Devuelve un array de n ángulos en radianes espaciados entre min_ang° y max_ang°"
    angulos_grados = np.linspace(min_ang, max_ang, n)  # Generar ángulos en grados
    angulos_radianes = np.deg2rad(angulos_grados)  # Convertir a radianes
    print("Generando angulos1 en radianes:", angulos_radianes)
    return angulos_radianes  # Devolver en radianes

# Función para obtener los ángulos2 que cumplen la restricción
def angulo2_obtener(m, b, L1, L2, angulo1_array):
    "Dado un array de angulo1 en radianes, obtener los valores de angulo2 que cumplen la restricción"
    
    angulo2 = sp.symbols('angulo2')
    angulo1_filtrado = []
    soluciones_reales = []

    for angulo1_rad in angulo1_array:
        print("Calculando angulo2 para angulo1:", angulo1_rad)
        
        x = L1 * sp.cos(angulo1_rad) + L2 * sp.cos(angulo1_rad + angulo2)
        y = L1 * sp.sin(angulo1_rad) + L2 * sp.sin(angulo1_rad + angulo2)
        holonomica = y - m * x - b
        
        solucion = sp.solve(holonomica, angulo2)
        solucion_numerica = [s.evalf() for s in solucion]
        
        # Filtrar solo soluciones reales
        soluciones_reales_actuales = [s for s in solucion_numerica if s.is_real]

        if soluciones_reales_actuales:  # Si hay al menos una solución real
            angulo1_filtrado.append(angulo1_rad)
            soluciones_reales.append(soluciones_reales_actuales)

        print("Soluciones reales para angulo1:", soluciones_reales_actuales)

    print("ángulos1 con soluciones reales:", angulo1_filtrado)
    print("Soluciones reales encontradas:", soluciones_reales)

    return angulo1_filtrado, soluciones_reales

# Llamar a la función para generar los ángulos1
angulo1_array = angulo1_generar(25)

# Graficar los resultados
def graficar_resultados(angulo1_filtrado, angulo2_reales):
    "Grafica los ángulos obtenidos"
    plt.figure(figsize=(8, 6))

    # Graficar cada conjunto de soluciones reales
    for i in range(len(angulo1_filtrado)):
        for angulo2 in angulo2_reales[i]:
            plt.scatter(angulo1_filtrado[i], float(angulo2), color='b', label="Solución real" if i == 0 else "")

    plt.xlabel("Ángulo 1 (θ1 en radianes)")
    plt.ylabel("Ángulo 2 (θ2 en radianes)")
    plt.title("Soluciones reales de θ2 para distintos valores de θ1")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid()
    plt.show()

# Llamada a la función con filtrado de soluciones reales
print("Buscando")
angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)

# Mostrar los pares de ángulos encontrados
print("\nPares de ángulos encontrados (en radianes):")
for i in range(len(angulo1_filtrado)):
    for angulo2 in angulo2_reales[i]:
        print(f"θ1: {angulo1_filtrado[i]:.4f}, θ2: {float(angulo2):.4f}")

# Llamar a la función para graficar
graficar_resultados(angulo1_filtrado, angulo2_reales)