# Dado un brazo (2D) con 2 dof, de dos eslabones (L1, angulo1 y L2, angulo2), con un punto fijo en el origen.
# Restringido a una recta y = mx + b.
# El objetivo es graficar las restricciones holonomicas y pfaffianas de un brazo de dos eslabones.

# g(01,02) = 0 ; y - mx -b = 0
# 01 = [...] de 30º a 30º -> Salida en RADINS for 1 a 360
# 02 => g(01,02) = 0, puede dar dos soluciones para cada angulo de 01
# Arrays de 01 y 02 a representar

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parámetros de la simulación
L1 = 1
L2 = 0.2
m = 0.5
b = 0.2

# Restricción holonómica
def restriccion(angulo2, angulo1, L1, L2, m, b):
    x = L1 * np.cos(angulo1) + L2 * np.cos(angulo1 + angulo2)
    y = L1 * np.sin(angulo1) + L2 * np.sin(angulo1 + angulo2)
    return y - m * x - b

# Devuelve n ángulos espaciados entre min_ang y max_ang
def angulo1_generar(n=36, min_ang=30, max_ang=55):
    """Devuelve un array de n ángulos en radianes espaciados entre min_ang° y max_ang°"""
    angulos_grados = np.linspace(min_ang, max_ang, n)  # Generar ángulos en grados
    angulos_radianes = np.deg2rad(angulos_grados)  # Convertir a radianes
    return angulos_radianes  # Devolver en radianes

# Función para obtener los ángulos2 que cumplen la restricción
def angulo2_obtener(m, b, L1, L2, angulo1_array):
    """Dado un array de angulo1 en radianes, obtener los valores de angulo2 que cumplen la restricción"""
    
    angulo1_filtrado = []
    soluciones_reales = []
    
    # Puntos de inicio para las dos posibles soluciones
    semillas = [0.0, np.pi]  # Empezamos buscando desde 0 y desde π
    
    for angulo1_rad in angulo1_array:
        soluciones_actuales = []
        
        for semilla in semillas:
            try:
                sol = fsolve(restriccion, semilla, args=(angulo1_rad, L1, L2, m, b))
                # Verificar si la solución es correcta
                residual = abs(restriccion(sol[0], angulo1_rad, L1, L2, m, b))
                
                if residual < 1e-6:  # Si el residuo es pequeño, la solución es válida
                    # Normalizar ángulo a [-π, π]
                    while sol[0] > np.pi:
                        sol[0] -= 2*np.pi
                    while sol[0] < -np.pi:
                        sol[0] += 2*np.pi
                    soluciones_actuales.append(sol[0])
            except:
                continue
                
        # Si hay soluciones, guardar
        if soluciones_actuales:
            angulo1_filtrado.append(angulo1_rad)
            soluciones_actuales = list(set(round(x, 4) for x in soluciones_actuales))
            soluciones_reales.append(soluciones_actuales)
    
    return angulo1_filtrado, soluciones_reales

# Función para expandir las soluciones (para graficar)
def expandir_soluciones(angulo1_filtrado, soluciones_reales):
    """Expande las listas para que cada solución tenga su propio punto para graficar"""
    angulos1_expandidos = []
    angulos2_expandidos = []
    
    for i, ang1 in enumerate(angulo1_filtrado):
        for sol in soluciones_reales[i]:
            angulos1_expandidos.append(ang1)
            angulos2_expandidos.append(sol)
    
    return angulos1_expandidos, angulos2_expandidos

# Graficar los resultados
def graficar_resultados(angulo1_filtrado, angulo2_reales):
    """Grafica los ángulos obtenidos"""
    # Expandir soluciones para graficar
    angulos1_exp, angulos2_exp = expandir_soluciones(angulo1_filtrado, angulo2_reales)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(angulos1_exp, angulos2_exp, color='b', marker='o')
    
    plt.xlabel("Ángulo 1 (θ1 en radianes)")
    plt.ylabel("Ángulo 2 (θ2 en radianes)")
    plt.title("Soluciones reales de θ2 para distintos valores de θ1")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True)
    
    # Agregar leyenda
    plt.legend(['Solución real'])
    
    plt.show()

# Llamada a la función con filtrado de soluciones reales
angulo1_array = angulo1_generar(100)  # Incrementamos el número para tener más detalle
print("Buscando soluciones...")

import time
start_time = time.time()

angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)

print(f"Cálculo completado en {time.time() - start_time:.2f} segundos")

# Mostrar los pares de ángulos encontrados
print("\nPares de ángulos encontrados (en radianes):")
for i in range(len(angulo1_filtrado)):
    for angulo2 in angulo2_reales[i]:
        print(f"θ1: {angulo1_filtrado[i]:.4f}, θ2: {angulo2:.4f}")

# Llamar a la función para graficar
graficar_resultados(angulo1_filtrado, angulo2_reales)

# Verificación visual: graficar el brazo para algunos casos
def graficar_brazo(angulo1, angulo2, L1, L2, m, b):
    """Grafica el brazo y la restricción para un par de ángulos"""
    plt.figure(figsize=(10, 8))
    
    # Posiciones de las articulaciones
    x1 = L1 * np.cos(angulo1)
    y1 = L1 * np.sin(angulo1)
    x2 = x1 + L2 * np.cos(angulo1 + angulo2)
    y2 = y1 + L2 * np.sin(angulo1 + angulo2)
    
    # Graficar eslabones
    plt.plot([0, x1, x2], [0, y1, y2], 'ro-', linewidth=3, markersize=8)
    
    # Graficar restricción (recta)
    x_recta = np.linspace(-1.5, 1.5, 100)
    y_recta = m * x_recta + b
    plt.plot(x_recta, y_recta, 'b--', label=f'y = {m}x + {b}')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Brazo Robótico: θ1 = {angulo1:.4f}, θ2 = {angulo2:.4f}')
    plt.legend()
    
    plt.show()

# Graficar un par de casos para verificar
if len(angulo1_filtrado) > 0:
    idx = len(angulo1_filtrado) // 2  # Tomamos un caso intermedio
    graficar_brazo(angulo1_filtrado[idx], angulo2_reales[idx][0], L1, L2, m, b)
    
    if len(angulo2_reales[idx]) > 1:  # Si hay segunda solución
        graficar_brazo(angulo1_filtrado[idx], angulo2_reales[idx][1], L1, L2, m, b)