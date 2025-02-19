import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from matplotlib.patches import Wedge

# Parámetros de la simulación
L1 = 1
L2 = 0.2
m = 0.5
b = 0.5

def restriccion(angulo2, angulo1, L1, L2, m, b):
    x = L1 * np.cos(angulo1) + L2 * np.cos(angulo1 + angulo2)
    y = L1 * np.sin(angulo1) + L2 * np.sin(angulo1 + angulo2)
    return y - m * x - b

def angulo1_generar(n=36, min_ang=0, max_ang=360):
    angulos_grados = np.linspace(min_ang, max_ang, n)
    return np.deg2rad(angulos_grados)

def angulo2_obtener(m, b, L1, L2, angulo1_array):
    angulo1_filtrado = []
    soluciones_reales = []
    semillas = [0.0, np.pi]
    
    for angulo1_rad in angulo1_array:
        soluciones_actuales = []
        for semilla in semillas:
            try:
                sol = fsolve(restriccion, semilla, args=(angulo1_rad, L1, L2, m, b))
                residual = abs(restriccion(sol[0], angulo1_rad, L1, L2, m, b))
                if residual < 1e-6:
                    soluciones_actuales.append(sol[0])
            except:
                continue
        
        if soluciones_actuales:
            angulo1_filtrado.append(angulo1_rad)
            soluciones_actuales = list(set(round(x, 4) for x in soluciones_actuales))
            soluciones_reales.append(soluciones_actuales)
    
    return angulo1_filtrado, soluciones_reales

def crear_animacion_brazo_con_conos(n):
    # Obtener soluciones
    angulo1_array = angulo1_generar(n)
    angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)
    
    # Convertir a grados para análisis
    angulos_grados = np.rad2deg(angulo1_filtrado)
    
    # Encontrar los rangos de ángulos para cada cuadrante
    primer_cuadrante = [ang for ang in angulos_grados if ang < 90]
    tercer_cuadrante = [ang for ang in angulos_grados if ang > 180]
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Agregar los conos (wedges)
    if primer_cuadrante:
        min_ang1 = min(primer_cuadrante)
        max_ang1 = max(primer_cuadrante)
        wedge1 = Wedge((0, 0), L1 + L2, min_ang1, max_ang1, 
                       alpha=0.2, color='green', label='Rango θ1 (1er cuadrante)')
        ax.add_patch(wedge1)
    
    if tercer_cuadrante:
        min_ang2 = min(tercer_cuadrante)
        max_ang2 = max(tercer_cuadrante)
        wedge2 = Wedge((0, 0), L1 + L2, min_ang2, max_ang2,
                       alpha=0.2, color='blue', label='Rango θ1 (3er cuadrante)')
        ax.add_patch(wedge2)
    
    # Graficar la línea de restricción
    x_recta = np.linspace(-1.5, 1.5, 100)
    y_recta = m * x_recta + b
    ax.plot(x_recta, y_recta, 'r--', label=f'y = {m}x + {b}')
    
    # Crear líneas para los eslabones
    line1, = ax.plot([], [], 'r-', linewidth=3, label='Eslabón 1')
    line2, = ax.plot([], [], 'b-', linewidth=3, label='Eslabón 2')
    punto_final, = ax.plot([], [], 'ro', markersize=8)
    
    # Preparar los datos para la animación
    frames = []
    for i in range(len(angulo1_filtrado)):
        for angulo2 in angulo2_reales[i]:
            frames.append((angulo1_filtrado[i], angulo2))
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        punto_final.set_data([], [])
        ax.legend()
        return line1, line2, punto_final

    def update(frame):
        angulo1, angulo2 = frame
        
        # Calcular posiciones
        x1 = L1 * np.cos(angulo1)
        y1 = L1 * np.sin(angulo1)
        x2 = x1 + L2 * np.cos(angulo1 + angulo2)
        y2 = y1 + L2 * np.sin(angulo1 + angulo2)
        
        # Actualizar posiciones
        line1.set_data([0, x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        punto_final.set_data([0, x1, x2], [0, y1, y2])
        
        ax.set_title(f'Brazo Robótico: θ1 = {np.rad2deg(angulo1):.1f}°, θ2 = {np.rad2deg(angulo2):.1f}°')
        return line1, line2, punto_final

    # Crear la animación
    anim = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True,
                        interval=(n/20 if int(n/20)>0 else n), repeat=True)
    
    plt.show()
    return anim

# Ejecutar la animación
anim = crear_animacion_brazo_con_conos(360)