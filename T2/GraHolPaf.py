"""
Dado un brazo (2D) con 2 dof, de dos eslabones (L1, angulo1 y L2, angulo2), con un punto fijo en el origen.
Restringido a una recta y = mx + b.
El objetivo es graficar las restricciones holonomicas y pfaffianas de un brazo de dos eslabones.

Holonomicas: g(01,02) = 0 ; y - mx -b = 0
01 = [...] de 30º a 30º -> Salida en RADINS for 1 a 360
02 => g(01,02) = 0, puede dar dos soluciones para cada angulo de 01
Arrays de 01 y 02 a representar en un grafico
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from matplotlib.patches import Wedge

# Funcion de restriccion
def restriccion(angulo2, angulo1, L1, L2, m, b):
    x = L1 * np.cos(angulo1) + L2 * np.cos(angulo1 + angulo2)
    y = L1 * np.sin(angulo1) + L2 * np.sin(angulo1 + angulo2)
    return y - m * x - b

# Función para generar ángulos1
def angulo1_generar(n=36, min_ang=0, max_ang=360):
    angulos_grados = np.linspace(min_ang, max_ang, n)
    return np.deg2rad(angulos_grados)

# Función para obtener los ángulos2 que cumplen la restricción
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
    
    print(f"\033[32mSe encontraron {len(angulo1_filtrado)}x{len(soluciones_reales)} parejas θ1xθ2.\033[0m")
    return angulo1_filtrado, soluciones_reales

# Función para identificar rangos continuos de ángulos
def identificar_rangos_angulos(angulos_grados):
    """
    Identifica rangos continuos de ángulos y los agrupa automáticamente.
    
    Args:
        angulos_grados: Array o lista de ángulos en grados
        
    Returns:
        lista de tuplas (min_angulo, max_angulo) para cada rango identificado
    """
    # Convertir a lista si es un array de numpy
    if isinstance(angulos_grados, np.ndarray):
        angulos_grados = angulos_grados.tolist()
    
    if len(angulos_grados) == 0:
        return []
    
    # Ordenar los ángulos
    angulos_ordenados = sorted(angulos_grados)
    
    # Inicializar variables
    rangos = []
    inicio_rango = angulos_ordenados[0]
    angulo_anterior = angulos_ordenados[0]
    
    # Umbral para determinar discontinuidad (en grados)
    umbral_discontinuidad = 20  # Ajustar según necesidad
    
    # Encontrar los rangos
    for i in range(1, len(angulos_ordenados)):
        angulo_actual = angulos_ordenados[i]
        
        # Si hay una discontinuidad significativa
        if angulo_actual - angulo_anterior > umbral_discontinuidad:
            # Guardar el rango anterior
            rangos.append((inicio_rango, angulo_anterior))
            # Iniciar nuevo rango
            inicio_rango = angulo_actual
        
        angulo_anterior = angulo_actual
    
    # Agregar el último rango
    rangos.append((inicio_rango, angulo_anterior))
    
    return rangos

# Función para crear animación del braz
def crear_animacion_brazo_con_conos_automatico(n):
    # Obtener soluciones
    angulo1_array = angulo1_generar(n)
    angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)
    # Convertir a grados para análisis
    angulos_grados = np.rad2deg(angulo1_filtrado)
    
    # Identificar rangos automáticamente
    rangos_angulos = identificar_rangos_angulos(angulos_grados)
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Colores para los diferentes rangos
    colores = ['green', 'blue', 'purple', 'orange', 'cyan']
    
    # Agregar los conos (wedges) para cada rango identificado
    for i, (min_ang, max_ang) in enumerate(rangos_angulos):
        color = colores[i % len(colores)]
        wedge = Wedge((0, 0), L1 + L2, min_ang, max_ang, 
                      alpha=0.2, color=color, 
                      label=f'Rango θ1 ({min_ang:.1f}° a {max_ang:.1f}°)')
        ax.add_patch(wedge)
    
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
        punto_final.set_data([x2], [y2])  # Solo el punto final
        
        ax.set_title(f'Brazo Robótico: θ1 = {np.rad2deg(angulo1):.1f}°, θ2 = {np.rad2deg(angulo2):.1f}°')
        return line1, line2, punto_final

    # Crear la animación
    anim = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True,
                        interval=(n//20 if n//20 > 0 else n), repeat=True)
    
    plt.show()
    return anim

# Función para graficar el brazo robótico
def graficar_brazo(angulo1, angulo2, L1, L2, m, b):
    """Grafica el brazo y la restricción para un par de ángulos."""
    plt.figure(figsize=(10, 8))
    
    # Posiciones de las articulaciones
    x1, y1 = L1 * np.cos(angulo1), L1 * np.sin(angulo1)
    x2, y2 = x1 + L2 * np.cos(angulo1 + angulo2), y1 + L2 * np.sin(angulo1 + angulo2)
    
    # Graficar brazo
    plt.plot([0, x1, x2], [0, y1, y2], 'ro-', linewidth=3, markersize=8, label='Brazo')
    
    # Graficar restricción (recta)
    x_recta = np.linspace(-1.5, 1.5, 100)
    plt.plot(x_recta, m * x_recta + b, 'b--', label=f'Restricción: y = {m}x + {b}')
    
    # Configuración del gráfico
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Brazo Robótico: θ1 = {np.rad2deg(angulo1):.1f}°, θ2 = {np.rad2deg(angulo2):.1f}°')
    plt.legend()
    plt.show()

# Graficar los resultados
def graficar_resultados(angulo1_filtrado, angulo2_reales):
    """Grafica los ángulos obtenidos"""
    # Expandir los arrays para que coincidan
    x_coords = []
    y_coords = []
    
    # Expandir las soluciones para graficar
    for i, angulo1 in enumerate(angulo1_filtrado):
        for angulo2 in angulo2_reales[i]:
            x_coords.append(angulo1)
            y_coords.append(angulo2)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, color='b', marker='o')
    
    plt.xlabel("Ángulo 1 (θ1 en radianes)")
    plt.ylabel("Ángulo 2 (θ2 en radianes)")
    plt.title("Soluciones reales de θ2 para distintos valores de θ1")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True)
    
    # Agregar leyenda
    plt.legend(['Solución real'])
    
    # Preguntar si desea guardar el gráfico
    guardar = input("¿Desea guardar el gráfico? (s/n): ")
    if guardar.lower() == 's':
        plt.savefig('holonomica_opt.png')
    
    plt.show()
    
def menu():
    """Menú interactivo para seleccionar acciones."""
    while True:
        print("\nMenú de Opciones:")
        print("Nota: El número de ángulos a generar no se corresponde con el número final de ángulos encontrados (sols en i / inexistentes).")
        print("1. Visualizar los 2 primeros angulos encontrados")
        print("2. Crear animación del brazo con conos automáticamente")
        print("3. Ver lista de soluciones")
        print("0. Salir")
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            n = int(input("Ingrese el número de ángulos a generar: "))
            angulo1_array = angulo1_generar(n)
            angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)
            
            count = 0
            for i in range(len(angulo1_filtrado)):
                for angulo2 in angulo2_reales[i]:
                    graficar_brazo(angulo1_filtrado[i], angulo2, L1, L2, m, b)
                    count += 1
                    if count >= 2:  # Break after showing 2 angles
                        return
        
        elif opcion == "2":
            n = int(input("Ingrese el número de ángulos a generar: "))
            anim = crear_animacion_brazo_con_conos_automatico(n)
            print("Iniciando animación...")
       
        elif opcion == "3":
            n = int(input("Ingrese el número de ángulos a generar: "))
            angulo1_array = angulo1_generar(n)
            print("Buscando soluciones...")
            
            import time
            start_time = time.time()
            
            angulo1_filtrado, angulo2_reales = angulo2_obtener(m, b, L1, L2, angulo1_array)
            
            print(f"Cálculo completado en {time.time() - start_time:.2f} segundos")
            
            for i in range(len(angulo1_filtrado)):
                for angulo2 in angulo2_reales[i]:
                    print(f"θ1: {angulo1_filtrado[i]:.4f}, θ2: {angulo2:.4f}")
            
            graficar_resultados(angulo1_filtrado, angulo2_reales)
        
        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

if __name__ == "__main__":
    # Parámetros de la simulación
    L1 = 1; L2 = 0.2; m = 0.5; b = 0.2
    menu()