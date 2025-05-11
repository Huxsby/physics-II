"""
Ejemplos de uso para la visualización del robot manipulador
"""

from class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite
from class_robot_plotter import plot_robot
import numpy as np
import matplotlib.pyplot as plt

# Ejemplo 1: Visualización simple
def ejemplo_visualizacion_simple():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Generar una configuración de ángulos en posición neutral
    thetas = np.zeros(len(robot.links))
    
    # Visualizar el robot
    print("Visualizando robot en posición neutral...")
    plot_robot(robot, thetas)

# Ejemplo 2: Visualización con configuración personalizada
def ejemplo_configuracion_personalizada():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Configuración personalizada (ajustar según los límites de las articulaciones)
    thetas = [0.5, -0.5, 0.7, 1.0, 0.5, 0.05, 1.2]
    
    # Asegurar que los ángulos estén dentro de los límites
    thetas = thetas_limite(robot, thetas)
    
    # Visualizar el robot
    print(f"Visualizando robot con configuración personalizada: {np.round(thetas, 3)}")
    plot_robot(robot, thetas)

# Ejemplo 3: Visualización con múltiples vistas
def ejemplo_multiples_vistas():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Generar una configuración aleatoria
    thetas, _ = thetas_aleatorias(robot)
    
    # Crear figura con múltiples vistas
    fig = plt.figure(figsize=(15, 10))
    
    # Vista frontal
    ax1 = fig.add_subplot(221, projection='3d')
    plot_robot(robot, thetas, ax=ax1, show=False, view_angles=(0, 0))
    ax1.set_title('Vista Frontal')
    
    # Vista lateral
    ax2 = fig.add_subplot(222, projection='3d')
    plot_robot(robot, thetas, ax=ax2, show=False, view_angles=(0, 90))
    ax2.set_title('Vista Lateral')
    
    # Vista superior
    ax3 = fig.add_subplot(223, projection='3d')
    plot_robot(robot, thetas, ax=ax3, show=False, view_angles=(90, 0))
    ax3.set_title('Vista Superior')
    
    # Vista isométrica
    ax4 = fig.add_subplot(224, projection='3d')
    plot_robot(robot, thetas, ax=ax4, show=False, view_angles=(30, 45))
    ax4.set_title('Vista Isométrica')
    
    plt.tight_layout()
    plt.show()

# Ejemplo 4: Animación entre dos configuraciones
def ejemplo_animacion():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Generar dos configuraciones aleatorias
    thetas_inicio, _ = thetas_aleatorias(robot)
    thetas_fin, _ = thetas_aleatorias(robot)
    
    print(f"Configuración inicial: {np.round(thetas_inicio, 3)}")
    print(f"Configuración final: {np.round(thetas_fin, 3)}")
    
    # Generar frames para la animación
    num_frames = 50
    thetas_anim = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        thetas_i = thetas_inicio * (1 - t) + thetas_fin * t
        thetas_anim.append(thetas_i)
    
    # Visualizar la animación
    print(f"Animando movimiento con {num_frames} frames...")
    plot_robot(robot, thetas_anim, animation_speed=50)

# Ejemplo 5: Visualización con trayectoria
def ejemplo_trayectoria():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Generar múltiples configuraciones aleatorias
    num_points = 5
    all_thetas = []
    
    for _ in range(num_points):
        thetas, _ = thetas_aleatorias(robot)
        all_thetas.append(thetas)
    
    # Crear una trayectoria más suave entre los puntos
    num_frames = 100
    trayectoria = []
    
    for i in range(num_points-1):
        start_thetas = all_thetas[i]
        end_thetas = all_thetas[i+1]
        
        # Generar puntos intermedios
        frames_segment = num_frames // (num_points - 1)
        for j in range(frames_segment):
            t = j / frames_segment
            thetas_j = start_thetas * (1 - t) + end_thetas * t
            trayectoria.append(thetas_j)
    
    # Visualizar la trayectoria
    print(f"Animando trayectoria con {len(trayectoria)} frames...")
    plot_robot(robot, trayectoria, animation_speed=20)

if __name__ == "__main__":
    print("\n1. Ejemplo de visualización simple")
    ejemplo_visualizacion_simple()
    
    print("\n2. Ejemplo de configuración personalizada")
    ejemplo_configuracion_personalizada()
    
    print("\n3. Ejemplo de múltiples vistas")
    ejemplo_multiples_vistas()
    
    print("\n4. Ejemplo de animación entre dos configuraciones")
    ejemplo_animacion()
    
    print("\n5. Ejemplo de trayectoria con múltiples puntos")
    ejemplo_trayectoria()