"""
Ejemplos de uso para la visualización del robot manipulador
"""

from class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite
from class_robot_plotter import plot_robot
import numpy as np
import matplotlib.pyplot as plt
from problema_cinematico_inverso_gen import CinematicaInversa
from class_rotaciones import Rp2Trans, Euler2R
from class_helicoidales import calcular_M_generalizado

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

# Ejemplo 6: Cinemática directa con configuración específica
def ejemplo_cinematica_directa():
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Configuración personalizada (ejemplo)
    thetas = [0.5, -0.5, 0.7, 1.0, 0.5, 0.05, 1.2]
    thetas = thetas_limite(robot, thetas)
    
    # Visualizar el robot
    print(f"Visualizando robot con cinemática directa: {np.round(thetas, 3)}")
    plot_robot(robot, thetas)

# Ejemplo 7: Cinemática inversa con trayectoria circular
def ejemplo_cinematica_inversa_circular():
    # Cargar el robot y matriz M
    robot = cargar_robot_desde_yaml("robot.yaml")
    M = calcular_M_generalizado(robot)
    
    # Generar puntos en una trayectoria circular
    num_puntos = 30
    radio = 0.15
    z = 0.3
    angulos = np.linspace(0, 2*np.pi, num_puntos)
    puntos = np.array([[radio*np.cos(theta), radio*np.sin(theta), z] for theta in angulos])
    
    # Calcular configuraciones articulares para cada punto
    thetas_anim = []
    for punto in puntos:
        # Orientación fija (ejemplo: orientación hacia abajo)
        Tsd = Rp2Trans(Euler2R(0, np.pi, 0), punto)
        thetas_follower = CinematicaInversa(robot, p_xyz=punto, RPY=[0, np.pi, 0])
        if thetas_follower:
            thetas_anim.extend(thetas_follower)  # Usar todas las iteraciones
            
    # Suavizar trayectoria si es necesario
    if len(thetas_anim) < 100:
        print("Aplicando interpolación para suavizar...")
        from scipy.interpolate import CubicSpline
        t_original = np.linspace(0, 1, len(thetas_anim))
        t_nuevo = np.linspace(0, 1, 100)
        thetas_anim = CubicSpline(t_original, thetas_anim, axis=0)(t_nuevo)
    
    # Visualizar y guardar animación
    print("Animando trayectoria circular...")
    fig, ax, anim = plot_robot(robot, thetas_anim, animation_speed=50, show=False)
    anim.save("trayectoria_circular.mp4", writer="ffmpeg", fps=30)
    plt.close()

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
    
    print("\n6. Ejemplo de cinemática directa")
    ejemplo_cinematica_directa()
    
    print("\n7. Ejemplo de cinemática inversa con trayectoria circular")
    ejemplo_cinematica_inversa_circular()