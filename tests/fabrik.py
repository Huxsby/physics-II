import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
from core import Robot, cargar_robot_desde_yaml, filtrar_configuraciones, str_config
from animation import plot_robot, guardar_animacion

"""
Módulo para implementar el algoritmo FABRIK (Forward and Backward Reaching Inverse Kinematics)

Objetivos marcados en el paper original:
- [X] Implementar un algoritmo de cinemática inversa que sea eficiente y fácil de implementar.
    - [ ] Implmentar restricciones.
    - [ ] Que respecte el tipo de articulación y los rangos de movimiento.
    - [ ] Provar que el paso de steps a configuraciones es correcto. 
- [X] Calcular los pasos de una cadena de eslabones en 3D para alcanzar un objetivo dado.
- [X] Proporcionar una visualización animada de los pasos del algoritmo FABRIK.

Extras:
- [X] Calcular los ángulos de las articulaciones del robot a partir de los pasos de posiciones.
"""


def fabrik_steps_3d(positions, lengths, target, tolerance=1e-3, max_iter=100):
    """ Calcula los pasos del algoritmo FABRIK para una cadena de eslabones en 3D. """
    t = time.time()
    n = len(positions)
    base = positions[0].copy()
    positions = positions.copy()
    steps = [positions.copy()]
    if np.linalg.norm(target - base) > sum(lengths):
        for i in range(1, n):
            r = np.linalg.norm(target - positions[i-1])
            positions[i] = positions[i-1] + (target - positions[i-1]) * (lengths[i-1]/r)
        steps.append(positions.copy())
        return steps

    for _ in range(max_iter):
        # Forward
        positions[-1] = target
        for i in reversed(range(n-1)):
            r = np.linalg.norm(positions[i+1] - positions[i])
            positions[i] = positions[i+1] + (positions[i] - positions[i+1]) * (lengths[i]/r)
        # Backward
        positions[0] = base
        for i in range(n-1):
            r = np.linalg.norm(positions[i+1] - positions[i])
            positions[i+1] = positions[i] + (positions[i+1] - positions[i]) * (lengths[i]/r)
        steps.append(positions.copy())
        if np.linalg.norm(positions[-1] - target) < tolerance:
            break

    print("\t\033[92mTiempo de calculo de la CI en FABRIK (fabrik_steps_3d): ", time.time()-t, "s\033[0m")
    return steps
    
def animate_fabrik(steps, target, initial_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target[0], target[1], target[2], c='g', marker='x', label='Objetivo')
    line, = ax.plot([], [], [], 'ro-', label='FABRIK')
    ax.plot(initial_positions[:,0], initial_positions[:,1], initial_positions[:,2], 'o--', label='Inicial')
    ax.legend()
    ax.set_title('FABRIK Cinemática Inversa 3D')

    def update(frame):
        """Actualiza la posición de la línea en la animación para el frame dado."""
        pos = steps[frame]  # Obtener las posiciones del frame actual
        line.set_data(pos[:, 0], pos[:, 1])  # Actualizar datos en el plano XY
        line.set_3d_properties(pos[:, 2])  # Actualizar datos en el eje Z
        return line,

    # Crear la animación utilizando FuncAnimation
    anim = FuncAnimation(
        fig,  # Figura donde se dibuja la animación
        update,  # Función de actualización para cada frame
        frames=len(steps),  # Número total de frames
        interval=500,  # Intervalo entre frames en milisegundos
        blit=True,  # Optimización para redibujar solo las partes necesarias
        repeat=True  # Repetir la animación automáticamente
    )

    plt.show() # Mostrar la animación
    return ax, fig, anim


def steps_to_angles(steps, lengths):
    """
    Convierte los pasos de posiciones en configuraciones de ángulos en radianes.
    
    Nota: Hasta que no se implementen restricciones en el algoritmo usado los ejes de rotación no coinciden con los ejes del robot.
    
    Args:
        steps (list of np.ndarray): Lista de pasos de posiciones calculados por FABRIK.
        lengths (list of float): Longitudes de los eslabones.

    Returns:
        list of list of float: Lista de configuraciones en radianes para cada paso.
    """
    angles_list = []
    for step in steps:
        angles = []
        for i in range(1, len(step)):
            # Vector del eslabón actual
            link_vector = step[i] - step[i - 1]
            # Ángulo en el plano XY
            theta = np.arctan2(link_vector[1], link_vector[0])
            # Ángulo en el plano XZ
            phi = np.arctan2(link_vector[2], np.linalg.norm(link_vector[:2]))
            angles.append((theta, phi))
        angles_list.append(angles)
    return angles_list

def calcular_angulos_robot(robot, fabrik_steps):
    """
    Calcula los ángulos de las articulaciones del robot a partir de los pasos de FABRIK.

    Args:
        robot (Robot): Objeto Robot que contiene la estructura del robot.
        fabrik_steps (list of np.ndarray): Lista de pasos de posiciones calculados por FABRIK.

    Returns:
        list of list of float: Lista de configuraciones angulares para cada paso.
    """
    angulos_robot = []
    for step in fabrik_steps:
        angulos = []
        for i, link in enumerate(robot.links):
            if link.tipo == "revolute":
                # Calcular el ángulo en el plano XY para articulaciones revolutas
                vector = step[i + 1] - step[i]
                theta = np.arctan2(vector[1], vector[0])
                angulos.append(theta)
            elif link.tipo == "prismatic":
                # Calcular la distancia para articulaciones prismáticas
                vector = step[i + 1] - step[i]
                distancia = np.linalg.norm(vector)
                angulos.append(distancia)
        angulos_robot.append(angulos)
    return angulos_robot

def trayectoria_circular(center, radius, normal, num_points):
    """
    Genera una trayectoria circular en 3D.

    Args:
        center (np.ndarray): Centro del círculo.
        radius (float): Radio del círculo.
        normal (np.ndarray): Vector normal al plano del círculo.
        num_points (int): Número de puntos en la trayectoria.

    Returns:
        np.ndarray: Puntos de la trayectoria circular.
    """
    normal = normal / np.linalg.norm(normal)
    v1 = np.array([-normal[1], normal[0], 0])
    if np.linalg.norm(v1) == 0:
        v1 = np.array([1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [center + radius * (np.cos(angle) * v1 + np.sin(angle) * v2) for angle in angles]
    return np.array(points)

if __name__ == "__main__":
    """ Prueba de FABRIK 3D """
    target = np.array([0.25, 0.25, 0.25]) # Definir objetivo

    lengths = [0.2, 0.15, 0.1]  # Inicializar posiciones y longitudes
    positions = [np.array([0.0, 0.0, 0.0])]
    for l in lengths:
        positions.append(positions[-1] + np.array([l, 0.0, 0.0]))
    positions = np.array(positions)

    steps = fabrik_steps_3d(positions, lengths, target)  # Calcular pasos FABRIK

    ax, fig, anim = animate_fabrik(steps, target, positions) # Animar FABRIK
    guardar_animacion(anim, "output/animations/fabrik_test_animation_vector_assemble", fps=2)  # Guardar animación

    """ Prueba de carga de robot y posiciones """
    # Cargar robot y procesar posiciones
    robot = cargar_robot_desde_yaml('config/robot.yaml')
    r_lengths = [link.length for link in robot.links]

    # Convertir las coordenadas relativas de los joints a coordenadas absolutas
    r_positions = [np.array([0.0, 0.0, 0.0])]
    for link in robot.links:
        r_positions.append(r_positions[-1] + np.array(link.joint_coords))
    r_positions = np.array(r_positions)

    r_steps = fabrik_steps_3d(r_positions, r_lengths, target)   # Calcular pasos FABRIK

    ax, fig, anim = animate_fabrik(r_steps, target, r_positions)  # Animar FABRIK
    guardar_animacion(anim, "output/animations/fabrik_test_animation_robot_assemble", fps=2)  # Guardar animación
    # for step in steps:
    #     print(step)
    # for step in r_steps:
    #     print(step)

    config = steps_to_angles(steps, lengths)    # Convertir pasos a configuraciones en radianes
    print("Configuraciones en radianes (FABRIK):")
    print("\tCada paso contiene una lista de tuplas (theta, phi), donde:")
    print("\t- theta: Ángulo en el plano XY (horizontal)")
    print("\t- phi: Ángulo en el plano XZ (vertical)\n")
    for i, angles in enumerate(config):
        print(f"\tPaso {i:2d}: ", end="")
        for theta, phi in angles:
            print(f"({theta:8.4f}, {phi:8.4f}) ", end="")
        print()

    r_config = steps_to_angles(r_steps, r_lengths)

    # Calcular los ángulos del robot a partir de los pasos de FABRIK
    r_config_theta = calcular_angulos_robot(robot, r_steps)

    filtrar_configuraciones(robot, r_config_theta)

    # print("Ángulos del robot calculados a partir de FABRIK:")
    # for i, angulos in enumerate(r_config_theta):
    #     print(f"Paso {i:2d}: {str_config(angulos)}")

    # Crear y guardar una animación para visualizar r_config_theta
    fig, ax, anim = plot_robot(robot, r_config_theta)
    guardar_animacion(anim, "output/animations/fabrik_test_config_theta_animation")

    # Definir parámetros de la trayectoria circular
    center = np.array([0.2, 0.2, 0.0])  # Centro del círculo
    radius = 0.1  # Radio del círculo
    normal = np.array([0.0, 0.0, 1.0])  # Vector normal al plano del círculo
    num_points = 50  # Número de puntos en la trayectoria

    # Generar la trayectoria circular
    circular_trajectory = trayectoria_circular(center, radius, normal, num_points)

    # Calcular pasos FABRIK para cada punto de la trayectoria
    circular_steps = []
    for target in circular_trajectory:
        steps = fabrik_steps_3d(positions, lengths, target)
        circular_steps.append(steps[-1])

    # Animar la trayectoria circular
    ax, fig, anim = animate_fabrik(circular_steps, center, positions)

    # Guardar la animación
    guardar_animacion(anim, "output/animations/fabrik_circular_leg_animation", fps=10)