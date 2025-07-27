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
    
    dist_total = sum(lengths)
    dist_objetivo = np.linalg.norm(target - base)

    if dist_objetivo > dist_total:
        # El objetivo está fuera de alcance, estirar el brazo hacia él
        for i in range(n - 1):
            # Vector unitario desde la articulación anterior hacia el objetivo
            direction = (target - positions[i]) / np.linalg.norm(target - positions[i])
            positions[i+1] = positions[i] + direction * lengths[i]
        steps.append(positions.copy())
        return steps

    iteraciones = 0
    while np.linalg.norm(positions[-1] - target) > tolerance and iteraciones < max_iter:
        # Forward pass (hacia adelante): desde el efector final a la base
        positions[-1] = target
        for i in reversed(range(n-1)):
            # Vector unitario desde la nueva posición de la articulación i+1 a la antigua i
            direction = (positions[i] - positions[i+1]) / np.linalg.norm(positions[i] - positions[i+1])
            # La nueva posición de i es la de i+1 más el vector unitario escalado por la longitud
            positions[i] = positions[i+1] + direction * lengths[i]
        
        # Backward pass (hacia atrás): desde la base al efector final
        positions[0] = base
        for i in range(n-1):
            # Vector unitario desde la nueva posición de la articulación i a la antigua i+1
            direction = (positions[i+1] - positions[i]) / np.linalg.norm(positions[i+1] - positions[i])
            # La nueva posición de i+1 es la de i más el vector unitario escalado por la longitud
            positions[i+1] = positions[i] + direction * lengths[i]
        
        steps.append(positions.copy())
        iteraciones += 1

    print(f"\t\033[92mFABRIK convergió en {iteraciones} iteraciones. Tiempo: {time.time()-t:.4f}s\033[0m")
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

def aplicar_restricciones(robot, positions, aplicar_rangos=False):
    """
    Aplica restricciones a las posiciones calculadas por FABRIK según los límites y tipos de articulaciones del robot.

    Args:
        robot (Robot): Objeto Robot que contiene la estructura del robot.
        positions (np.ndarray): Posiciones calculadas por FABRIK.
        aplicar_rangos (bool): Indica si se deben aplicar los límites de los ángulos. Por defecto es True.

    Returns:
        np.ndarray: Posiciones ajustadas para cumplir con las restricciones.
    """
    print("\t\033[92mAplicando restricciones a las posiciones...\033[0m")
    for i, link in enumerate(robot.links):
        eje_giro = link.orientation  # Obtener el eje de giro real del eslabón
        # eje_giro = eje_giro / np.linalg.norm(eje_giro)  # Asegurar que el eje está normalizado
        print(f"Eje de giro (link {i}): {eje_giro}")

        if link.tipo == "revolute":
            # Calcular el vector del eslabón actual
            vector = positions[i + 1] - positions[i]
            # Normalizar el vector para evitar errores de escala
            vector_normalizado = vector / np.linalg.norm(vector)
            print(f"Vector normalizado (link {i}): {vector_normalizado}")

            # Calcular el ángulo entre el vector y el eje de giro
            theta = np.arccos(np.clip(np.dot(vector_normalizado, eje_giro), -1.0, 1.0))
            print(f"Ángulo calculado (link {i}): {theta}")

            if aplicar_rangos:
                # Verificar si el ángulo está dentro de los límites
                if link.joint_limits[0] <= theta <= link.joint_limits[1]:
                    print(f"Ángulo dentro de los límites (link {i}): {theta}")
                else:
                    # Ajustar el ángulo para que esté dentro de los límites
                    theta = np.clip(theta, link.joint_limits[0], link.joint_limits[1])
                    print(f"Ángulo ajustado (link {i}): {theta}")

                    # Recalcular el vector basado en el ángulo ajustado
                    r = np.linalg.norm(vector)
                    nuevo_vector = r * (np.cos(theta) * eje_giro + np.sin(theta) * np.cross(eje_giro, vector_normalizado))
                    nuevo_vector = nuevo_vector / np.linalg.norm(nuevo_vector) * r  # Asegurar que la longitud se mantiene
                    positions[i + 1] = positions[i] + nuevo_vector
                    print(f"Nueva posición ajustada (link {i + 1}): {positions[i + 1]}")

        elif link.tipo == "prismatic":
            # Restringir la distancia dentro de los límites
            vector = positions[i + 1] - positions[i]
            distancia = np.linalg.norm(vector)
            print(f"Distancia calculada (link {i}): {distancia}")

            distancia = np.clip(distancia, link.joint_limits[0], link.joint_limits[1])
            print(f"Distancia restringida (link {i}): {distancia}")

            # Recalcular la posición basada en la distancia restringida
            positions[i + 1] = positions[i] + (vector / np.linalg.norm(vector)) * distancia
            print(f"Nueva posición (link {i + 1}): {positions[i + 1]}")

    return positions

def fabrik_steps_3d_con_restricciones(robot, positions, lengths, target, tolerance=1e-3, max_iter=100):
    """
    Calcula los pasos del algoritmo FABRIK para una cadena de eslabones en 3D con restricciones.

    Args:
        robot (Robot): Objeto Robot que contiene la estructura del robot.
        positions (np.ndarray): Posiciones iniciales de los eslabones.
        lengths (list of float): Longitudes de los eslabones.
        target (np.ndarray): Objetivo a alcanzar.
        tolerance (float): Tolerancia para detener el algoritmo.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        list of np.ndarray: Lista de pasos calculados por FABRIK.
    """
    t = time.time()
    n = len(positions)
    base = positions[0].copy()
    positions = positions.copy()
    steps = [positions.copy()]

    dist_total = sum(lengths)
    dist_objetivo = np.linalg.norm(target - base)

    if dist_objetivo > dist_total:
        # El objetivo está fuera de alcance, estirar el brazo hacia él
        for i in range(n - 1):
            direction = (target - positions[i]) / np.linalg.norm(target - positions[i])
            positions[i+1] = positions[i] + direction * lengths[i]
        steps.append(positions.copy())
        return steps

    iteraciones = 0
    while np.linalg.norm(positions[-1] - target) > tolerance and iteraciones < max_iter:
        # Forward pass (hacia adelante): desde el efector final a la base
        positions[-1] = target
        for i in reversed(range(n - 1)):
            direction = (positions[i] - positions[i+1]) / np.linalg.norm(positions[i] - positions[i+1])
            positions[i] = positions[i+1] + direction * lengths[i]
        
        # Aplicar restricciones después del forward pass
        positions = aplicar_restricciones(robot, positions)

        # Backward pass (hacia atrás): desde la base al efector final
        positions[0] = base
        for i in range(n - 1):
            direction = (positions[i+1] - positions[i]) / np.linalg.norm(positions[i+1] - positions[i])
            positions[i+1] = positions[i] + direction * lengths[i]

        # Aplicar restricciones después del backward pass
        positions = aplicar_restricciones(robot, positions)

        steps.append(positions.copy())
        iteraciones += 1
    
    print(f"\t\033[92mFABRIK con restricciones convergió en {iteraciones} iteraciones. Tiempo: {time.time() - t:.4f}s\033[0m")
    return steps

def main():
    """
    Función principal para manejar las opciones del programa usando un menú con bucle while True.
    """
    while True:
        print("\nSeleccione una opción:")
        print("1. Ejecutar FABRIK básico")
        print("2. Ejecutar FABRIK con restricciones")
        print("3. Depurar ejes de giro")
        print("4. Generar trayectoria circular")
        print("5. Salir")

        opcion = input("Ingrese el número de la opción deseada: ")

        if opcion == "1":  # Ejecutar FABRIK básico
            target = np.array([0.25, 0.25, 0.25])
            lengths = [0.2, 0.15, 0.1]
            positions = [np.array([0.0, 0.0, 0.0])]
            for l in lengths:
                positions.append(positions[-1] + np.array([l, 0.0, 0.0]))
            positions = np.array(positions)

            steps = fabrik_steps_3d(positions, lengths, target)
            ax, fig, anim = animate_fabrik(steps, target, positions)
            guardar_animacion(anim, "output/animations/fabrik_test_animation_vector_assemble", fps=2)

        elif opcion == "2":  # Ejecutar FABRIK con restricciones
            robot = cargar_robot_desde_yaml('config/robot.yaml')
            r_lengths = [link.length for link in robot.links]
            r_positions = [np.array([0.0, 0.0, 0.0])]
            for link in robot.links:
                r_positions.append(r_positions[-1] + np.array(link.joint_coords))
            r_positions = np.array(r_positions)

            target = np.array([0.25, 0.25, 0.25])
            steps_with_restrictions = fabrik_steps_3d_con_restricciones(robot, r_positions, r_lengths, target)
            ax, fig, anim = animate_fabrik(steps_with_restrictions, target, r_positions)
            guardar_animacion(anim, "output/animations/fabrik_test_animation_restricciones", fps=2)

        elif opcion == "3":  # Depurar ejes de giro
            robot = cargar_robot_desde_yaml('config/robot.yaml')
            print("\nObtener_eje_de_giro")
            for i in range(robot.num_links):
                robot.links[i].obtener_eje_de_giro()

        elif opcion == "4":  # Generar trayectoria circular
            center = np.array([0.2, 0.2, 0.0])
            radius = 0.1
            normal = np.array([0.0, 0.0, 1.0])
            num_points = 50

            circular_trajectory = trayectoria_circular(center, radius, normal, num_points)
            lengths = [0.2, 0.15, 0.1]
            positions = [np.array([0.0, 0.0, 0.0])]
            for l in lengths:
                positions.append(positions[-1] + np.array([l, 0.0, 0.0]))
            positions = np.array(positions)

            circular_steps = []
            for target in circular_trajectory:
                steps = fabrik_steps_3d(positions, lengths, target)
                circular_steps.append(steps[-1])

            ax, fig, anim = animate_fabrik(circular_steps, center, positions)
            guardar_animacion(anim, "output/animations/fabrik_circular_leg_animation", fps=10)

        elif opcion == "5":  # Salir del programa
            print("Saliendo del programa.")
            break

        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()