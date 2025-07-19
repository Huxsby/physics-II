import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from core import Robot, cargar_robot_desde_yaml

def fabrik_steps_3d(positions, lengths, target, tolerance=1e-3, max_iter=100):
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

    for iter_num in range(max_iter):
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
        pos = steps[frame]
        line.set_data(pos[:,0], pos[:,1])
        line.set_3d_properties(pos[:,2])
        return line,

    ani = FuncAnimation(fig, update, frames=len(steps), interval=500, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    """ Prueba de FABRIK 3D """
    # Inicializar posiciones y longitudes
    lengths = [0.2, 0.15, 0.1]
    positions = [np.array([0.0, 0.0, 0.0])]
    for l in lengths:
        positions.append(positions[-1] + np.array([l, 0.0, 0.0]))
    positions = np.array(positions)

    # Definir objetivo y calcular pasos FABRIK
    target = np.array([0.25, 0.25, 0.25])
    steps = fabrik_steps_3d(positions, lengths, target)

    # Animar FABRIK
    animate_fabrik(steps, target, positions)

    """ Prueba de carga de robot y posiciones """
    # Cargar robot y procesar posiciones
    robot = cargar_robot_desde_yaml('config/robot.yaml')
    r_lengths = [link.length for link in robot.links]

    # Convertir las coordenadas relativas de los joints a coordenadas absolutas
    r_positions = [np.array([0.0, 0.0, 0.0])]
    for link in robot.links:
        r_positions.append(r_positions[-1] + np.array(link.joint_coords))
    r_positions = np.array(r_positions)


    # Definir objetivo y calcular pasos FABRIK
    target = np.array([0.25, 0.25, 0.25])
    r_steps = fabrik_steps_3d(r_positions, r_lengths, target)

    # Animar FABRIK
    animate_fabrik(r_steps, target, r_positions)
    for step in steps:
        print(step)
    
    for step in r_steps:
        print(step)