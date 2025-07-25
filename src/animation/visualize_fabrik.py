
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.class_robot_structure import Robot

def visualizar_iteraciones_fabrik(robot: Robot, p_history: list, p_xyz_objetivo: list, save_gif: bool = False):
    """
    Crea una animación 3D de las iteraciones del algoritmo FABRIK.

    Args:
        robot (Robot): El objeto robot.
        p_history (list): Una lista de listas, donde cada lista interna contiene las 
                          posiciones de las articulaciones [p0, p1, ..., pn] para una iteración.
        p_xyz_objetivo (list): La coordenada [x, y, z] del punto objetivo.
        save_gif (bool): Si es True, guarda la animación como un GIF.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convertir el objetivo a un array de numpy
    p_xyz_objetivo = np.array(p_xyz_objetivo)

    # Determinar los límites del gráfico a partir de todas las posiciones históricas
    all_points = np.vstack([item for sublist in p_history for item in sublist])
    all_points = np.vstack([all_points, p_xyz_objetivo]) # Incluir el objetivo
    
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0

    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Función que se llama en cada frame de la animación
    def update(frame):
        ax.cla() # Limpiar el eje para el nuevo frame
        
        # Obtener las posiciones de las articulaciones para la iteración actual
        p_actual = p_history[frame]
        p_actual = np.array(p_actual)

        # Dibujar los eslabones del robot
        for i in range(len(p_actual) - 1):
            ax.plot([p_actual[i][0], p_actual[i+1][0]], 
                    [p_actual[i][1], p_actual[i+1][1]], 
                    [p_actual[i][2], p_actual[i+1][2]], 'b-', linewidth=3)

        # Dibujar las articulaciones
        ax.scatter(p_actual[:, 0], p_actual[:, 1], p_actual[:, 2], c='r', s=50, label='Articulaciones')
        
        # Dibujar la base y el efector final
        ax.scatter(p_actual[0][0], p_actual[0][1], p_actual[0][2], c='k', s=100, marker='s', label='Base')
        ax.scatter(p_actual[-1][0], p_actual[-1][1], p_actual[-1][2], c='g', s=100, marker='*', label='Efector Final')

        # Dibujar el punto objetivo
        ax.scatter(p_xyz_objetivo[0], p_xyz_objetivo[1], p_xyz_objetivo[2], c='m', s=150, marker='x', label='Objetivo')

        # Configuración del gráfico
        ax.set_title(f'Iteración FABRIK: {frame + 1}/{len(p_history)}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.grid(True)
        
        # Mantener los mismos límites en todos los frames
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Crear la animación
    anim = FuncAnimation(fig, update, frames=len(p_history), interval=500, repeat=False)

    if save_gif:
        print("Guardando animación como 'fabrik_animation.gif'...")
        anim.save('output/animations/fabrik_animation.gif', writer='pillow', fps=2)
        print("Animación guardada.")

    plt.show()
