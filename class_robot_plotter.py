"""
robot_plotter.py
=====
Este módulo proporciona funciones para visualizar un robot manipulador en 3D,
utilizando Matplotlib. Permite visualizar la estructura del robot en una configuración
estática o crear animaciones de movimiento.

Funciones:
    plot_robot: Visualiza el robot en una configuración específica o crea una animación.
    calcular_transformaciones: Calcula las transformaciones para cada eslabón basadas en la cinemática.
    
Ejemplo de uso:
    >>> robot = cargar_robot_desde_yaml("robot.yaml")
    >>> thetas = [0, 0, 0, 0, 0, 0, 0]  # Una configuración estática
    >>> plot_robot(robot, thetas)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

def matriz_rotacion(eje, angulo):
    """
    Calcula la matriz de rotación alrededor de un eje dado.
    
    Args:
        eje (numpy.ndarray): Vector unitario del eje de rotación.
        angulo (float): Ángulo de rotación en radianes.
        
    Returns:
        numpy.ndarray: Matriz de rotación 3x3.
    """
    eje = np.array(eje)
    if np.linalg.norm(eje) < 1e-10:  # Si el eje es casi cero
        return np.eye(3)  # Devuelve la matriz identidad
        
    eje = eje / np.linalg.norm(eje)  # Normalizar el eje
    
    x, y, z = eje
    c = np.cos(angulo)
    s = np.sin(angulo)
    C = 1 - c
    
    R = np.array([
        [x*x*C + c, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])
    
    return R

def calcular_transformaciones(robot, thetas):
    """
    Calcula las transformaciones para cada eslabón del robot.
    
    Args:
        robot: Objeto Robot que contiene los eslabones.
        thetas (list): Lista de valores articulares.
        
    Returns:
        list: Lista de matrices de transformación homogénea 4x4 para cada eslabón.
    """
    n_links = robot.num_links
    T = np.eye(4)  # Transformación inicial (identidad)
    transformaciones = [T.copy()]  # Guardar la transformación inicial
    
    for i in range(n_links):
        link = robot.links[i]
        theta = thetas[i]
        
        # Crear una matriz de transformación homogénea local
        T_local = np.eye(4)
        
        # Para articulaciones prismáticas, aplicar desplazamiento en la dirección del eje
        if link.tipo == "prismatic":
            # Aplicar desplazamiento
            T_local[:3, 3] = np.array(link.joint_coords) + theta * np.array(link.joint_axis)
        else:  # Para articulaciones rotacionales
            # Primero aplicar la traslación para ubicar en el origen de la articulación
            T_local[:3, 3] = np.array(link.joint_coords)
            
            # Luego aplicar la rotación alrededor del eje de la articulación
            R = matriz_rotacion(link.joint_axis, theta)
            T_local[:3, :3] = R
        
        # Actualizar la transformación global
        T = T @ T_local
        transformaciones.append(T.copy())
    
    return transformaciones

def guardar_animacion(anim, nombre_archivo):
    """
    Guarda la animación en un archivo, intentando primero con ffmpeg y luego con Pillow si falla.
    El resultado final será un archivo nombre_archivo.mp4 (preferido) o .gif (si ffmpeg falla).
    """
    if not anim:
        print("No hay animación para guardar.")
        return

    print("Guardando animación...")
    
    try:
        _guardar_animacion_ffmpeg(anim, nombre_archivo)
    except Exception as e:
        print(f"\t\033[31mError al guardar con ffmpeg: {e}\033[0m")
        try:
            _guardar_animacion_pillow(anim, nombre_archivo)
        except Exception as e:
            print(f"\t\033[31mError al guardar con Pillow: {e}\033[0m")
            print(f"\t\033[31mNo se pudo guardar la animación '{nombre_archivo}'.\n\tAsegúrate de tener ffmpeg o Pillow instalado correctamente.\033[0m")

def _guardar_animacion_ffmpeg(anim, nombre_archivo):
    """Intenta guardar la animación como MP4 usando ffmpeg."""
    print("\tIntentando guardar la animación como MP4 con ffmpeg...")
    anim.save(f"{nombre_archivo}.mp4", writer="ffmpeg", fps=30, dpi=225)  # dpi=225 para 1080p
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.mp4' usando ffmpeg.\033[0m")

def _guardar_animacion_pillow(anim, nombre_archivo):
    """Intenta guardar la animación como GIF usando Pillow."""
    print("\tIntentando guardar la animación como GIF con Pillow...")
    anim.save(f"{nombre_archivo}.gif", writer="pillow", fps=30)
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.gif' usando Pillow.\033[0m")

def plot_robot(robot, thetas, ax=None, show=True, trayectoria=None, animation_speed=200, view_angles=None):
    """
    Visualiza un robot manipulador en 3D.
    
    Args:
        robot: Objeto Robot que contiene los eslabones.
        thetas: Lista de valores articulares o lista de listas para animación.
        ax (matplotlib.axes.Axes, optional): Ejes de matplotlib para dibujar. Si es None, se crea uno nuevo.
        show (bool, optional): Si es True, se muestra la figura. Si es False, se devuelve la figura y los ejes 
                                (y el objeto de animación si es una animación).
        animation_speed (int, optional): Velocidad de la animación en ms entre frames.
        view_angles (tuple, optional): Tupla (elevación, azimut) para la vista 3D.
        trayectoria (list or numpy.ndarray, optional): Un array de puntos (Nx3) que representan
                                                        una trayectoria a dibujar. Si se proporciona, 
                                                        se dibuja en la visualización estática o en cada frame de la animación.
        
    Returns:
        tuple o None: Si show es False, devuelve (fig, ax) para visualización estática, 
                        o (fig, ax, anim) para animación. De lo contrario None.
    """
    animacion = isinstance(thetas[0], (list, np.ndarray)) and hasattr(thetas[0], "__len__")
    
    if not ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    if view_angles:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    else:
        ax.view_init(elev=30, azim=60) 
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualización del Robot Manipulador')

    def _draw_provided_trajectory(ax_to_plot_on, trajectory_points):
        if trajectory_points is not None:
            try:
                path_data = np.asarray(trajectory_points)
                if path_data.ndim == 2 and path_data.shape[1] == 3 and path_data.shape[0] > 0:
                    ax_to_plot_on.plot(path_data[:, 0], path_data[:, 1], path_data[:, 2], 
                                        color='cyan', linestyle='--', linewidth=1.5, label='Trayectoria Proporcionada')
                # Optionally, add a legend if label is used: ax_to_plot_on.legend()
            except Exception as e:
                # Non-critical error, so print a warning
                print(f"Advertencia: No se pudo dibujar la trayectoria proporcionada. Error: {e}")

    if not animacion:
        _plot_frame(robot, thetas, ax)
        _draw_provided_trajectory(ax, trayectoria)
        _adjust_axis_limits(ax)
        
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax
    
    num_frames = len(thetas)
    max_limits = _get_animation_limits(robot, thetas) # Consider enhancing this to include 'trayectoria' points for limit calculation

    def init():
        ax.clear() # Clear axis before drawing the first frame
        # Set view, labels, title, limits for the first frame (as update will also do this)
        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        else:
            ax.view_init(elev=30, azim=60)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Visualización del Robot Manipulador - Frame 1/{num_frames}')
        ax.set_xlim(max_limits['x'])
        ax.set_ylim(max_limits['y'])
        ax.set_zlim(max_limits['z'])

        _plot_frame(robot, thetas[0], ax)
        _draw_provided_trajectory(ax, trayectoria)
        return []
    
    def update(frame_index):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Visualización del Robot Manipulador - Frame {frame_index+1}/{num_frames}')
        
        ax.set_xlim(max_limits['x'])
        ax.set_ylim(max_limits['y'])
        ax.set_zlim(max_limits['z'])
        
        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        else:
            ax.view_init(elev=30, azim=60)
        
        _plot_frame(robot, thetas[frame_index], ax)
        _draw_provided_trajectory(ax, trayectoria)
        
        return []
    
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                            interval=animation_speed, blit=True)
    
    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax, anim

def _plot_frame(robot, thetas, ax):
    """Función interna para dibujar un solo frame del robot."""
    transformaciones = calcular_transformaciones(robot, thetas)
    
    # Obtener las posiciones de las articulaciones
    positions = [T[:3, 3] for T in transformaciones]
    positions = np.array(positions)
    
    # Dibujar los enlaces entre articulaciones
    for i in range(len(positions)-1):
        link = robot.links[i]
        p1 = positions[i]
        p2 = positions[i+1]
        
        # Dibujar el enlace como una línea
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)
        
        # Dibujar la articulación
        if link.tipo == "revolute":
            # Articulación revoluta - círculo rojo
            ax.scatter([p2[0]], [p2[1]], [p2[2]], color='r', s=50, marker='o') # Adjust size here
            
            # Dibujar el eje de rotación como un vector
            rot_axis = np.array(link.joint_axis)
            rot_axis = rot_axis / np.linalg.norm(rot_axis) * 0.05  # Escalar el eje para la visualización
            
            # Transformar el eje de rotación al sistema de coordenadas global
            R = transformaciones[i+1][:3, :3]
            rot_axis_global = R @ rot_axis
            
            # Dibujar el eje de rotación
            ax.quiver(p2[0], p2[1], p2[2], 
                        rot_axis_global[0], rot_axis_global[1], rot_axis_global[2], 
                        color='g', arrow_length_ratio=0.3)
            
        elif link.tipo == "prismatic":
            # Articulación prismática - cuadrado verde
            ax.scatter([p2[0]], [p2[1]], [p2[2]], color='g', s=50, marker='s') # Adjust size here
            
            # Dibujar el eje prismático como un vector
            trans_axis = np.array(link.joint_axis)
            trans_axis = trans_axis / np.linalg.norm(trans_axis) * 0.05  # Escalar el eje para la visualización
            
            # Transformar el eje prismático al sistema de coordenadas global
            R = transformaciones[i+1][:3, :3]
            trans_axis_global = R @ trans_axis
            
            # Dibujar el eje prismático
            # Dibujar la extensión de la prismática en verde
            p_extend = p2 + thetas[i] * trans_axis_global  # thetas[i] es la cantidad que se extiende
            ax.plot([p2[0], p_extend[0]], [p2[1], p_extend[1]], [p2[2], p_extend[2]],
                    color='g', linestyle='-', linewidth=3)
    
    # Dibujar el punto final (efector final)
    ax.scatter([positions[-1][0]], [positions[-1][1]], [positions[-1][2]], color='k', s=150, marker='*')
    
    # Dibujar los sistemas de coordenadas para cada eslabón
    for i, T in enumerate(transformaciones):
        _draw_coordinate_system(ax, T, scale=0.05)

def _draw_coordinate_system(ax, T, scale=0.1):
    """
    Dibuja un sistema de coordenadas en la posición y orientación dada.
    
    Args:
        ax: Ejes de matplotlib.
        T (numpy.ndarray): Matriz de transformación homogénea 4x4.
        scale (float): Escala para los ejes del sistema de coordenadas.
    """
    origin = T[:3, 3]
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    
    # Dibujar los ejes X, Y, Z
    ax.quiver(origin[0], origin[1], origin[2], 
                x_axis[0], x_axis[1], x_axis[2], 
                color='r', arrow_length_ratio=0.3)
    
    ax.quiver(origin[0], origin[1], origin[2], 
                y_axis[0], y_axis[1], y_axis[2], 
                color='g', arrow_length_ratio=0.3)
    
    ax.quiver(origin[0], origin[1], origin[2], 
                z_axis[0], z_axis[1], z_axis[2], 
                color='b', arrow_length_ratio=0.3)

def _adjust_axis_limits(ax):
    """Ajusta los límites de los ejes para mantener proporciones iguales."""
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    
    # Encontrar el centro de cada eje
    x_center = np.mean(x_lim)
    y_center = np.mean(y_lim)
    z_center = np.mean(z_lim)
    
    # Encontrar el rango máximo
    max_range = max(x_lim[1] - x_lim[0], 
                    y_lim[1] - y_lim[0], 
                    z_lim[1] - z_lim[0]) / 2.0
    
    # Ajustar los límites para mantener proporciones iguales
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)

def _get_animation_limits(robot, thetas_list):
    """
    Determina los límites de los ejes para todos los frames de la animación.
    
    Args:
        robot: Objeto Robot que contiene los eslabones.
        thetas_list (list): Lista de listas de valores articulares.
        
    Returns:
        dict: Diccionario con los límites para cada eje {'x': (min, max), 'y': (min, max), 'z': (min, max)}.
    """
    x_all = []
    y_all = []
    z_all = []
    
    for thetas in thetas_list:
        transformaciones = calcular_transformaciones(robot, thetas)
        positions = [T[:3, 3] for T in transformaciones]
        positions = np.array(positions)
        
        x_all.extend(positions[:, 0])
        y_all.extend(positions[:, 1])
        z_all.extend(positions[:, 2])
    
    # Calcular los límites con un margen adicional
    margin = 0.1
    x_range = (min(x_all) - margin, max(x_all) + margin)
    y_range = (min(y_all) - margin, max(y_all) + margin)
    z_range = (min(z_all) - margin, max(z_all) + margin)
    
    # Asegurar que los ejes tengan proporciones iguales
    ranges = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
    max_range = max(ranges) / 2.0
    
    x_center = (x_range[0] + x_range[1]) / 2.0
    y_center = (y_range[0] + y_range[1]) / 2.0
    z_center = (z_range[0] + z_range[1]) / 2.0
    
    x_range = (x_center - max_range, x_center + max_range)
    y_range = (y_center - max_range, y_center + max_range)
    z_range = (z_center - max_range, z_center + max_range)
    
    return {'x': x_range, 'y': y_range, 'z': z_range}

# Ejemplo de uso
if __name__ == "__main__":
    from class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias
    
    # Cargar el robot
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    # Generar una configuración aleatoria
    thetas, _ = thetas_aleatorias(robot)
    print(f"Configuración aleatoria: {str_config(thetas, 3)}")
    
    # Visualizar el robot en una configuración estática
    plot_robot(robot, thetas)
    
    # Crear una animación
    num_frames = 50
    thetas_anim = []
    
    # Generar una secuencia de configuraciones
    thetas_start, _ = thetas_aleatorias(robot)
    thetas_end, _ = thetas_aleatorias(robot)
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        thetas_i = thetas_start * (1 - t) + thetas_end * t
        thetas_anim.append(thetas_i)
    
    # Visualizar la animación
    plot_robot(robot, thetas_anim, animation_speed=50)