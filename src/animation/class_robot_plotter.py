"""
robot_plotter.py
=====
Este módulo proporciona funciones para visualizar un robot manipulador en 3D,
utilizando Matplotlib. Permite visualizar la estructura del robot en una configuración
estática o crear animaciones de movimiento.

Funciones:
    plot_robot: Visualiza el robot en una configuración específica o crea una animación.
    graficar_workspace: Genera un gráfico del espacio de trabajo del robot, con opciones
    para mostrar puntos muestreados y superponer animaciones.    
Ejemplo de uso:
    >>> robot = cargar_robot_desde_yaml("config/robot.yaml")
    >>> thetas = [0, 0, 0, 0, 0, 0, 0]  # Una configuración estática
    >>> plot_robot(robot, thetas)
    >>> # Para animación, se puede usar una lista de configuraciones
    >>> thetas_anim = [[0, 0, 0, 0, 0, 0, 0], [np.pi/4, 0, 0, 0, 0, 0, 0], ...]
    >>> fig, ax, anim = plot_robot(robot, thetas_anim)
    >>> fig, ax, anim = graficar_workspace(robot, N=1000, thetas_anim=thetas_anim)
    >>> guardar_animacion(anim, "robot_animation")
"""
import sys
import os
from core import thetas_aleatorias, Robot, limits, get_limits_negative, get_limits_positive, cargar_robot_desde_yaml
from calculations.class_helicoidales import CinematicaDirecta

import numpy as np                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                 # Import Matplotlib for 3D plotting
from matplotlib.animation import FuncAnimation  # Import FuncAnimation for animations
from scipy.spatial import ConvexHull            # Import ConvexHull for workspace visualization
import time                                     # Import time for timing operations
from datetime import datetime                   # Import datetime for timestamping
import pandas as pd                             # Import pandas for data manipulation

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
    
    R_mat = np.array([ # Renamed R to R_mat
        [x*x*C + c, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])
    
    return R_mat

def calcular_transformaciones(robot: Robot, thetas):
    """
    Calcula las transformaciones para cada eslabón del robot.
    
    Args:
        robot: Objeto Robot que contiene los eslabones.
        thetas (list): Lista de valores articulares.
        
    Returns:
        list: Lista de matrices de transformación homogénea 4x4 para cada eslabón.
    """
    n_links = robot.num_links
    T_current = np.eye(4)  # Transformación acumulada actual, renamed T to T_current
    transformaciones = [T_current.copy()]  # Guardar la transformación inicial
    
    for i in range(n_links):
        link_obj = robot.links[i] # Renamed link to link_obj
        theta_val = thetas[i] # Renamed theta to theta_val
        
        T_local = np.eye(4)
        
        if link_obj.tipo == "prismatic":
            T_local[:3, 3] = np.array(link_obj.joint_coords) + theta_val * np.array(link_obj.joint_axis)
        else:  # Para articulaciones rotacionales
            T_local[:3, 3] = np.array(link_obj.joint_coords)
            R_val = matriz_rotacion(link_obj.joint_axis, theta_val)
            T_local[:3, :3] = R_val
        
        T_current = T_current @ T_local
        transformaciones.append(T_current.copy())
    
    return transformaciones

""" Funciones para importar y exportar series de configuraciones """
def importar_trayectoria_cartesian(path):
    """
    Importa una trayectoria desde un archivo CSV en formato XYZ.
    
    Args:
        path (str): Ruta al archivo CSV.
        
    Returns:
        list: Lista de numpy.ndarray con posiciones [x, y, z] del efector final.
    """
    full_path = os.path.join("data/motion/cartesian", path)
    try:
        df = pd.read_csv(full_path, header=None)
        # Siempre devolver arrays de NumPy para consistencia
        trayectoria = [np.array(row, dtype=float) for row in df.values]
        print(f"\t\033[92mTrayectoria importada exitosamente desde '{full_path}'\033[0m")
        print(f"\t\033[94m{len(trayectoria)} puntos como numpy.ndarray\033[0m")
        return trayectoria
    except Exception as e:
        print(f"\t\033[31mError al importar trayectoria desde '{full_path}': {e}\033[0m")
        return []

def exportar_trayectoria_cartesian(path, trayectoria):
    """
    Exporta una trayectoria a un archivo CSV en formato XYZ.
    
    Args:
        path (str): Ruta al archivo CSV.
        trayectoria (list): Lista de posiciones [x, y, z] del efector final.
                           Acepta tanto numpy.ndarray como listas de Python.
    """
    full_path = os.path.join("data/motion/cartesian", path)
    try:
        if len(trayectoria) > 0: # Convertir todo a arrays de NumPy para uniformidad
            # Asegurar que todos los puntos son arrays de NumPy
            data_for_export = []
            for punto in trayectoria:
                if isinstance(punto, np.ndarray):
                    data_for_export.append(punto.tolist())
                else:
                    # Convertir lista a array y luego a lista para pandas
                    data_for_export.append(np.array(punto).tolist())
        else:
            data_for_export = []
        
        df = pd.DataFrame(data_for_export)
        df.to_csv(full_path, index=False, header=False)
        print(f"\t\033[92mTrayectoria exportada exitosamente a '{full_path}'\033[0m")
        print(f"\t\033[94m{len(data_for_export)} puntos exportados\033[0m")
        
    except Exception as e:
        print(f"\t\033[31mError al exportar trayectoria a '{full_path}': {e}\033[0m")

def importar_trayectoria_angular(path):
    """
    Importa configuraciones desde un archivo CSV.
    
    Args:
        path (str): Ruta al archivo CSV.
        
    Returns:
        list: Lista de configuraciones como listas de ángulos.
    """
    full_path = os.path.join("data/motion/angular", path)
    try:
        df = pd.read_csv(full_path)
        print(f"\t\033[92mConfiguraciones importadas exitosamente desde '{full_path}'\033[0m")
        print(f"\t\033[94m{len(df)} configuraciones importadas\033[0m")
        return df.values.tolist()
    except Exception as e:
        print(f"\t\033[31mError al importar configuraciones desde '{full_path}': {e}\033[0m")
        return []

def exportar_trayectoria_angular(path, configuraciones):
    """
    Exporta configuraciones a un archivo CSV.
    
    Args:
        path (str): Ruta al archivo CSV.
        configuraciones (list): Lista de configuraciones como listas de ángulos.
    """
    full_path = os.path.join("data/motion/angular", path)
    
    try:
        df = pd.DataFrame(configuraciones)
        df.to_csv(full_path, index=False)
        print(f"\t\033[92mConfiguraciones exportadas exitosamente a '{full_path}'\033[0m")
        print(f"\t\033[94m{len(configuraciones)} configuraciones exportadas\033[0m")
    except Exception as e:
        print(f"\t\033[31mError al exportar configuraciones a '{full_path}': {e}\033[0m")

""" Funciones para guardar animaciones en diferentes formatos """
def guardar_animacion(anim, nombre_archivo, fps=30, dpi=225):
    # Extraer solo el nombre de archivo (sin path) para mostrar al usuario
    ask = input(f"\t¿Deseas guardar la animación \033[93m{nombre_archivo.split('/')[-1].split('\\')[-1]}\033[0m? (s/n): \033[95m").strip().lower() != 's'
    print("\033[0m", end="")  # Reset color after input
    if ask:
        print("\tAnimación no guardada.")
        return 
    nombre_archivo = 'output/animations/' + nombre_archivo
    if not anim:
        print("No hay animación para guardar.")
        return
    print(f"Guardando animación ({nombre_archivo})...")
    try:
        _guardar_animacion_ffmpeg(anim, nombre_archivo, fps, dpi)
    except Exception as e:
        print(f"\t\033[31mError al guardar con ffmpeg: {e}\033[0m")
        _guardar_animacion_pillow(anim, nombre_archivo, fps, dpi)
    except Exception as e_final:
        print(f"\t\033[31mNo se pudo guardar la animación '{nombre_archivo}'.\n\tAsegúrate de tener ffmpeg o Pillow instalado correctamente.\033[0m")
        print(f"\t\033[31mError: {e_final}\033[0m")

def _guardar_animacion_ffmpeg(anim, nombre_archivo, fps=30, dpi=225):
    print("\tIntentando guardar la animación como MP4 con ffmpeg...")
    anim.save(f"{nombre_archivo}.mp4", writer="ffmpeg", fps=fps, dpi=dpi)
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.mp4' usando ffmpeg.\033[0m")

def _guardar_animacion_pillow(anim, nombre_archivo, fps=30, dpi=225):
    print("\tIntentando guardar la animación como GIF con Pillow...")
    anim.save(f"{nombre_archivo}.gif", writer="pillow", fps=fps, dpi=dpi)
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.gif' usando Pillow.\033[0m")

""" Función principal para visualizar el robot manipulador en 3D """
def plot_robot(robot: Robot, thetas, ax=None, show=True, trayectoria=None, 
               animation_speed=200, view_angles=None, is_overlay=False):
    """
    Visualiza un robot manipulador en 3D con Matplotlib.
    
    Esta función permite visualizar un robot manipulador en un espacio 3D, ya sea como una pose
    estática o como una animación de múltiples poses. También permite la visualización de trayectorias
    predefinidas y la superposición sobre gráficos existentes.
    
    Parameters
    ----------
    robot : Robot
        Objeto Robot que se desea visualizar. Debe contener la información 
        cinemática necesaria para el cálculo de las poses.
    thetas : array-like
        Ángulos de las articulaciones del robot. 
        - Si es un array 1D: visualiza una pose estática.
        - Si es un array 2D con shape[0] > 1: genera una animación con cada fila como una pose.
    ax : matplotlib.axes.Axes, optional
        Eje 3D de Matplotlib donde realizar la visualización. Si es None,
        se crea uno nuevo. Default: None.
    show : bool, optional
        Si es True, muestra el plot inmediatamente. Si es False, 
        permite manipulaciones adicionales antes de mostrarlo. Default: True.
    trayectoria : array-like, optional
        Puntos de una trayectoria para visualizar junto con el robot.
        Debe ser un array de shape (n,3) donde cada fila es un punto [x,y,z].
        Default: None.
    animation_speed : int, optional
        Velocidad de la animación en milisegundos por frame. Default: 200.
    view_angles : list or tuple, optional
        Ángulos de visualización [elevación, azimut] para la vista 3D.
        Si es None, se usa [30, 60]. Default: None.
    is_overlay : bool, optional
        Si es True, superpone la visualización sobre un gráfico existente sin
        reinicializar los ejes. Default: False.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        La figura de Matplotlib que contiene la visualización.
    ax : matplotlib.axes.Axes
        El objeto de ejes 3D que contiene la visualización.
    anim : matplotlib.animation.FuncAnimation, optional
        Solo se devuelve cuando se genera una animación. Objeto de animación
        que puede usarse para guardar la animación como un archivo.
    """
    tiempo_inicio = time.time()  # Captura el tiempo de inicio para medir duración
    
    thetas = np.asarray(thetas)
    if thetas.ndim == 2 and thetas.shape[0] > 1:
        animacion = True
    else:
        animacion = False
    
    fig_provided_ax = ax is not None
    if not fig_provided_ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    if not is_overlay:
        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        elif not fig_provided_ax:
            ax.view_init(elev=30, azim=60)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    ax.set_title('Visualización del Robot Manipulador')

    def _draw_provided_trajectory(ax_to_plot_on, trajectory_points):
        """ Dibuja la trayectoria proporcionada en el eje especificado. """
        if trajectory_points is not None:
            try:
                path_data = np.asarray(trajectory_points)
                if path_data.ndim == 2 and path_data.shape[1] == 3 and path_data.shape[0] > 0:
                    ax_to_plot_on.plot(path_data[:, 0], path_data[:, 1], path_data[:, 2], 
                                       color='cyan', linestyle='--', linewidth=1.5, label='Trayectoria Proporcionada')
            except Exception as e:
                print(f"Advertencia: No se pudo dibujar la trayectoria proporcionada. Error: {e}")

    if not animacion:
        _plot_frame(robot, thetas, ax)
        if trayectoria is not None:
            _draw_provided_trajectory(ax, trayectoria)
        # Ajustar límites de los ejes siempre para evitar distorsión
        if not is_overlay:
            _adjust_axis_limits(ax)

        if show:
            plt.tight_layout()
            # Medición de tiempo SOLO de los cálculos previos al plot
            print(f"\t\033[92mTiempo de cálculo de plot_robot: {time.time() - tiempo_inicio:.4f} segundos\033[0m")
            plt.show()
        return fig, ax

    num_frames = len(thetas)
    max_limits = _get_animation_limits(robot, thetas)

    def init():
        if not is_overlay:
            ax.clear()
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
        # --- CAMBIO: limpiar y guardar artistas también en overlay ---
        if is_overlay:
            for artist in artists_overlay:
                try:
                    artist.remove()
                except Exception:
                    pass
            artists_overlay.clear()
        # Dibuja el primer frame y guarda los artistas
        artists = _plot_frame(robot, thetas[0], ax)
        if is_overlay:
            artists_overlay.extend(artists)
        if trayectoria is not None:
            _draw_provided_trajectory(ax, trayectoria)
        return []

    artists_overlay = []

    def update(frame_index):
        # Limpiar los artistas del frame anterior SOLO si es overlay
        if is_overlay:
            for artist in artists_overlay:
                try:
                    artist.remove()
                except Exception:
                    pass
            artists_overlay.clear()
        elif not is_overlay:
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
        artists = _plot_frame(robot, thetas[frame_index], ax)        # Aquí guarda los artistas del frame actual
        if is_overlay:
            artists_overlay.extend(artists)
        if trayectoria is not None:
            _draw_provided_trajectory(ax, trayectoria)
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                         interval=animation_speed, blit=True)

    if show:
        plt.tight_layout()
        # Medición de tiempo SOLO de los cálculos previos al plot
        print(f"\t\033[92mTiempo de cálculo de plot_robot: {time.time() - tiempo_inicio:.4f} segundos\033[0m")
        plt.show()

    return fig, ax, anim

def _plot_frame(robot: Robot, thetas, ax):
    """ Dibuja un frame del robot en una configuración específica. """
    artists = []
    transformaciones = calcular_transformaciones(robot, thetas)
    positions = [T_mat[:3, 3] for T_mat in transformaciones] # Renamed T to T_mat
    positions = np.array(positions)
    
    # Verificar si las articulaciones están dentro de los límites
    joint_within_limits = []
    if robot.limits_dict is not None and len(robot.limits_dict) > 0:
        for i in range(len(thetas)):
            if i < len(robot.links) and f'joint_{i+1}' in robot.limits_dict:
                limit_min, limit_max = robot.limits_dict[f'joint_{i+1}']
                is_within = limit_min <= thetas[i] <= limit_max
                joint_within_limits.append(is_within)
            else:
                joint_within_limits.append(True)  # Si no hay límites, asumimos que está dentro
    else:
        joint_within_limits = [True] * len(thetas)  # Si no hay límites definidos, todas están dentro
    
    for i in range(len(positions)-1):
        link_obj = robot.links[i]
        p1 = positions[i]
        p2 = positions[i+1]
        
        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)
        artists.append(line)
        
        # The visualization of joint type (revolute/prismatic) and its axis
        # is associated with link_obj (robot.links[i]), but drawn at p2 (positions[i+1]).
        # This implies joint i's visual representation is at the end of link i.
        if link_obj.tipo == "revolute":
            # Elegir color basado en si la articulación está dentro de los límites
            joint_color = 'r' if joint_within_limits[i] else 'purple'
            scatter_marker = ax.scatter([p2[0]], [p2[1]], [p2[2]], color=joint_color, s=50, marker='o')
            artists.append(scatter_marker)
            
            rot_axis_local = np.array(link_obj.joint_axis)
            norm_val = np.linalg.norm(rot_axis_local) # Renamed norm to norm_val
            if norm_val > 1e-9:
                rot_axis_local_unit = rot_axis_local / norm_val
            else:
                rot_axis_local_unit = np.array([0,0,1]) 
            
            rot_axis_scaled = rot_axis_local_unit * 0.05
            R_transform = transformaciones[i+1][:3, :3]
            rot_axis_global = R_transform @ rot_axis_scaled
            
            # Flecha del eje de rotación con z-order alto para que se muestre por encima
            quiv = ax.quiver(p2[0], p2[1], p2[2], 
                             rot_axis_global[0], rot_axis_global[1], rot_axis_global[2], 
                             color='goldenrod', arrow_length_ratio=0.3, 
                             alpha=0.9, zorder=10)
            artists.append(quiv)
            
        elif link_obj.tipo == "prismatic":
            # Elegir color basado en si la articulación está dentro de los límites
            joint_color = 'g' if joint_within_limits[i] else 'purple'
            scatter_marker = ax.scatter([p2[0]], [p2[1]], [p2[2]], color=joint_color, s=50, marker='s')
            artists.append(scatter_marker)
            
            # Para articulaciones prismáticas, podemos mostrar el eje de traslación con una flecha
            local_joint_axis = np.array(link_obj.joint_axis)
            norm_val_prism = np.linalg.norm(local_joint_axis) # Renamed norm to norm_val_prism
            if norm_val_prism > 1e-9:
                local_joint_axis_unit = local_joint_axis / norm_val_prism
            else:
                local_joint_axis_unit = np.array([0,0,1])

            R_transform = transformaciones[i+1][:3, :3] 
            global_joint_axis_unit = R_transform @ local_joint_axis_unit
            
            # Mostrar el eje de traslación con una flecha (sin representar extensión errónea)
            axis_scale = 0.05  # Escala para la flecha del eje
            quiv_prism = ax.quiver(p2[0], p2[1], p2[2], 
                                   global_joint_axis_unit[0] * axis_scale, 
                                   global_joint_axis_unit[1] * axis_scale, 
                                   global_joint_axis_unit[2] * axis_scale, 
                                   color='orange', arrow_length_ratio=0.3, 
                                   alpha=0.8, zorder=5)
            artists.append(quiv_prism)

    end_effector_scatter = ax.scatter([positions[-1][0]], [positions[-1][1]], [positions[-1][2]], color='k', s=150, marker='*')
    artists.append(end_effector_scatter)
    
    for i_cs, T_cs in enumerate(transformaciones):
        cs_artists = _draw_coordinate_system(ax, T_cs, scale=0.05)
        artists.extend(cs_artists)
    return artists

def _draw_coordinate_system(ax, T_mat, scale=0.05):
    """ Dibuja un sistema de coordenadas en el espacio 3D. """
    artists = []
    origin = T_mat[:3, 3]
    x_axis = T_mat[:3, 0] * scale
    y_axis = T_mat[:3, 1] * scale
    z_axis = T_mat[:3, 2] * scale
    
    quiv_x = ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], 
                       color='r', arrow_length_ratio=0.3, zorder=1)
    artists.append(quiv_x)
    quiv_y = ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], 
                       color='g', arrow_length_ratio=0.3, zorder=1)
    artists.append(quiv_y)
    quiv_z = ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], 
                       color='b', arrow_length_ratio=0.3, zorder=1)
    artists.append(quiv_z)
    return artists

def _adjust_axis_limits(ax):
    """ Ajusta los límites de los ejes para que tengan el mismo rango. """
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    
    x_center = np.mean(x_lim)
    y_center = np.mean(y_lim)
    z_center = np.mean(z_lim)
    
    current_ranges = [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    # Ensure ranges are positive before taking max
    valid_ranges = [r for r in current_ranges if r > 1e-9]
    if not valid_ranges: # All ranges are tiny or zero
        max_abs_range = 1.0 # Default span, results in 0.5 half-range
    else:
        max_abs_range = max(valid_ranges)

    half_max_range = max_abs_range / 2.0
    if half_max_range < 1e-6: # if range is tiny, set a default span
        half_max_range = 0.5

    ax.set_xlim(x_center - half_max_range, x_center + half_max_range)
    ax.set_ylim(y_center - half_max_range, y_center + half_max_range)
    ax.set_zlim(z_center - half_max_range, z_center + half_max_range)

def _get_animation_limits(robot: Robot, thetas_list):
    """ Obtiene los límites del espacio de trabajo del robot para la animación. """
    x_all, y_all, z_all = [], [], []
    
    for thetas_single_frame in thetas_list:
        transformaciones = calcular_transformaciones(robot, thetas_single_frame)
        positions = [T_mat[:3, 3] for T_mat in transformaciones] # Renamed T to T_mat
        if not positions: continue
        positions_np = np.array(positions)
        
        x_all.extend(positions_np[:, 0])
        y_all.extend(positions_np[:, 1])
        z_all.extend(positions_np[:, 2])
    
    if not x_all: 
        return {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.5, 0.5)}

    margin = 0.1 
    min_x, max_x = min(x_all) - margin, max(x_all) + margin
    min_y, max_y = min(y_all) - margin, max(y_all) + margin
    min_z, max_z = min(z_all) - margin, max(z_all) + margin
    
    x_range_val = (min_x, max_x)
    y_range_val = (min_y, max_y)
    z_range_val = (min_z, max_z)
    
    ranges_dims = [x_range_val[1] - x_range_val[0], 
                   y_range_val[1] - y_range_val[0], 
                   z_range_val[1] - z_range_val[0]]
    
    valid_ranges = [d for d in ranges_dims if d > 1e-9]
    if not valid_ranges:
        max_dim_abs_range = 1.0 # Default span
    else:
        max_dim_abs_range = max(valid_ranges)

    half_max_dim_range = max_dim_abs_range / 2.0
    if half_max_dim_range < 1e-6: 
        half_max_dim_range = 0.5

    x_center = (x_range_val[0] + x_range_val[1]) / 2.0
    y_center = (y_range_val[0] + y_range_val[1]) / 2.0
    z_center = (z_range_val[0] + z_range_val[1]) / 2.0

    return {
        'x': (x_center - half_max_dim_range, x_center + half_max_dim_range),
        'y': (y_center - half_max_dim_range, y_center + half_max_dim_range),
        'z': (z_center - half_max_dim_range, z_center + half_max_dim_range)
    }

def _log_performance_data(robot: Robot, N, execution_time):
    """
    Registra los datos de rendimiento de la generación del workspace en un archivo CSV.
    """
    log_file = 'data/workspace_performance_log.csv'
    
    # Datos a registrar
    new_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'robot_name': [robot.name],
        'num_links': [robot.num_links],
        'N': [N],
        'execution_time': [execution_time]
    }
    
    df_new = pd.DataFrame(new_data)
    
    # Si el archivo no existe, se crea con cabeceras
    if not os.path.exists(log_file):
        df_new.to_csv(log_file, index=False, header=True)
    else:
        # Si ya existe, se añaden los datos sin cabeceras
        df_new.to_csv(log_file, mode='a', index=False, header=False)

""" Función para graficar los rangos de los límites del robot """
def graficar_limites(robot: Robot, ax=None, show=True):
    """
    Visualiza los límites de las articulaciones del robot en gráficos de rueda (dial/gauge).
    
    Parameters
    ----------
    robot : Robot
        Objeto Robot que contiene la información cinemática del robot.
    ax : matplotlib.axes.Axes, optional
        No se usa en esta implementación de ruedas. Default: None.
    show : bool, optional
        Si es True, muestra el plot inmediatamente. Si es False, 
        permite manipulaciones adicionales antes de mostrarlo. Default: True.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        La figura de Matplotlib que contiene la visualización.
    axes : list
        Lista de objetos de ejes que contienen cada rueda.
    """
    import matplotlib.patches as patches
    
    # Calcular el número de articulaciones
    num_joints = robot.num_links
    
    # Determinar el layout de subplots
    if num_joints <= 3:
        rows, cols = 1, num_joints
    elif num_joints <= 6:
        rows, cols = 2, 3
    else:
        rows = int(np.ceil(num_joints / 4))
        cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_joints == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Ocultar ejes no utilizados
    for i in range(num_joints, len(axes)):
        axes[i].set_visible(False)
    
    for i in range(num_joints):
        ax = axes[i]
        link = robot.links[i]
        
        # Obtener límites de la articulación
        if robot.limits_dict and f'joint_{i+1}' in robot.limits_dict:
            limit_min, limit_max = robot.limits_dict[f'joint_{i+1}']
        else:
            limit_min, limit_max = -np.pi, np.pi
        
        # Configurar el eje
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if link.tipo == "revolute":
            # Dibujar el círculo base
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
            ax.add_patch(circle)
            
            # Convertir límites a grados para visualización
            angle_min_deg = np.degrees(limit_min)
            angle_max_deg = np.degrees(limit_max)
            
            # Crear arco de rango válido
            if limit_max > limit_min:
                # Ángulos en coordenadas matplotlib (0° = este, 90° = norte)
                theta_start = np.radians(90 - angle_max_deg)  # Inicio del arco
                theta_end = np.radians(90 - angle_min_deg)    # Final del arco
                
                # Si el arco cruza el meridiano 0°, dividirlo en dos partes
                if theta_start > theta_end:
                    # Crear dos arcos separados
                    angles1 = np.linspace(theta_start, 2*np.pi, 50)
                    angles2 = np.linspace(0, theta_end, 50)
                    angles = np.concatenate([angles1, angles2])
                else:
                    angles = np.linspace(theta_start, theta_end, 100)
                
                # Crear el sector circular para el rango válido
                angles_fill = np.concatenate([[0], angles, [0]])
                x_fill = np.concatenate([[0], 0.9 * np.cos(angles), [0]])
                y_fill = np.concatenate([[0], 0.9 * np.sin(angles), [0]])
                ax.fill(x_fill, y_fill, alpha=0.3, color='green', label='Rango válido')
                
                # Líneas de límites más gruesas y mejor posicionadas
                x_min = np.cos(theta_end)
                y_min = np.sin(theta_end)
                x_max = np.cos(theta_start)
                y_max = np.sin(theta_start)
                
                ax.plot([0, x_min], [0, y_min], 'r-', linewidth=4, label=f'Límite mín')
                ax.plot([0, x_max], [0, y_max], 'b-', linewidth=4, label=f'Límite máx')
                
                # Añadir puntos en los extremos de los límites
                ax.scatter([x_min], [y_min], color='red', s=100, zorder=5)
                ax.scatter([x_max], [y_max], color='blue', s=100, zorder=5)
            
            # Marcas de graduación cada 30 grados
            for angle_deg in range(0, 360, 30):
                x_outer = 1.0 * np.cos(np.radians(90 - angle_deg))
                y_outer = 1.0 * np.sin(np.radians(90 - angle_deg))
                x_inner = 0.85 * np.cos(np.radians(90 - angle_deg))
                y_inner = 0.85 * np.sin(np.radians(90 - angle_deg))
                
                ax.plot([x_inner, x_outer], [y_inner, y_outer], 'k-', linewidth=1)
                
                # Etiquetas cada 90 grados mostrando grados y radianes
                if angle_deg % 90 == 0:
                    x_label = 1.2 * np.cos(np.radians(90 - angle_deg))
                    y_label = 1.2 * np.sin(np.radians(90 - angle_deg))
                    angle_rad = np.radians(angle_deg)
                    ax.text(x_label, y_label, f'{angle_deg}°\n{angle_rad:.2f}rad', 
                           ha='center', va='center', fontsize=7, 
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
            
            # Mostrar valores de límites con mejor formato en la esquina inferior derecha
            limit_min_deg = np.degrees(limit_min)
            limit_max_deg = np.degrees(limit_max)
            
            limit_text = f'Min: {limit_min_deg:.1f}° ({limit_min:.3f} rad)\n'
            limit_text += f'Max: {limit_max_deg:.1f}° ({limit_max:.3f} rad)\n'
            limit_text += f'Rango: {limit_max_deg - limit_min_deg:.1f}°'
            
            ax.text(1.4, -1.4, limit_text, ha='right', va='bottom', fontsize=6,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
            
            color = 'darkgreen'
            joint_type = 'Revoluta'
            
        elif link.tipo == "prismatic":
            # Para articulaciones prismáticas, mostrar como barra vertical
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.5, 1.5)
            
            # Dibujar barra vertical
            bar_width = 0.4
            bar_height = 2.5
            rect = patches.Rectangle((-bar_width/2, -bar_height/2), bar_width, bar_height, 
                                   linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
            ax.add_patch(rect)
            
            # Marcas de límites
            if limit_max > limit_min:
                # Normalizar límites al rango de la barra
                range_total = limit_max - limit_min
                if range_total > 0:
                    y_min_norm = -bar_height/2 + (0) / range_total * bar_height
                    y_max_norm = -bar_height/2 + (range_total) / range_total * bar_height
                    
                    # Región válida
                    valid_rect = patches.Rectangle((-bar_width/2, y_min_norm), bar_width, 
                                                 y_max_norm - y_min_norm, 
                                                 facecolor='green', alpha=0.3, label='Rango válido')
                    ax.add_patch(valid_rect)
                    
                    # Líneas de límites
                    ax.plot([-bar_width/2, bar_width/2], [y_min_norm, y_min_norm], 'r-', linewidth=4)
                    ax.plot([-bar_width/2, bar_width/2], [y_max_norm, y_max_norm], 'b-', linewidth=4)
                    
                    # Puntos en los límites
                    ax.scatter([0], [y_min_norm], color='red', s=100, zorder=5)
                    ax.scatter([0], [y_max_norm], color='blue', s=100, zorder=5)
                    
                    # Etiquetas de límites
                    ax.text(bar_width/2 + 0.2, y_min_norm, f'Min: {limit_min:.3f}m', 
                           va='center', fontsize=8, color='red', weight='bold')
                    ax.text(bar_width/2 + 0.2, y_max_norm, f'Max: {limit_max:.3f}m', 
                           va='center', fontsize=8, color='blue', weight='bold')
                    
                    # Información adicional en la esquina inferior derecha
                    range_text = f'Rango: {limit_max - limit_min:.3f}m'
                    ax.text(0.9, -1.4, range_text, ha='right', va='bottom', fontsize=6,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.8))
            
            color = 'darkblue'
            joint_type = 'Prismática'
        
        # Título de cada rueda/articulación usando link.id
        ax.set_title(f'{link.id}: {joint_type}', fontsize=8, color=color, weight='bold')
    
    # Título general
    fig.suptitle(f'Límites de Articulaciones - {robot.name}', fontsize=14, weight='bold')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, axes

""" Función para graficar el espacio de trabajo del robot manipulador """
def graficar_workspace(robot: Robot, N=2000, show_points=True, half_space_axis=None,
                       thetas_anim=None, animation_speed=200, view_angles=(30, 45),
                       save_animation_name=None, subtitle=None, show_plot=True):
    """
    Visualiza el espacio de trabajo de un robot y opcionalmente superpone una animación.
    Esta función genera un gráfico 3D del espacio de trabajo del robot mediante muestreo
    aleatorio de configuraciones válidas. Puede mostrar la frontera convexa del espacio
    de trabajo y superponer animaciones del robot.
    Parameters
    ----------
    robot : Robot
        Objeto Robot que contiene la información cinemática del robot.
    N : int, optional
        Número de muestras aleatorias para generar el espacio de trabajo (default: 2000).
    show_points : bool, optional
        Si True, muestra los puntos muestreados del espacio de trabajo (default: True).
    half_space_axis : str, optional
        Filtro para mostrar solo un semi-espacio. Formato: "+X", "-Y", "+Z", etc.
        (default: None).
    thetas_anim : list, optional
        Lista de configuraciones de ángulos para superponer una animación (default: None).
    animation_speed : int, optional
        Velocidad de la animación en milisegundos entre frames (default: 200).
    view_angles : tuple, optional
        Ángulos de vista (elevación, azimut) para la cámara 3D (default: None).
    save_animation_name : str, optional
        Nombre del archivo para guardar la animación (default: None).
    subtitle : str, opcional
        Subtítulo para el gráfico (default: None).
    show_plot : bool, opcional
        Si True, muestra el gráfico al final (default: True). Útil para desactivar
        al recolectar datos de rendimiento.
    Returns
    -------
    tuple
        Una tupla conteniendo:
        - fig : matplotlib.figure.Figure
            La figura de matplotlib generada.
        - ax : matplotlib.axes._subplots.Axes3DSubplot
            Los ejes 3D de la figura.
        - anim_obj : matplotlib.animation.FuncAnimation or None
            Objeto de animación si se proporcionó thetas_anim, None en caso contrario.
    Notes
    -----
    - La función calcula puntos del espacio de trabajo mediante muestreo aleatorio
        de configuraciones válidas del robot.
    - Si el robot tiene límites definidos, se agregan puntos adicionales en las
        configuraciones límite.
    - La frontera convexa se calcula y dibuja si hay suficientes puntos (≥4).
    - El filtro half_space_axis permite visualizar solo una porción del espacio
        de trabajo (ej: "+Z" muestra solo puntos con Z≥0).
    - Si se proporciona thetas_anim, se superpone una animación del robot sobre
        el espacio de trabajo.
    Examples
    --------
    >>> # Visualizar espacio de trabajo básico
    >>> fig, ax, anim = graficar_workspace(mi_robot, N=1000)
    >>> # Visualizar solo el semi-espacio superior en Z
    >>> fig, ax, anim = graficar_workspace(mi_robot, half_space_axis="+Z")
    >>> # Superponer animación
    >>> trayectoria = [config1, config2, config3]
    >>> fig, ax, anim = graficar_workspace(mi_robot, thetas_anim=trayectoria)
    """
    
    def format_time_hms(seconds):
        """Format seconds into hours, minutes, and seconds."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds_part = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds_part}s"
        elif minutes > 0:
            return f"{minutes}m {seconds_part}s"
        else:
            return f"{seconds:.2f}s"

    print(f"\n\t\033[92mIniciando graficar_workspace... Tiempo esperado (N={N}): {format_time_hms(1.905905e-01 + (2.69602793e-06 * N) + (9.54891620e-12 * N * N))}\033[0m") # 19/06/2025
    tiempo_inicio = time.time()
    puntos_ws = []
    M = robot.M
    apply_filter = False
    filter_axis_idx = -1
    filter_positive_side = True

    if half_space_axis and isinstance(half_space_axis, str) and len(half_space_axis) == 2:
        sign_char, axis_char = half_space_axis[0], half_space_axis[1].lower()
        if sign_char == '+': filter_positive_side = True
        elif sign_char == '-': filter_positive_side = False
        else: sign_char = None
        if axis_char == 'x': filter_axis_idx = 0
        elif axis_char == 'y': filter_axis_idx = 1
        elif axis_char == 'z': filter_axis_idx = 2
        else: filter_axis_idx = -1
        if sign_char is not None and filter_axis_idx != -1: apply_filter = True
        else:
            print(f"Advertencia: 'half_space_axis' ('{half_space_axis}') inválido. Se ignorará el filtro.")
            half_space_axis = None
    elif half_space_axis:
        print(f"Advertencia: 'half_space_axis' ('{half_space_axis}') con formato incorrecto. Se ignorará el filtro.")
        half_space_axis = None

    try:
        print("\t(Puedes presionar Ctrl+C para detener la generación de puntos y visualizar los resultados parciales)")
        for _ in range(N):
            thetas_rand, _ = thetas_aleatorias(robot)
            T_mat = CinematicaDirecta(robot.ejes_helicoidales, thetas_rand, M)
            punto = T_mat[:3, 3]
            if apply_filter:
                if (filter_positive_side and punto[filter_axis_idx] >= 0) or \
                   (not filter_positive_side and punto[filter_axis_idx] < 0):
                    puntos_ws.append(punto)
            else: puntos_ws.append(punto)

        if robot.limits_dict and robot.num_links > 0:
            min_l, max_l = get_limits_negative(robot), get_limits_positive(robot) # Shorter names
            if min_l is not None and max_l is not None and len(min_l) == robot.num_links and len(max_l) == robot.num_links:
                mid_l = (np.array(min_l) + np.array(max_l)) / 2.0
                cfgs_check = [min_l, max_l] # Renamed configurations_to_check
                for i in range(robot.num_links):
                    for lim_type in ["min", "max"]:
                        th_at_lim = mid_l.copy() # Renamed thetas_at_limit
                        th_at_lim[i] = min_l[i] if lim_type == "min" else max_l[i]
                        cfgs_check.append(th_at_lim)

                unique_cfgs_set, final_cfgs = set(), [] # Renamed
                for cfg in cfgs_check: # Renamed config to cfg
                    cfg_tuple = tuple(cfg)
                    if cfg_tuple not in unique_cfgs_set:
                        final_cfgs.append(list(cfg))
                        unique_cfgs_set.add(cfg_tuple)

                for th_cfg in final_cfgs: # Renamed thetas_config to th_cfg
                    if limits(robot, th_cfg)[0]:
                        T_m_cfg = CinematicaDirecta(robot.ejes_helicoidales, th_cfg, M) # Renamed
                        p_cfg = T_m_cfg[:3, 3] # Renamed
                        if apply_filter:
                            if (filter_positive_side and p_cfg[filter_axis_idx] >= 0) or \
                               (not filter_positive_side and p_cfg[filter_axis_idx] < 0):
                                puntos_ws.append(p_cfg)
                        else: puntos_ws.append(p_cfg)
            else: print("Advertencia: No se pudieron obtener los límites para puntos adicionales del workspace.")
            cuted = False
    except KeyboardInterrupt:
        print("\n\t\033[93mInterrupción por el usuario. Visualizando con los puntos generados hasta ahora...\033[0m")
        cuted = True
        
    puntos_ws_array = np.array(puntos_ws) if puntos_ws else np.empty((0,3))
    num_actual_puntos_ws = puntos_ws_array.shape[0]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []
    if num_actual_puntos_ws > 0:
        all_x.extend(puntos_ws_array[:, 0]); all_y.extend(puntos_ws_array[:, 1]); all_z.extend(puntos_ws_array[:, 2])
    if thetas_anim:
        for t_anim_f in thetas_anim: # Renamed
            tf_f = calcular_transformaciones(robot, t_anim_f) # Renamed
            pos_f = np.array([T_m_f[:3, 3] for T_m_f in tf_f]) # Renamed
            if pos_f.size > 0:
                all_x.extend(pos_f[:, 0]); all_y.extend(pos_f[:, 1]); all_z.extend(pos_f[:, 2])
    if not all_x: all_x, all_y, all_z = [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]

    min_x_all, max_x_all = min(all_x) - 0.1, max(all_x) + 0.1
    min_y_all, max_y_all = min(all_y) - 0.1, max(all_y) + 0.1
    min_z_all, max_z_all = min(all_z) - 0.1, max(all_z) + 0.1
    x_c, y_c, z_c = (min_x_all+max_x_all)/2, (min_y_all+max_y_all)/2, (min_z_all+max_z_all)/2 # Renamed
    max_r_d = max(max_x_all-min_x_all, max_y_all-min_y_all, max_z_all-min_z_all) # Renamed
    h_span = max(0.5, max_r_d / 2.0) # Renamed
    ax.set_xlim(x_c - h_span, x_c + h_span); ax.set_ylim(y_c - h_span, y_c + h_span); ax.set_zlim(z_c - h_span, z_c + h_span)

    if view_angles: ax.view_init(elev=view_angles[0], azim=view_angles[1])
    else: ax.view_init(elev=25, azim=45)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    if show_points and num_actual_puntos_ws > 0:
        ax.scatter(puntos_ws_array[:,0], puntos_ws_array[:,1], puntos_ws_array[:,2], s=5, alpha=0.15, label='Muestras Espacio Trabajo', color='darkgrey')

    title_str = "Espacio de Trabajo"
    if apply_filter and half_space_axis: title_str = f"Medio Espacio de Trabajo ({half_space_axis})"

    hull_plotted, hull_message = False, ""
    if num_actual_puntos_ws >= 4:
        try:
            hull_obj = ConvexHull(puntos_ws_array) # Renamed hull to hull_obj
            ax.plot_trisurf(puntos_ws_array[:,0], puntos_ws_array[:,1], puntos_ws_array[:,2], triangles=hull_obj.simplices,
                            color='cornflowerblue', alpha=0.25, edgecolor='black', linewidth=0.15, label='Frontera Convexa')
            if not (apply_filter and half_space_axis) : title_str += " y Frontera Convexa"
            hull_plotted = True
        except Exception as e_hull:
            hull_message = f" (Frontera Convexa no dibujada: {type(e_hull).__name__})"
            print(f"Advertencia: No se pudo calcular/dibujar la frontera convexa: {e_hull}")
    else:
        hull_message = " (Frontera Convexa no dibujada - Pocos puntos)"
        if num_actual_puntos_ws > 0 : print(f"Advertencia: No hay suficientes puntos ({num_actual_puntos_ws}) para dibujar la frontera convexa (se necesitan al menos 4).")

    main_title = f"{title_str} de {robot.name} ({num_actual_puntos_ws} puntos){hull_message}"
    if subtitle:
        main_title = f"{main_title}\n{subtitle}"
    if thetas_anim: main_title += "\tSuperponiendo Animación"
    ax.set_title(main_title, fontsize=10)

    anim_obj = None
    if thetas_anim and len(thetas_anim) > 0:
        print("\tSuperponiendo animación sobre el espacio de trabajo...")
        _fig_anim, _ax_anim, anim_obj = plot_robot(
            robot, thetas_anim, ax=ax, show=False,
            animation_speed=animation_speed, view_angles=None,
            trayectoria=None, is_overlay=True # Pass is_overlay=True
        )

    # Captura del tiempo de cálculo después de las operaciones principales y antes de mostrar/guardar
    tiempo_calculo = time.time() - tiempo_inicio
    print(f"\t\033[92mTiempo de cálculo{' interrumpido' if cuted else ''} (graficar_workspace(N={N})): {format_time_hms(tiempo_calculo)}\033[0m")

    # Registrar datos de rendimiento si no se cortó la ejecución
    if not cuted: _log_performance_data(robot, N, tiempo_calculo)

    if show_points or hull_plotted: ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show_plot: # Usar el nuevo parámetro
        if not (anim_obj and save_animation_name): # Condición original para mostrar
            plt.show()

    if save_animation_name and anim_obj:
        print(f"Intentando guardar la animación como '\033[94m{save_animation_name}\033[0m'...")
        guardar_animacion(anim_obj, save_animation_name)
    
    return fig, ax, anim_obj, tiempo_calculo # Devolver tiempo_calculo