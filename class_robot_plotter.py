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

from class_robot_structure import str_config, cargar_robot_desde_yaml, thetas_aleatorias, Robot, limits, get_limits_negative, get_limits_positive
from class_helicoidales import calcular_M_generalizado, calcular_T_robot

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

""" Funciones para guardar animaciones en diferentes formatos """
def guardar_animacion(anim, nombre_archivo):
    if input("\t¿Deseas guardar la animación? (s/n): ").strip().lower() != 's':
        print("\tAnimación no guardada.")
        return 
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
        except Exception as e_pillow: # Renamed e to e_pillow
            print(f"\t\033[31mError al guardar con Pillow: {e_pillow}\033[0m")
            print(f"\t\033[31mNo se pudo guardar la animación '{nombre_archivo}'.\n\tAsegúrate de tener ffmpeg o Pillow instalado correctamente.\033[0m")

def _guardar_animacion_ffmpeg(anim, nombre_archivo):
    print("\tIntentando guardar la animación como MP4 con ffmpeg...")
    anim.save(f"{nombre_archivo}.mp4", writer="ffmpeg", fps=30, dpi=225)
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.mp4' usando ffmpeg.\033[0m")

def _guardar_animacion_pillow(anim, nombre_archivo):
    print("\tIntentando guardar la animación como GIF con Pillow...")
    anim.save(f"{nombre_archivo}.gif", writer="pillow", fps=30)
    print(f"\t\033[92mAnimación guardada como '{nombre_archivo}.gif' usando Pillow.\033[0m")

""" Función principal para visualizar el robot manipulador en 3D """
def plot_robot(robot: Robot, thetas, ax=None, show=True, trayectoria=None, 
               animation_speed=200, view_angles=None, is_overlay=False):
    animacion = isinstance(thetas[0], (list, np.ndarray)) and hasattr(thetas[0], "__len__")
    
    fig_provided_ax = ax is not None # Renamed fig_provided to fig_provided_ax
    if not fig_provided_ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    if not is_overlay:
        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
        elif not fig_provided_ax : 
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
                    line, = ax_to_plot_on.plot(path_data[:, 0], path_data[:, 1], path_data[:, 2], 
                                               color='cyan', linestyle='--', linewidth=1.5, label='Trayectoria Proporcionada')
                    return [line] 
            except Exception as e:
                print(f"Advertencia: No se pudo dibujar la trayectoria proporcionada. Error: {e}")
        return []

    if not animacion:
        _plot_frame(robot, thetas, ax) 
        if trayectoria is not None:
            _draw_provided_trajectory(ax, trayectoria)
        if not is_overlay and not fig_provided_ax : 
            _adjust_axis_limits(ax) 
        
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax
    
    num_frames = len(thetas)
    max_limits = _get_animation_limits(robot, thetas) 
    robot_artists_collection = [] 

    def init_animation():
        nonlocal robot_artists_collection
        for artist in robot_artists_collection:
            artist.remove()
        robot_artists_collection.clear()
        trajectory_artists = []

        if not is_overlay:
            ax.clear() 
            if view_angles:
                ax.view_init(elev=view_angles[0], azim=view_angles[1])
            else:
                ax.view_init(elev=30, azim=60)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(max_limits['x'])
            ax.set_ylim(max_limits['y'])
            ax.set_zlim(max_limits['z'])
            if trayectoria is not None: 
                trajectory_artists.extend(_draw_provided_trajectory(ax, trayectoria))
        
        ax.set_title(f'Visualización del Robot Manipulador - Frame 1/{num_frames}')
        new_robot_artists = _plot_frame(robot, thetas[0], ax)
        robot_artists_collection.extend(new_robot_artists)
        return robot_artists_collection + trajectory_artists

    def update_animation(frame_index):
        nonlocal robot_artists_collection
        for artist in robot_artists_collection:
            artist.remove()
        robot_artists_collection.clear()
        ax.set_title(f'Visualización del Robot Manipulador - Frame {frame_index+1}/{num_frames}')
        
        # Axis properties (limits, view, labels) are set in init_animation if not is_overlay,
        # or by the caller if is_overlay. No need to change them per frame here.

        new_robot_artists = _plot_frame(robot, thetas[frame_index], ax)
        robot_artists_collection.extend(new_robot_artists)
        return robot_artists_collection
    
    anim = FuncAnimation(fig, update_animation, frames=num_frames, init_func=init_animation,
                            interval=animation_speed, blit=True)
    
    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax, anim

def _plot_frame(robot: Robot, thetas, ax):
    artists = []
    transformaciones = calcular_transformaciones(robot, thetas)
    positions = [T_mat[:3, 3] for T_mat in transformaciones] # Renamed T to T_mat
    positions = np.array(positions)
    
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
            scatter_marker = ax.scatter([p2[0]], [p2[1]], [p2[2]], color='r', s=50, marker='o')
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
            
            quiv = ax.quiver(p2[0], p2[1], p2[2], 
                             rot_axis_global[0], rot_axis_global[1], rot_axis_global[2], 
                             color='g', arrow_length_ratio=0.3)
            artists.append(quiv)
            
        elif link_obj.tipo == "prismatic":
            scatter_marker = ax.scatter([p2[0]], [p2[1]], [p2[2]], color='g', s=50, marker='s')
            artists.append(scatter_marker)
            
            local_joint_axis = np.array(link_obj.joint_axis)
            norm_val_prism = np.linalg.norm(local_joint_axis) # Renamed norm to norm_val_prism
            if norm_val_prism > 1e-9:
                local_joint_axis_unit = local_joint_axis / norm_val_prism
            else:
                local_joint_axis_unit = np.array([0,0,1])

            R_transform = transformaciones[i+1][:3, :3] 
            global_joint_axis_unit = R_transform @ local_joint_axis_unit
            
            p_start_ext_line = p2 
            # Draw line of length thetas[i] (actual displacement of joint i)
            p_end_ext_line = p_start_ext_line + thetas[i] * global_joint_axis_unit 
            
            prismatic_ext_line, = ax.plot([p_start_ext_line[0], p_end_ext_line[0]], 
                                          [p_start_ext_line[1], p_end_ext_line[1]], 
                                          [p_start_ext_line[2], p_end_ext_line[2]],
                                          color='lime', linestyle='-', linewidth=3) # Changed color
            artists.append(prismatic_ext_line)

    end_effector_scatter = ax.scatter([positions[-1][0]], [positions[-1][1]], [positions[-1][2]], color='k', s=150, marker='*')
    artists.append(end_effector_scatter)
    
    for i_cs, T_cs in enumerate(transformaciones):
        cs_artists = _draw_coordinate_system(ax, T_cs, scale=0.05)
        artists.extend(cs_artists)
    return artists

def _draw_coordinate_system(ax, T_mat, scale=0.1): # Renamed T to T_mat
    artists = []
    origin = T_mat[:3, 3]
    x_axis = T_mat[:3, 0] * scale
    y_axis = T_mat[:3, 1] * scale
    z_axis = T_mat[:3, 2] * scale
    
    quiv_x = ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.3)
    artists.append(quiv_x)
    quiv_y = ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.3)
    artists.append(quiv_y)
    quiv_z = ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.3)
    artists.append(quiv_z)
    return artists

def _adjust_axis_limits(ax):
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
    
    final_x_range = (x_center - half_max_dim_range, x_center + half_max_dim_range)
    final_y_range = (y_center - half_max_dim_range, y_center + half_max_dim_range)
    final_z_range = (z_center - half_max_dim_range, z_center + half_max_dim_range)
    
    return {'x': final_x_range, 'y': final_y_range, 'z': final_z_range}

from scipy.spatial import ConvexHull
def graficar_workspace(robot: Robot, N=2000, show_points=True, half_space_axis=None,
                       thetas_anim=None, animation_speed=200, view_angles=None,
                       save_animation_name=None):
    puntos_ws = []
    M = calcular_M_generalizado(robot)
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

    for _ in range(N):
        thetas_rand, _ = thetas_aleatorias(robot)
        T_mat = calcular_T_robot(robot.ejes_helicoidales, thetas_rand, M)
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
                    T_m_cfg = calcular_T_robot(robot.ejes_helicoidales, th_cfg, M) # Renamed
                    p_cfg = T_m_cfg[:3, 3] # Renamed
                    if apply_filter:
                        if (filter_positive_side and p_cfg[filter_axis_idx] >= 0) or \
                           (not filter_positive_side and p_cfg[filter_axis_idx] < 0):
                            puntos_ws.append(p_cfg)
                    else: puntos_ws.append(p_cfg)
        else: print("Advertencia: No se pudieron obtener los límites para puntos adicionales del workspace.")

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
    if thetas_anim: main_title += "\nSuperponiendo Animación"
    ax.set_title(main_title, fontsize=10)

    anim_obj = None
    if thetas_anim and len(thetas_anim) > 0:
        print("Superponiendo animación sobre el espacio de trabajo...")
        _fig_anim, _ax_anim, anim_obj = plot_robot(
            robot, thetas_anim, ax=ax, show=False, 
            animation_speed=animation_speed, view_angles=None, 
            trayectoria=None, is_overlay=True # Pass is_overlay=True
        )
    
    if show_points or hull_plotted: ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    # Show plot here if not handled by plot_robot (i.e., if plot_robot's show=False, which it is for animation overlay)
    # Or if it's a static workspace plot (no animation)
    if not (anim_obj and save_animation_name): # Avoid double plt.show() if anim_obj will be saved (which might show it)
                                              # More simply, always call plt.show() unless plot_robot did it.
                                              # Since plot_robot is called with show=False in animation case,
                                              # graficar_workspace is responsible for plt.show().
        plt.show()


    if save_animation_name and anim_obj:
        print(f"Intentando guardar la animación como '{save_animation_name}'...")
        guardar_animacion(anim_obj, save_animation_name) 
    
    return fig, ax, anim_obj


if __name__ == "__main__":
    robot = cargar_robot_desde_yaml("robot.yaml")

    print("Ejemplo 1: Visualización estática")
    thetas_static, _ = thetas_aleatorias(robot)
    print(f"Configuración aleatoria: {str_config(thetas_static, 3)}")
    plot_robot(robot, thetas_static)

    print("\nEjemplo 2: Espacio de trabajo básico")
    graficar_workspace(robot, N=500, show_points=True, half_space_axis=None) # Reduced N for tests

    print("\nEjemplo 3: Espacio de trabajo -z")
    graficar_workspace(robot, N=500, show_points=True, half_space_axis='-z')

    print("\nEjemplo 4: Espacio de trabajo +y, sin puntos")
    graficar_workspace(robot, N=500, show_points=False, half_space_axis='+y')

    num_frames_anim = 30 # Renamed
    thetas_anim_list = [] # Renamed
    thetas_s, _ = thetas_aleatorias(robot) # Renamed
    thetas_e, _ = thetas_aleatorias(robot) # Renamed
    thetas_s_np, thetas_e_np = np.array(thetas_s), np.array(thetas_e) # Renamed
    for i in range(num_frames_anim):
        t_param = i / (num_frames_anim - 1) if num_frames_anim > 1 else 0.0 # Renamed
        thetas_i = thetas_s_np * (1 - t_param) + thetas_e_np * t_param
        thetas_anim_list.append(thetas_i.tolist())
    
    print("\nEjemplo 5: Animación simple")
    plot_robot(robot, thetas_anim_list, animation_speed=50)

    print("\nEjemplo 6: Espacio de trabajo con animación superpuesta y guardado")
    graficar_workspace(robot, N=500, show_points=True, half_space_axis=None,
                       thetas_anim=thetas_anim_list, animation_speed=50,
                       save_animation_name="ws_anim_test")

    print("\nEjemplo 7: Espacio de trabajo +x con animación (sin puntos) y guardado")
    graficar_workspace(robot, N=500, show_points=False, half_space_axis='+x',
                       thetas_anim=thetas_anim_list, animation_speed=50,
                       save_animation_name="ws_mas_x_anim_test")

    print("\nEjemplo 8: Espacio de trabajo -y (sin animación, sin puntos)")
    graficar_workspace(robot, N=500, show_points=False, half_space_axis='-y')

    print("\nEjemplo 9: Espacio de trabajo completo (sin animación)")
    graficar_workspace(robot, N=500, show_points=True, half_space_axis=None)
