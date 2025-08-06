#!/usr/bin/env python3
"""
Demo automático de los controles de teclado para FABRIK 3D.
Muestra el robot siguiendo una trayectoria predefinida.
"""

import numpy as np
from core import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite, Robot
from calculations import Rp2Trans
from animation import importar_trayectoria_cartesian, exportar_trayectoria_cartesian
from fabrik_paper_constrained_3d import Fabrik_3D

def demo_keyboard_controls(robot: Robot = None):
    """
    Demostración de los controles de teclado.
    """
    print("DEMO: Controles de Teclado para FABRIK 3D")
    print("=" * 60)
    
    # Crear instancia del sistema
    ik_system = Fabrik_3D.from_robot_yaml(robot)
    ik_system.print_help()
    
    # Ejecutar automáticamente algunos comandos para demostración
    print("\nDemo automático en 3 segundos...")
    
    # Iniciar la visualización
    ik_system.setup_plot()

def calcular_cinematica_directa(robot, thetas):
    """
    Calcula la cinemática directa del robot para obtener la posición del efector final.
    
    Implementación basada en matrices de transformación DH simplificadas para compatibilidad
    con el sistema FABRIK que espera posiciones variables según la configuración articular.
    
    Args:
        robot: Instancia del robot con sus ejes helicoidales
        thetas: Lista de ángulos de las articulaciones
    
    Returns:
        np.array: Posición [x, y, z] del efector final en milímetros (para FABRIK)
    """
    try:
        # Inicializar transformación desde la base
        x, y, z = 0.0, 0.0, 0.0
        
        # Matriz de rotación acumulada (inicialmente identidad)
        R = np.eye(3)
        
        # Verificar que tenemos suficientes articulaciones
        if len(thetas) < len(robot.links):
            print(f"Warning: {len(thetas)} thetas para {len(robot.links)} links")
        
        # Aplicar transformaciones para cada articulación
        for i, link in enumerate(robot.links):
            if i < len(thetas) and hasattr(link, 'length') and link.length > 0:
                theta = thetas[i]
                
                # Crear matriz de rotación para esta articulación
                if hasattr(link, 'joint_axis') and link.joint_axis is not None:
                    axis = np.array(link.joint_axis)
                    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array([0, 0, 1])
                    
                    # Matriz de rotación usando fórmula de Rodrigues
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    
                    # K es la matriz antisimétrica del eje
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    
                    # Fórmula de Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
                    R_local = np.eye(3) + sin_theta * K + (1 - cos_theta) * K @ K
                    
                    # Actualizar rotación acumulada
                    R = R @ R_local
                else:
                    # Asumir rotación en Z si no hay eje definido
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    R_z = np.array([
                        [cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]
                    ])
                    R = R @ R_z
                
                # Vector del link en su marco local (asumiendo que apunta en X)
                link_vector_local = np.array([link.length, 0, 0])
                
                # Transformar el vector del link al marco global
                link_vector_global = R @ link_vector_local
                
                # Agregar la contribución de este link a la posición final
                x += link_vector_global[0]
                y += link_vector_global[1]
                z += link_vector_global[2]
        
        # Escalar de metros (robot TAD) a milímetros (FABRIK)
        result = np.array([x * 1000, y * 1000, z * 1000])
        
        return result
        
    except Exception as e:
        print(f"Error en cinemática directa: {e}")
        # Fallback simple con cinemática planar básica
        if len(thetas) > 0:
            # Usar solo las primeras articulaciones para una aproximación
            base_angle = thetas[0] if len(thetas) > 0 else 0
            
            # Aproximación: considerar que el robot se extiende radialmente
            # con las articulaciones afectando la dirección y extension
            max_reach = 0.5  # ~50% del alcance máximo teórico
            
            # Calcular extensión basada en configuración articular
            extension_factor = 0.3 + 0.4 * abs(np.cos(sum(thetas[:min(3, len(thetas))])))
            
            radius = max_reach * extension_factor
            x = radius * np.cos(base_angle)
            y = radius * np.sin(base_angle)
            z = 0.1 + 0.1 * abs(sum(thetas[1:min(3, len(thetas))]))  # Altura variable
            
            return np.array([x * 1000, y * 1000, z * 1000])
        else:
            return np.array([200, 200, 200])

def generate_smooth_trajectory(robot: Robot):
    """
    Genera una trayectoria suave usando configuraciones aleatorias válidas del robot
    e interpolación con cinemática directa.
    
    Args:
        robot: Instancia del robot cargado desde YAML
        
    Returns:
        list: Lista de posiciones [x, y, z] del efector final
    """
    print("Generando trayectoria suave con configuraciones aleatorias...")
    
    # Preparar datos de animación una vez
    num_waypoints = 5
    frames_per_segment = 40
    thetas_anim_list = []
    waypoints = []
    
    # Generar waypoints aleatorios válidos
    for i in range(num_waypoints - 1):
        if robot is not None:
            theta_waypoint, _ = thetas_aleatorias(robot)
            waypoints.append(theta_waypoint)
        else:
            # Fallback si no tenemos las funciones del robot
            theta_waypoint = np.random.uniform(-np.pi, np.pi, len(robot.joints)-1)
            waypoints.append(theta_waypoint.tolist())
    
    # Cerrar el loop volviendo al primer waypoint
    waypoints.append(waypoints[0])
    
    print(f"Waypoints generados: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i}: {[f'{x:.2f}' for x in wp]}")
    
    # Interpolar suavemente entre waypoints
    for i in range(num_waypoints):
        theta_start = np.array(waypoints[i])
        theta_end = np.array(waypoints[(i + 1) % num_waypoints])
        
        for j in range(frames_per_segment):
            # Parámetro de interpolación con suavizado cosenoidal
            t_param = j / frames_per_segment
            t_smooth = 0.5 - 0.5 * np.cos(t_param * np.pi)
            
            # Interpolación lineal suavizada
            theta_interpolated = theta_start * (1 - t_smooth) + theta_end * t_smooth
            
            # Aplicar límites del robot
            if robot is not None:
                theta_clipped = thetas_limite(robot, theta_interpolated.tolist())
            else:
                theta_clipped = np.clip(theta_interpolated, -np.pi, np.pi).tolist()
            
            thetas_anim_list.append(theta_clipped)
    
    print(f"Configuraciones interpoladas generadas: {len(thetas_anim_list)}")
    
    # Convertir configuraciones articulares a posiciones del efector final
    trajectory = []
    for thetas in thetas_anim_list:
        pos_ef = calcular_cinematica_directa(robot, thetas)
        trajectory.append(pos_ef)
    
    print(f"Trayectoria del efector final calculada: {len(trajectory)} puntos")
    
    return trajectory

def create_trajectory_demo(robot: Robot):
    """
    Crea una demostración con trayectoria automática.
    """
    print("DEMO: Trayectoria Automática 3D")
    print("=" * 50)
    
    
    # Crear instancia del sistema FABRIK
    try:
        ik_system = Fabrik_3D.from_robot_yaml(robot)
        print("Sistema FABRIK inicializado desde YAML")
    except Exception as e:
        print(f"Error cargando FABRIK desde YAML: {e}")
        ik_system = Fabrik_3D()
        print("Usando sistema FABRIK por defecto")
    
    # Generar trayectoria mejorada
    if robot is not None:
        trajectory = generate_smooth_trajectory(robot)
    else:
        # Fallback: trayectoria simple como antes
        print("Usando trayectoria helicoidal de fallback...")
        t_values = np.linspace(0, 4*np.pi, 100)
        trajectory = []
        for t in t_values:
            x = ik_system.limbs_len * 0.3 * np.cos(t)
            y = ik_system.limbs_len * 0.3 * np.sin(t)
            z = ik_system.limbs_len * 0.2 + ik_system.limbs_len * 0.1 * np.sin(t/2)
            trajectory.append(np.array([x, y, z]))

    print("Trayectoria generada:")
    print(f" - {len(trajectory)} puntos")
    print(f" - Rango X: [{min(p[0] for p in trajectory):.1f}, {max(p[0] for p in trajectory):.1f}]")
    print(f" - Rango Y: [{min(p[1] for p in trajectory):.1f}, {max(p[1] for p in trajectory):.1f}]")
    print(f" - Rango Z: [{min(p[2] for p in trajectory):.1f}, {max(p[2] for p in trajectory):.1f}]")
    
    # Exportar la trayectoria a un archivo para visualización, prueba y análisis
    exportar_trayectoria_cartesian(path='trayectoria_new.xyz', trayectoria=trajectory)
    importacion = importar_trayectoria_cartesian(path='trayectoria_new.xyz')

    print("Trayectoria generada:", type(trajectory), trajectory[:3])
    print("Trayectoria importada:", type(importacion), importacion[:3])  # Mostrar solo los primeros 10 puntos

    # Agregar la trayectoria al sistema
    ik_system.demo_trajectory = trajectory
    ik_system.demo_index = 0
    ik_system.demo_active = True
    
    # Modificar el método animate para incluir la demo
    original_animate = ik_system.animate
    
    def animate_with_demo(frame):
        # Si la demo está activa, usar puntos de la trayectoria
        if hasattr(ik_system, 'demo_active') and ik_system.demo_active:
            if hasattr(ik_system, 'demo_trajectory') and hasattr(ik_system, 'demo_index'):
                if ik_system.demo_index < len(ik_system.demo_trajectory):
                    ik_system.target = ik_system.demo_trajectory[ik_system.demo_index].copy()
                    ik_system.demo_index += 1
                else:
                    # Reiniciar la trayectoria
                    ik_system.demo_index = 0
        
        return original_animate(frame)
    
    ik_system.animate = animate_with_demo
    ik_system.setup_plot()

if __name__ == "__main__":
    print("FABRIK 3D - Demos de Control")
    robot = cargar_robot_desde_yaml('config/robot-niryo.yaml')
    
    while True:
        print("=" * 60)
        print("1. Demo de controles de teclado (robot-niryo.yaml)")
        print("2. Demo con trayectoria automática (robot-niryo.yaml)")
        print("0. Salir")
        print("=" * 60)
        opcion = input("Selecciona demo (1, 2 o 0): ").strip()

        if opcion == "1":
            demo_keyboard_controls(robot)
        elif opcion == "2":
            create_trajectory_demo(robot)
        elif opcion == "0":
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")