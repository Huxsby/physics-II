"""
Ejemplos de uso para la visualización del robot manipulador
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.animation.class_robot_plotter import plot_robot, guardar_animacion, graficar_workspace                         # Para la visualización del robot
from src.calculations.problema_cinematico_inverso_gen import CinematicaInversa, CinematicaDirecta                       # Para la cinemática inversa
from src.core.class_robot_structure import (Robot,
                                            str_config,
                                            thetas_aleatorias,
                                            thetas_limite,
                                            filtrar_configuraciones,
                                            cargar_robot_desde_yaml)  # Para la manipulación de configuraciones
from src.calculations.class_rotaciones import Rp2Trans, Euler2R                                                          # Para la matriz de transformación homogénea
from src.calculations.class_jacobian import calcular_jacobiana, prueba_singularidades                                    # Para la matriz jacobiana
import numpy as np                                                                                      # Para la manipulación de matrices
import matplotlib.pyplot as plt                                                                         # Para la visualización
import os                                                                                               # Para limpiar la pantalla en Windows/Linux

# Ejemplo 1: Visualización simple
def ejemplo_visualizacion_simple(robot: Robot):
    # Generar una configuración de ángulos en posición neutral
    thetas = np.zeros(robot.num_links)
    
    # Visualizar el robot
    print("Visualizando robot en posición neutral...")
    plot_robot(robot, thetas)

# Ejemplo 2: Visualización con configuración personalizada
def ejemplo_configuracion_personalizada(robot: Robot):
    # Configuración personalizada (ajustar según los límites de las articulaciones)
    thetas = [0.5, -0.5, 0.7, 1.0, 0.5, 0.05, 1.2]
    
    # Asegurar que los ángulos estén dentro de los límites
    thetas = thetas_limite(robot, thetas)
    
    # Visualizar el robot
    print(f"Visualizando robot con configuración personalizada: {str_config(thetas, 3)}")
    plot_robot(robot, thetas)

# Ejemplo 3: Visualización con múltiples vistas
def ejemplo_multiples_vistas(robot: Robot):
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
def ejemplo_animacion(robot: Robot, nombre_archivo="animacion_dos_configuraciones"):
    # Generar dos configuraciones aleatorias
    thetas_inicio, _ = thetas_aleatorias(robot)
    thetas_fin, _ = thetas_aleatorias(robot)
    
    print(f"Configuración inicial: {str_config(thetas_inicio, 3)}")
    print(f"Configuración final: {str_config(thetas_fin, 3)}")
    
    # Generar frames para la animación
    num_frames = 50
    thetas_anim = []
    puntos_trayectoria = []
    M = robot.M

    for i in range(num_frames):
        t = i / (num_frames - 1)
        thetas_i = thetas_inicio * (1 - t) + thetas_fin * t
        thetas_anim.append(thetas_i)
        # Calcular la posición del efector final para la trayectoria
        M_actual = CinematicaDirecta(robot.ejes_helicoidales, thetas_i, M)
        puntos_trayectoria.append(M_actual[:3, 3])
    
    puntos_trayectoria = np.array(puntos_trayectoria)

    # Visualizar la animación
    print(f"Animando movimiento con {num_frames} frames...")
    fig, ax, anim = plot_robot(robot, thetas_anim, animation_speed=50, show=True, trayectoria=puntos_trayectoria)
    
    guardar_animacion(anim, nombre_archivo) # dpi=225 para altura de 1080px si la figura es de 6.4x4.8 pulgadas (predeterminado Matplotlib)
    plt.close(fig)

# Ejemplo 5: Visualización con trayectoria
def ejemplo_trayectoria(robot: Robot):
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
def ejemplo_cinematica_directa(robot: Robot):
    # Configuración personalizada (ejemplo)
    thetas = [0.5, -0.5, 0.7, 1.0, 0.5, 0.05, 1.2]
    thetas = thetas_limite(robot, thetas)
    
    # Visualizar el robot
    print(f"Visualizando robot con cinemática directa: {str_config(thetas, 3)}")
    plot_robot(robot, thetas)

# Ejemplo 7: Animación de articulaciones prismáticas
def ejemplo_animacion_prismatica(robot: Robot):
    # Encuentra los índices de las articulaciones prismáticas
    prismatic_joint_indices = [i for i, link in enumerate(robot.links) if link.tipo == "prismatic"]
    
    if not prismatic_joint_indices:
        print("El robot no tiene articulaciones prismáticas.")
        return
    
    # Genera una configuración base con todas las articulaciones en cero
    thetas = np.zeros(robot.num_links)
    
    # Define el número de frames para la animación
    num_frames = 50
    
    # Crea una lista para almacenar las configuraciones de cada frame
    thetas_anim = []
    
    # Para cada frame, ajusta la posición de las articulaciones prismáticas
    for i in range(num_frames):
        frame_thetas = np.copy(thetas)  # Comienza con la configuración base
        
        for joint_index in prismatic_joint_indices:
            # Obtiene los límites de la articulación prismática
            lower_limit, upper_limit = robot.links[joint_index].joint_limits
            
            # Calcula el valor de la articulación prismática para este frame
            # Esto crea una animación lineal desde el límite inferior hasta el superior
            t = i / (num_frames - 1)
            frame_thetas[joint_index] = lower_limit + t * (upper_limit - lower_limit)
        
        thetas_anim.append(frame_thetas)
    
    # Visualiza la animación
    print(f"Animando articulaciones prismáticas con {num_frames} frames...")
    plot_robot(robot, thetas_anim, animation_speed=50, show=True)

# Ejemplo 8 y 9: Cinemática inversa con trayectoria circular
def ejemplo_cinematica_inversa_circular(robot: Robot, nombre_archivo="trayectoria_circular"):
    # Generar puntos en una trayectoria circular
    num_puntos = 90
    radio = 0.15
    z = 0.3
    angulos = np.linspace(0, 2*np.pi, num_puntos)
    puntos = np.array([[radio*np.cos(theta), radio*np.sin(theta), z] for theta in angulos])
    
    # # Graficar la trayectoria deseada
    # fig_traj = plt.figure()
    # ax_traj = fig_traj.add_subplot(111, projection='3d')
    # ax_traj.plot(puntos[:, 0], puntos[:, 1], puntos[:, 2], 'r--', label='Trayectoria Circular Deseada')
    # ax_traj.set_xlabel('X')
    # ax_traj.set_ylabel('Y')
    # ax_traj.set_zlabel('Z')
    # ax_traj.set_title('Trayectoria Circular Deseada')
    # ax_traj.legend()
    # plt.show()


    # Calcular configuraciones articulares para cada punto
    thetas_anim = []
    # Initialize with a default guess (e.g., zero configuration).
    # This will be updated with the last successful IK solution to provide continuity.
    # Calculate initial thetas for the first point of the trajectory
    # to ensure the animation starts smoothly.
    initial_point = puntos[0]
    # Initial guess for the first point can be zeros or a neutral configuration

    # Calcular
    initial_ik_guess = np.zeros(robot.num_links)
    Jacobiana_tuple = calcular_jacobiana(robot)
    thetas_iniciales_trayectoria = CinematicaInversa(robot, Jacobiana_tuple, thetas_actuales=initial_ik_guess, p_xyz=initial_point, RPY=[0, np.pi, 0], show=False)

    if thetas_iniciales_trayectoria:
        ik_initial_guess_thetas = thetas_iniciales_trayectoria[-1] # Use the last iteration of the first point's solution
        # Add the initial configuration to the animation to ensure the first point is visualized
        thetas_anim.extend(thetas_iniciales_trayectoria) 
    else:
        print(f"Advertencia: Cinemática Inversa falló para el punto inicial ({initial_point}). "
              "Usando configuración cero como punto de partida.")
        ik_initial_guess_thetas = np.zeros(robot.num_links)
        # Optionally, add the zero configuration if you want the animation to start from there
        # in case of initial IK failure.
        # thetas_anim.append(ik_initial_guess_thetas) 

    print(f"Punto inicial cinemática inversa de la animación: {ik_initial_guess_thetas}")

    for punto_idx, punto in enumerate(puntos): # Using enumerate for clearer logging on failure
        # Orientación fija (ejemplo: orientación hacia abajo)
        # Tsd = Rp2Trans(Euler2R(0, np.pi, 0), punto)
        
        # Use the solution from the previous point (or initial guess) as the 'thetas_actuales' for the IK solver.
        thetas_follower = CinematicaInversa(robot, Jacobiana_tuple, thetas_actuales=ik_initial_guess_thetas, p_xyz=punto, RPY=[0, np.pi, 0], show=False)
        
        if thetas_follower: # Assumes thetas_follower is a list of configurations (iterations) on success.
            thetas_anim.extend(thetas_follower)  # Add all iterations to the animation.
            ik_initial_guess_thetas = thetas_follower[-1] # Update the guess for the next point.
        else:
            # Handle cases where Inverse Kinematics fails to find a solution.
            print(f"Advertencia: Cinemática Inversa falló para el punto {punto_idx} ({punto}). "
                  f"La animación podría tener un salto o usar la configuración anterior como base para el siguiente punto.")
            # ik_initial_guess_thetas remains unchanged, so the next IK attempt will start
            # from the last known good configuration.
            
    # # Suavizar trayectoria si es necesario
    # if len(thetas_anim) < 100:
    #     print("Aplicando interpolación para suavizar...")
    #     from scipy.interpolate import CubicSpline
    #     t_original = np.linspace(0, 1, len(thetas_anim))
    #     t_nuevo = np.linspace(0, 1, 100)
    #     thetas_anim = CubicSpline(t_original, thetas_anim, axis=0)(t_nuevo)
    
    # Visualizar y guardar animación
    print("Animando trayectoria circular...")
    fig, ax, anim = plot_robot(robot, thetas_anim, animation_speed=50, show=True, trayectoria=puntos)
    guardar_animacion(anim, nombre_archivo) # dpi=225 para altura de 1080px si la figura es de 6.4x4.8 pulgadas (predeterminado Matplotlib)
    plt.close()

# Ejemplo 10: Prueba de singularidades
def ejemplo_prueba_singularidades(robot: Robot):
    # Calcular la Jacobiana
    Jacobian, thetas_s = calcular_jacobiana(robot)
    
    # Encontrar configuraciones singulares
    singular_configurations = prueba_singularidades(Jacobian, thetas_s, show=False)
    singular_configurations = filtrar_configuraciones(robot, singular_configurations) # Filtrar configuraciones fuera de límites

    if singular_configurations:
        print("\nConfiguraciones singulares encontradas:")
        
        # Determine grid dimensions based on number of configurations
        num_configs = len(singular_configurations)
        rows = int(np.ceil(np.sqrt(num_configs)))
        cols = int(np.ceil(num_configs / rows))
        
        # Create figure for subplots
        fig = plt.figure(figsize=(cols*4, rows*4))
        
        # Set default view angle for all subplots
        view_angles = (30, 45)  # Elevation and azimuth
        
        # Plot each singular configuration in a subplot
        for i, config in enumerate(singular_configurations):
            print(f"\tConfiguración {i+1}: {str_config(config, 2)}")
            # Convert symbolic values to float
            theta_values = [float(val) for val in config]
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            plot_robot(robot, theta_values, ax=ax, show=False, view_angles=view_angles)
            ax.set_title(f'Configuración singular {i+1} {str_config(config, 2)}')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo se encontraron configuraciones singulares.")

def menu_plotter():
    # Cargar el robot desde un archivo YAML
    def limpiar_pantalla(stop=True):
        """Limpia la pantalla de la consola."""
        if stop: input("\033[93mPresione Enter para continuar...\033[0m")
        os.system('cls' if os.name == 'nt' else 'clear')

    robot = cargar_robot_desde_yaml("config/robot.yaml")
    
    while True:
        print("\n" + "="*90)    # Separador
        print(" "*32 + "MENÚ DE EJEMPLOS DE VISUALIZACIÓN")
        print("="*90)   # Separador
        print("1. Visualización simple")
        print("2. Configuración personalizada")
        print("3. Múltiples vistas")
        print("4. Animación entre dos configuraciones")
        print("5. Trayectoria con múltiples puntos")
        print("6. Cinemática directa")
        print("7. Animación de articulaciones prismáticas")
        print("8. Cinemática inversa con trayectoria circular (robot.yaml)")
        print("9. Cinemática inversa con trayectoria circular (robot-niryo.yaml)")
        print("10. Prueba de singularidades (robot.yaml)")
        print("11. Probar todos los graficos y animaciones")
        print("-"*90)   # Separador
        print("0. Salir")

        opcion = input("Seleccione un ejemplo (0-12): ")

        if opcion == '1':
            print("Ejecutando: Visualización simple")
            ejemplo_visualizacion_simple(robot)
  
        elif opcion == '2':
            print("Ejecutando: Configuración personalizada")
            ejemplo_configuracion_personalizada(robot)
   
        elif opcion == '3':
            print("Ejecutando: Múltiples vistas")
            ejemplo_multiples_vistas(robot)
    
        elif opcion == '4':
            print("Ejecutando: Animación entre dos configuraciones")
            ejemplo_animacion(robot)
    
        elif opcion == '5':
            print("Ejecutando: Trayectoria con múltiples puntos")
            ejemplo_trayectoria(robot)
    
        elif opcion == '6':
            print("Ejecutando: Cinemática directa")
            ejemplo_cinematica_directa(robot)
        
        elif opcion == '7':
            print("Ejecutando: Animación de articulaciones prismáticas")
            ejemplo_animacion_prismatica(robot)
        
        elif opcion == '8':
            print("Ejecutando: Cinemática inversa con trayectoria circular (robot.yaml)")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_brazo_dron")
        
        elif opcion == '9':
            print("Ejecutando: Cinemática inversa con trayectoria circular (robot-niryo.yaml)")
            robot = cargar_robot_desde_yaml("config/robot-niryo.yaml")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_niryo")
            robot = cargar_robot_desde_yaml("config/robot.yaml") # Reset robot to default
        
        elif opcion == '10':
            print("Ejecutando: Prueba de singularidades")
            ejemplo_prueba_singularidades(robot)

        elif opcion == '11':
            print("Ejecutando: Probar todos los graficos y animaciones")
            ejemplo_visualizacion_simple(robot)
            ejemplo_configuracion_personalizada(robot)
            ejemplo_multiples_vistas(robot)
            ejemplo_animacion(robot)
            ejemplo_trayectoria(robot)
            ejemplo_cinematica_directa(robot)
            ejemplo_animacion_prismatica(robot)
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_brazo_dron")
            robot = cargar_robot_desde_yaml("config/robot-niryo.yaml")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_niryo")
            robot = cargar_robot_desde_yaml("config/robot.yaml") # Reset robot to default
            ejemplo_prueba_singularidades(robot)
            
        elif opcion == '0':
            print("Saliendo del programa.")
            limpiar_pantalla()
            break # Fin del bucle
        
        else:
            print("Opción no válida. Intente de nuevo.")
        
        limpiar_pantalla()

""" Ejemplos de workspace """

def menu_graficar_workspace():
    def limpiar_pantalla(stop=True):
        """Limpia la pantalla de la consola."""
        if stop: input("\033[93mPresione Enter para continuar...\033[0m")
        os.system('cls' if os.name == 'nt' else 'clear')

    robot = cargar_robot_desde_yaml("config/robot.yaml")

    # Preparar datos de animación una vez
    num_waypoints = 5
    frames_per_segment = 40
    thetas_anim_list = []
    waypoints = []
    for i in range(num_waypoints - 1):
        theta_waypoint, _ = thetas_aleatorias(robot)
        waypoints.append(theta_waypoint)
    waypoints.append(waypoints[0])
    for i in range(num_waypoints):
        theta_start = np.array(waypoints[i])
        theta_end = np.array(waypoints[(i + 1) % num_waypoints])
        for j in range(frames_per_segment):
            t_param = j / frames_per_segment
            t_smooth = 0.5 - 0.5 * np.cos(t_param * np.pi)
            theta_interpolated = theta_start * (1 - t_smooth) + theta_end * t_smooth
            theta_clipped = thetas_limite(robot, theta_interpolated.tolist())
            thetas_anim_list.append(theta_clipped)

    while True:
        print("\n" + "="*90)
        print(" "*30 + "MENÚ DE EJEMPLOS DE WORKSPACE")
        print("="*90)
        print("1. Visualización estática")
        print("2. Espacio de trabajo básico (1k puntos)")
        print("3. Espacio de trabajo -z")
        print("4. Espacio de trabajo +y (solo frontera)")
        print("5. Animación simple de trayectoria")
        print("6. Workspace con animación superpuesta")
        print("7. Workspace +x con animación (solo frontera)")
        print("8. Workspace -y (solo frontera)")
        print("9. Workspace completo (estático, 1k puntos)")
        print("10. Workspace con animación (10k puntos, solo frontera)")
        print("11. Workspace con animación (10M puntos, solo frontera)")
        print("12. Probar todos los ejemplos de workspace")
        print("-"*90)
        print("0. Volver al menú principal")

        opcion = input("Seleccione un ejemplo (0-12): ")

        if opcion == '1':
            print("\nEjecutando: Visualización estática")
            thetas_static, _ = thetas_aleatorias(robot)
            print(f"Configuración aleatoria: {str_config(thetas_static, 3)}")
            plot_robot(robot, thetas_static)
        elif opcion == '2':
            print("\nEjecutando: Espacio de trabajo básico")
            graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None, subtitle="Visualización básica del espacio de trabajo")
        elif opcion == '3':
            print("\nEjecutando: Espacio de trabajo -z")
            graficar_workspace(robot, N=1000, show_points=True, half_space_axis='-z', subtitle="Espacio de trabajo filtrado para Z < 0")
        elif opcion == '4':
            print("\nEjecutando: Espacio de trabajo +y, sin puntos")
            graficar_workspace(robot, N=1000, show_points=False, half_space_axis='+y', subtitle="Espacio de trabajo filtrado para Y >= 0 (solo frontera)")
        elif opcion == '5':
            print("\nEjecutando: Animación simple")
            plot_robot(robot, thetas_anim_list, animation_speed=50)
        elif opcion == '6':
            print("\nEjecutando: Espacio de trabajo con animación superpuesta y guardado")
            graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None,
                               thetas_anim=thetas_anim_list, animation_speed=50,
                               save_animation_name="ws_anim_test", subtitle="Animación superpuesta en espacio de trabajo completo")
        elif opcion == '7':
            print("\nEjecutando: Espacio de trabajo +x con animación (sin puntos) y guardado")
            graficar_workspace(robot, N=1000, show_points=False, half_space_axis='+x',
                               thetas_anim=thetas_anim_list, animation_speed=50,
                               save_animation_name="ws_mas_x_anim_test", subtitle="Animación en espacio de trabajo filtrado para X >= 0 (solo frontera)")
        elif opcion == '8':
            print("\nEjecutando: Espacio de trabajo -y (sin animación, sin puntos)")
            graficar_workspace(robot, N=1000, show_points=False, half_space_axis='-y', subtitle="Espacio de trabajo filtrado para Y < 0 (solo frontera)")
        elif opcion == '9':
            print("\nEjecutando: Espacio de trabajo completo (sin animación)")
            graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None, subtitle="Visualización completa del espacio de trabajo (estático)")
        elif opcion == '10':
            print("\nEjecutando: Espacio de trabajo completo con animación (10k puntos, solo frontera) y guardado")
            graficar_workspace(robot, N=10000, show_points=False, half_space_axis=None,
                               thetas_anim=thetas_anim_list, animation_speed=50,
                               save_animation_name="ws_anim_final_form", subtitle="Animación en espacio de trabajo (solo frontera)")
        elif opcion == '11':
            print("\nEjecutando: Espacio de trabajo completo con animación (10M puntos, solo frontera) y guardado")
            graficar_workspace(robot, N=10000000, show_points=False, half_space_axis=None,
                               thetas_anim=thetas_anim_list, animation_speed=50,
                               save_animation_name="ws_anim_10M", subtitle="Animación en espacio de trabajo (solo frontera)")
        elif opcion == '12':
            print("\nEjecutando: Probar todos los ejemplos de workspace")
            print("\nEjecutando: Visualización estática"); thetas_static, _ = thetas_aleatorias(robot); print(f"Configuración aleatoria: {str_config(thetas_static, 3)}"); plot_robot(robot, thetas_static); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo básico"); graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None, subtitle="Visualización básica del espacio de trabajo"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo -z"); graficar_workspace(robot, N=1000, show_points=True, half_space_axis='-z', subtitle="Espacio de trabajo filtrado para Z < 0"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo +y, sin puntos"); graficar_workspace(robot, N=1000, show_points=False, half_space_axis='+y', subtitle="Espacio de trabajo filtrado para Y >= 0 (solo frontera)"); limpiar_pantalla()
            print("\nEjecutando: Animación simple"); plot_robot(robot, thetas_anim_list, animation_speed=50); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo con animación superpuesta y guardado"); graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None, thetas_anim=thetas_anim_list, animation_speed=50, save_animation_name="ws_anim_test", subtitle="Animación superpuesta en espacio de trabajo completo"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo +x con animación (sin puntos) y guardado"); graficar_workspace(robot, N=1000, show_points=False, half_space_axis='+x', thetas_anim=thetas_anim_list, animation_speed=50, save_animation_name="ws_mas_x_anim_test", subtitle="Animación en espacio de trabajo filtrado para X >= 0 (solo frontera)"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo -y (sin animación, sin puntos)"); graficar_workspace(robot, N=1000, show_points=False, half_space_axis='-y', subtitle="Espacio de trabajo filtrado para Y < 0 (solo frontera)"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo completo (sin animación)"); graficar_workspace(robot, N=1000, show_points=True, half_space_axis=None, subtitle="Visualización completa del espacio de trabajo (estático)"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo completo con animación (10k puntos, solo frontera) y guardado"); graficar_workspace(robot, N=10000, show_points=False, half_space_axis=None, thetas_anim=thetas_anim_list, animation_speed=50, save_animation_name="ws_anim_final_form", subtitle="Animación en espacio de trabajo (solo frontera)"); limpiar_pantalla()
            print("\nEjecutando: Espacio de trabajo completo con animación (10M puntos, solo frontera) y guardado"); graficar_workspace(robot, N=10000000, show_points=False, half_space_axis=None, thetas_anim=thetas_anim_list, animation_speed=50, save_animation_name="ws_anim_10M", subtitle="Animación en espacio de trabajo (solo frontera)")
        elif opcion == '0':
            print("Fin de los ejemplos de workspace.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")
        
        limpiar_pantalla()

if __name__ == "__main__":

    menu_plotter()
    menu_graficar_workspace()