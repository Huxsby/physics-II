"""
Ejemplos de uso para la visualización del robot manipulador
"""

from class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite, filtrar_configuraciones, Robot      # Para cargar el robot desde un archivo YAML
from class_robot_plotter import plot_robot, guardar_animacion                                           # Para la visualización del robot
import numpy as np                                                                                      # Para la manipulación de matrices
import matplotlib.pyplot as plt                                                                         # Para la visualización
from problema_cinematico_inverso_gen import CinematicaInversa, CinematicaDirecta                        # Para la cinemática inversa
from class_rotaciones import Rp2Trans, Euler2R                                                          # Para la matriz de transformación homogénea
from class_helicoidales import calcular_M_generalizado                                                  # Para la matriz de transformación homogénea
from class_jacobian import calcular_jacobiana, prueba_singularidades                                    # Para la matriz jacobiana
import os                                                                                               # Para limpiar la pantalla en Windows/Linux

# Ejemplo 1: Visualización simple
def ejemplo_visualizacion_simple(robot: Robot):
    # Generar una configuración de ángulos en posición neutral
    thetas = np.zeros(len(robot.links))
    
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
    print(f"Visualizando robot con configuración personalizada: {np.round(thetas, 3)}")
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
    
    print(f"Configuración inicial: {np.round(thetas_inicio, 3)}")
    print(f"Configuración final: {np.round(thetas_fin, 3)}")
    
    # Generar frames para la animación
    num_frames = 50
    thetas_anim = []
    puntos_trayectoria = []
    M = calcular_M_generalizado(robot)

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
    print(f"Visualizando robot con cinemática directa: {np.round(thetas, 3)}")
    plot_robot(robot, thetas)

# Ejemplo 7: Cinemática inversa con trayectoria circular
def ejemplo_cinematica_inversa_circular(robot: Robot, nombre_archivo="trayectoria_circular"):
    # Cargar el robot y matriz M
    M = calcular_M_generalizado(robot)
    
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
    initial_ik_guess = np.zeros(len(robot.links))
    Jacobiana_tuple = calcular_jacobiana(robot)
    thetas_iniciales_trayectoria = CinematicaInversa(robot, Jacobiana_tuple, thetas_actuales=initial_ik_guess, p_xyz=initial_point, RPY=[0, np.pi, 0])

    if thetas_iniciales_trayectoria:
        ik_initial_guess_thetas = thetas_iniciales_trayectoria[-1] # Use the last iteration of the first point's solution
        # Add the initial configuration to the animation to ensure the first point is visualized
        thetas_anim.extend(thetas_iniciales_trayectoria) 
    else:
        print(f"Advertencia: Cinemática Inversa falló para el punto inicial ({initial_point}). "
              "Usando configuración cero como punto de partida.")
        ik_initial_guess_thetas = np.zeros(len(robot.links))
        # Optionally, add the zero configuration if you want the animation to start from there
        # in case of initial IK failure.
        # thetas_anim.append(ik_initial_guess_thetas) 

    print(f"Punto inicial cinemática inversa de la animación: {ik_initial_guess_thetas}")

    for punto_idx, punto in enumerate(puntos): # Using enumerate for clearer logging on failure
        # Orientación fija (ejemplo: orientación hacia abajo)
        # Tsd = Rp2Trans(Euler2R(0, np.pi, 0), punto)
        
        # Use the solution from the previous point (or initial guess) as the 'thetas_actuales' for the IK solver.
        thetas_follower = CinematicaInversa(robot, Jacobiana_tuple, thetas_actuales=ik_initial_guess_thetas, p_xyz=punto, RPY=[0, np.pi, 0])
        
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

# Ejemplo 8: Animación de articulaciones prismáticas
def ejemplo_animacion_prismatica(robot: Robot):
    # Encuentra los índices de las articulaciones prismáticas
    prismatic_joint_indices = [i for i, link in enumerate(robot.links) if link.tipo == "prismatic"]
    
    if not prismatic_joint_indices:
        print("El robot no tiene articulaciones prismáticas.")
        return
    
    # Genera una configuración base con todas las articulaciones en cero
    thetas = np.zeros(len(robot.links))
    
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

# Ejemplo 9: Visualización de configuración singular
def ejemplo_configuracion_singular(robot: Robot):
    # Configuración singular específica
    # {t0: 0, t1: -1.57079632679490, t2: 0, t3: 0, t4: 0, t5: 0, t6: 0}
    # Corresponde a theta2 = -pi/2, y el resto 0.
    # Asumiendo que el robot tiene 7 articulaciones como en otros ejemplos.
    # Si el robot.yaml tiene un número diferente de articulaciones, esto necesitará ajuste.
    thetas_singular = [0, -np.pi/2, 0, 0, 0, 0, 0]
    
    # Asegurar que los ángulos estén dentro de los límites (aunque para singularidad, esto es más conceptual)
    # Si el robot tiene menos de 7 articulaciones, ajustar la longitud de thetas_singular
    if len(robot.links) < len(thetas_singular):
        thetas_singular = thetas_singular[:len(robot.links)]
    elif len(robot.links) > len(thetas_singular):
        # Si el robot tiene más articulaciones, rellenar con ceros
        thetas_singular.extend([0] * (len(robot.links) - len(thetas_singular)))

    thetas_singular_np = np.array(thetas_singular)
    thetas_singular_np = thetas_limite(robot, thetas_singular_np)
    
    # Visualizar el robot
    print(f"Visualizando robot en configuración singular: {np.round(thetas_singular_np, 3)}")
    plot_robot(robot, thetas_singular_np)

# Ejemplo 10: Visualizar multiples configuraciones en subplots
def ejemplo_multiples_configuraciones_subplots(robot: Robot):
    # Generar múltiples configuraciones aleatorias
    num_configuraciones = 4  # Limit to 4 for a 2x2 grid
    all_thetas = []
    
    for _ in range(num_configuraciones):
        thetas, _ = thetas_aleatorias(robot)
        all_thetas.append(thetas)
    
    # Crear una figura y subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Define el ángulo de vista que quieres usar para todos los subplots
    view_angles = (30, 45)  # Elevación y azimut
    
    # Iterar a través de las configuraciones y crear subplots
    for i, thetas in enumerate(all_thetas):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')  # Grid de 2x2
        plot_robot(robot, thetas, ax=ax, show=False, view_angles=view_angles)
        ax.set_title(f'Configuración {i+1}')
    
    plt.tight_layout()
    plt.show()

# Ejemplo 11: Prueba de singularidades
def ejemplo_prueba_singularidades(robot: Robot):
    # Calcular la Jacobiana
    Jacobian, thetas_s = calcular_jacobiana(robot)
    
    # Encontrar configuraciones singulares
    singular_configurations = prueba_singularidades(Jacobian, thetas_s, show=False)
    
    singular_configurations = filtrar_configuraciones(robot, singular_configurations)

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
            print(f"\tConfiguración {i+1}: {np.round(config, 2)}")
            # Convert symbolic values to float
            theta_values = [float(val) for val in config]
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            plot_robot(robot, theta_values, ax=ax, show=False, view_angles=view_angles)
            ax.set_title(f'Configuración singular {i+1} {np.round(config, 2)}')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo se encontraron configuraciones singulares.")

def menu_plotter():
    # Cargar el robot desde un archivo YAML
    def limpiar_pantalla():
        """Limpia la pantalla de la consola."""
        input("\033[93mPresione Enter para continuar...\033[0m")
        os.system('cls' if os.name == 'nt' else 'clear')

    robot = cargar_robot_desde_yaml("robot.yaml")
    
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
        print("7. Configuración singular")
        print("8. Animación de articulaciones prismáticas")
        print("9. Cinemática inversa con trayectoria circular (robot.yaml)")
        print("10. Cinemática inversa con trayectoria circular (robot-niryo.yaml)")
        print("11. Prueba de singularidades (robot.yaml)")
        print("12. Probar todos los graficos y animaciones")
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
            print("Ejecutando: Configuración singular")
            ejemplo_configuracion_singular(robot)
       
        elif opcion == '8':
            print("Ejecutando: Animación de articulaciones prismáticas")
            ejemplo_animacion_prismatica(robot)
        
        elif opcion == '9':
            print("Ejecutando: Cinemática inversa con trayectoria circular (robot.yaml)")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_brazo_dron")
        
        elif opcion == '10':
            print("Ejecutando: Cinemática inversa con trayectoria circular (robot-niryo.yaml)")
            robot = cargar_robot_desde_yaml("robot-niryo.yaml")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_niryo")
            robot = cargar_robot_desde_yaml("robot.yaml") # Reset robot to default
        
        elif opcion == '11':
            print("Ejecutando: Prueba de singularidades")
            ejemplo_prueba_singularidades(robot)

        elif opcion == '12':
            print("Ejecutando: Probar todos los graficos y animaciones")
            ejemplo_visualizacion_simple(robot)
            ejemplo_configuracion_personalizada(robot)
            ejemplo_multiples_vistas(robot)
            ejemplo_animacion(robot)
            ejemplo_trayectoria(robot)
            ejemplo_cinematica_directa(robot)
            ejemplo_configuracion_singular(robot)
            ejemplo_animacion_prismatica(robot)
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_brazo_dron")
            robot = cargar_robot_desde_yaml("robot-niryo.yaml")
            ejemplo_cinematica_inversa_circular(robot, nombre_archivo="trayectoria_circular_niryo")
            robot = cargar_robot_desde_yaml("robot.yaml") # Reset robot to default
            ejemplo_prueba_singularidades(robot)
            
        elif opcion == '0':
            print("Saliendo del programa.")
            limpiar_pantalla()
            break # Fin del bucle
        
        else:
            print("Opción no válida. Intente de nuevo.")
        
        limpiar_pantalla()

if __name__ == "__main__":
    menu_plotter()