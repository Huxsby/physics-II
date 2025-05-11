"""
Ejemplos de uso para la visualización del robot manipulador
"""

from class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite, Robot
from class_robot_plotter import plot_robot
import numpy as np
import scipy
import matplotlib.pyplot as plt
from problema_cinematico_inverso_gen import CinematicaInversa, CinematicaDirecta
from class_rotaciones import Rp2Trans, Euler2R
from class_helicoidales import calcular_M_generalizado

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
def ejemplo_animacion(robot: Robot):
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
    
    if anim:
        print("Guardando animación en 'animacion_dos_configuraciones.mp4'...")
        anim.save("animacion_dos_configuraciones.mp4", writer="ffmpeg", fps=30, dpi=225) # dpi=225 para altura de 1080px si la figura es de 6.4x4.8 pulgadas (predeterminado Matplotlib)
        print("Animación guardada.")
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
def ejemplo_cinematica_inversa_circular(robot: Robot):
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
    thetas_iniciales_trayectoria = CinematicaInversa(robot, thetas_actuales=initial_ik_guess, p_xyz=initial_point, RPY=[0, np.pi, 0])

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

    input(f"Presione Enter para continuar con la animación... {ik_initial_guess_thetas}")

    for punto_idx, punto in enumerate(puntos): # Using enumerate for clearer logging on failure
        # Orientación fija (ejemplo: orientación hacia abajo)
        # Tsd = Rp2Trans(Euler2R(0, np.pi, 0), punto)
        
        # Use the solution from the previous point (or initial guess) as the 'thetas_actuales' for the IK solver.
        thetas_follower = CinematicaInversa(robot, thetas_actuales=ik_initial_guess_thetas, p_xyz=punto, RPY=[0, np.pi, 0])
        
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
    anim.save("trayectoria_circular.mp4", writer="ffmpeg", fps=30, dpi=225) # dpi=225 para altura de 1080px si la figura es de 6.4x4.8 pulgadas (predeterminado Matplotlib)
    plt.close()

if __name__ == "__main__":
    # Cargar el robot desde un archivo YAML
    robot = cargar_robot_desde_yaml("robot.yaml")

    print("\n1. Ejemplo de visualización simple")
    ejemplo_visualizacion_simple(robot)
    
    print("\n2. Ejemplo de configuración personalizada")
    ejemplo_configuracion_personalizada(robot)
    
    print("\n3. Ejemplo de múltiples vistas")
    ejemplo_multiples_vistas(robot)
    
    print("\n4. Ejemplo de animación entre dos configuraciones")
    ejemplo_animacion(robot)
    
    print("\n5. Ejemplo de trayectoria con múltiples puntos")
    ejemplo_trayectoria(robot)
    
    print("\n6. Ejemplo de cinemática directa")
    ejemplo_cinematica_directa(robot)
    
    # Cargar el robot Niryo desde un archivo YAML, errores persistentes en Cinematica Inversa para Prismaticas
    robot = cargar_robot_desde_yaml("niryo-robot.yaml")
    print("\n7. Ejemplo de cinemática inversa con trayectoria circular, con robot Niryo")
    ejemplo_cinematica_inversa_circular(robot)