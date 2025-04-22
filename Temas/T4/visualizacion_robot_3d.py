"""
Módulo para la visualización 3D del Robot Niryo One utilizando matplotlib
con etiquetas que identifican cada pieza según la nomenclatura real del robot
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Importamos las clases definidas en el archivo original
from class_robot_structure import Robot, Link, cargar_robot_desde_yaml

class Arrow3D(FancyArrowPatch):
    """Clase para dibujar flechas en 3D que representan los ejes de las articulaciones"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def visualizar_robot_3d_actualizado(robot, angulos=None):
    """
    Visualiza el robot Niryo One en 3D utilizando matplotlib con etiquetas para cada pieza.
    
    Args:
        robot (Robot): Instancia de la clase Robot con la estructura del Niryo One
        angulos (list, optional): Lista de ángulos de las articulaciones en radianes.
                                Por defecto es None, lo que representa la posición cero.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Definimos colores para cada eslabón basados en la imagen real
    colores = {
        'Base': '#FF7F00',      # Naranja
        'Hombro': '#FFFF33',    # Amarillo
        'Brazo': '#4DAF4A',     # Verde
        'Codo': '#4DAF4A',      # Verde (similar al brazo)
        'Antebrazo': '#377EB8', # Azul
        'Muñeca': '#984EA3',    # Púrpura
        'Pinza-Mano': '#E41A1C' # Rojo
    }
    
    # Si no se proporcionan ángulos, asumimos que todos son cero
    if angulos is None:
        angulos = [0] * len(robot.links)
    
    # Matriz de transformación acumulada
    T_acum = np.eye(4)
    
    # Dibujamos la base del robot (cilindro)
    base_height = robot.links[0].length
    radio_base = 0.05
    z = np.linspace(0, base_height, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radio_base * np.cos(theta_grid)
    y_grid = radio_base * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=colores['Base'], alpha=0.7)
    
    # Ejes coordenados globales
    longitud_eje = 0.1
    # Eje X - Rojo
    ax.quiver(0, 0, 0, longitud_eje, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.text(longitud_eje*1.1, 0, 0, "X", color='red', size=12)
    # Eje Y - Verde
    ax.quiver(0, 0, 0, 0, longitud_eje, 0, color='green', arrow_length_ratio=0.1)
    ax.text(0, longitud_eje*1.1, 0, "Y", color='green', size=12)
    # Eje Z - Azul
    ax.quiver(0, 0, 0, 0, 0, longitud_eje, color='blue', arrow_length_ratio=0.1)
    ax.text(0, 0, longitud_eje*1.1, "Z", color='blue', size=12)
    
    # Para cada eslabón
    joint_positions = []  # Almacenaremos las posiciones de las articulaciones para las etiquetas
    
    for i, link in enumerate(robot.links):
        # Representamos la articulación como una esfera
        joint_position = T_acum[:3, 3]
        ax.scatter(*joint_position, color=colores.get(link.id, 'black'), s=100)
        joint_positions.append(joint_position)
        
        # Extraemos los datos del eslabón
        joint_coords = np.array(link.joint_coords)
        joint_axis = np.array(link.joint_axis)
        length = link.length
        
        # Calculamos la matriz de rotación según el eje y ángulo de la articulación
        if link.tipo == "revolute" and not np.array_equal(joint_axis, [0, 0, 0]):
            # Matriz de rotación para articulaciones de revolución
            angle = angulos[i]
            # Matriz de rotación según el eje (utilizando la fórmula de Rodrigues)
            axis = joint_axis / np.linalg.norm(joint_axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            # Matriz de transformación homogénea
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = joint_coords
        else:  # prismatic o sin rotación
            # Para articulaciones prismáticas o sin rotación
            T = np.eye(4)
            T[:3, 3] = joint_coords
        
        # Actualizamos la matriz de transformación acumulada
        T_acum = np.dot(T_acum, T)
        
        # Calculamos el vector del eslabón
        if i < len(robot.links) - 1:
            next_joint_coords = np.array(robot.links[i+1].joint_coords)
            end_point = T_acum[:3, 3] + np.dot(T_acum[:3, :3], next_joint_coords)
        else:
            # Para el último eslabón, usamos su orientación y longitud
            orientation = np.array(link.orientation)
            end_point = T_acum[:3, 3] + np.dot(T_acum[:3, :3], orientation * length)
        
        # Dibujamos el eslabón como un cilindro
        start_point = T_acum[:3, 3]
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], 
                color=colores.get(link.id, 'black'), linewidth=5, label=f'Eslabón {link.id}')
        
        # Dibujamos el eje de rotación si no es [0,0,0]
        if not np.array_equal(joint_axis, [0, 0, 0]):
            axis_length = 0.08
            axis_vector = joint_axis * axis_length
            arrow = Arrow3D([start_point[0], start_point[0] + axis_vector[0]],
                            [start_point[1], start_point[1] + axis_vector[1]],
                            [start_point[2], start_point[2] + axis_vector[2]],
                            mutation_scale=15, lw=2, arrowstyle='-|>', color='k')
            ax.add_artist(arrow)
    
    # Añadimos un punto para representar el efector final
    ax.scatter(*end_point, color='black', s=100, marker='X', label='Efector Final')
    
    # Añadimos etiquetas con los nombres de cada pieza
    for i, link in enumerate(robot.links):
        # Posición para la etiqueta (con un pequeño offset para mejor visualización)
        if i < len(joint_positions):
            pos = joint_positions[i]
            # Añadimos un offset para que las etiquetas no se superpongan con las articulaciones
            offset = np.array([0.02, 0.02, 0.02])
            if i == 0:  # Para la base, la etiqueta va un poco más abajo
                offset = np.array([0.02, 0.02, -0.02])
            
            ax.text(pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2], 
                   link.id, color='black', fontsize=10, 
                   backgroundcolor='white', alpha=0.7)
    
    # Añadimos una etiqueta para el efector final
    ax.text(end_point[0] + 0.02, end_point[1] + 0.02, end_point[2] + 0.02, 
           "Efector Final", color='black', fontsize=10, 
           backgroundcolor='white', alpha=0.7)
    
    # Configuración de los ejes
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Ajustamos los límites de los ejes para que se vea bien el robot
    max_dim = 0.6  # Ajusta según el tamaño del robot
    ax.set_xlim([-max_dim/2, max_dim])
    ax.set_ylim([-max_dim/2, max_dim/2])
    ax.set_zlim([0, max_dim])
    
    # Título y leyenda
    ax.set_title('Visualización 3D del Robot Niryo One')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Configuramos la vista para que sea similar a la imagen de referencia
    ax.view_init(elev=20, azim=30)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Cargamos el robot desde el archivo YAML
    robot = cargar_robot_desde_yaml("robot.yaml")
    
    if robot:
        # Visualizamos el robot en su posición cero con etiquetas
        print("Visualizando el robot Niryo One en su posición cero con etiquetas actualizadas...")
        visualizar_robot_3d_actualizado(robot)
    else:
        print("No se pudo cargar el robot. Verifica que el archivo robot.yaml existe.")