""""
Clase para la creación de matrices de rotación en 3D utilizando diferentes métodos.
"""

import numpy as np                              # Para operaciones matemáticas
import matplotlib.pyplot as plt                 # Para visualización
import time                                     # Para medir el tiempo de ejecución
from class_matrices import *                    # Importar funciones de matrices

# Funciones de rotación en torno a los ejes de coordenadas
def Rotx(θ):
    return np.array([[1, 0, 0],
                    [0, np.cos(θ), -np.sin(θ)],
                    [0, np.sin(θ), np.cos(θ)]])

def Roty(θ):
    return np.array([[np.cos(θ), 0, np.sin(θ)],
                    [0, 1, 0],
                    [-np.sin(θ), 0, np.cos(θ)]])

def Rotz(θ):
    return np.array([[np.cos(θ), -np.sin(θ), 0],
                    [np.sin(θ), np.cos(θ), 0],
                    [0, 0, 1]])

# Función de rotación genérica
def RotGen(w, θ):
    """
    Matriz de rotación de un ángulo θ entorno a un eje genérico w.
    \nParámetros: w (array) (vector normalizado), θ (float) en radianes
    \nRetorna: matrix (Matriz de rotación de un eje génerico w)
    \nNota: Una matriz de rotación cumple con las propiedades de una matriz de rotación si es ortogonal y su determinante es 1.
    """
    start = time.time()
    w1, w2, w3 = w      # Componentes del vector de rotación   # Convertir a radianes
    s = np.sin(θ)       # Seno de θ
    c = np.cos(θ)       # Coseno de θ
    t = 1 - c           # 1 - cos(θ)
    
    matrix = np.array([[c + w1**2*t, w1*w2*t - w3*s, w1*w3*t + w2*s],
                    [w1*w2*t + w3*s, c + w2**2*t, w2*w3*t - w1*s],
                    [w1*w3*t - w2*s, w2*w3*t + w1*s, c + w3**2*t]])
    print(f"\tTiempo de ejecución de Rot(w, θ) {time.time() - start} segundos")
    return matrix

# Función de rotación con Rodrigues
def RotRodrigues(w, θ):
    """
    Calcula la matriz de rotación de un ángulo θ entorno a un eje w utilizando la fórmula de Rodrigues.
    
    La fórmula de Rodrigues permite transformar una rotación expresada mediante un eje y un ángulo
    en una matriz de rotación en 3D. Esta implementación utiliza la forma:
    Rot(w, θ) = I + sin(θ)W + (1 - cos(θ))W^2
    donde I es la matriz identidad y W es la matriz antisimétrica asociada al vector w.
    
    La matriz resultante pertenece al grupo SO(3) (grupo de rotaciones en 3D).
    
    Parameters:
        w (numpy.ndarray): Vector unitario (normalizado) que representa el eje de rotación.
        θ (float): Ángulo de rotación en radianes.
        
    Returns:
        numpy.ndarray: Matriz de rotación 3x3 correspondiente a la rotación especificada.
        
    Note:
        - El vector w debe estar normalizado (tener módulo 1).
        - La implementación incluye medición del tiempo de ejecución.
        - W es la matriz antisimétrica del vector w, y W^2 es el producto matricial de W consigo misma.
    """
    start = time.time()
    
    # Crear la matriz antisimétrica W
    W = antisimetrica(w)
    
    # Calcular W^2 (producto matricial)
    W2 = np.dot(W, W)
    
    # Aplicar la fórmula de Rodrigues
    I = np.eye(3)
    matrix = I + np.sin(θ) * W + (1 - np.cos(θ)) * W2
    
    print(f"\tTiempo de ejecución de RotRodrigues(w, θ) {time.time() - start} segundos")
    return matrix

# Función de rotación con logaritmica
def LogRot(R):
    """
    Calcula el logaritmo de la matriz de rotación R.
    Parámetros:
    R : matriz de rotación 3x3 (R SO(3))
    Retorna:
    theta : ángulo de rotación en radianes.
    log_R : matriz antisimétrica [hat_omega ]*theta que representa el logaritmo de R.
    Se aplican las siguientes condiciones para robustez numérica:
    - Si trace(R) >= 3, se asume R I y theta = 0.
    - Si trace(R) <= -1, se maneja el caso especial con theta = .
    """
    tr = np.trace(R)
    
    # Caso: R casi es la identidad
    if tr >= 3.0:
        theta = 0.0
        return theta, np.zeros(3)
   
    # Caso especial: theta cercano a
    elif tr <= -1.0:
        theta = np.pi
        # Se selecciona el índice con mayor valor diagonal para evitar indeterminaciones
        diag = np.diag(R)           # Diagonal de R
        idx = np.argmax(diag)       # Índice con mayor valor diagonal
        hat_omega = np.zeros(3)     # Vector de rotación, le asignamos el valor inicial de 0
        
        # Cálculo de la componente correspondiente al índice con mayor valor diagonal
        hat_omega[idx] = np.sqrt((R[idx , idx] - R[(idx+1) %3, (idx+1) %3] + R[(idx +2) %3, (idx +2) %3] + 1) / 2) 
        
        # Evitar división por cero
        if np.abs(hat_omega[idx]) < 1e-6:
            hat_omega[idx] = 1e-6
        hat_omega = hat_omega / np.linalg.norm(hat_omega)
        # Usar la fórmula general; aunque en este caso , s_theta 0,
        # la expresión (R - R.T)/(2* sin(theta)) es válida para theta = .
        log_R = (R - R.T) / (2 * np.sin(theta))
        return theta, antisimetrica_vector(log_R)
    
    # Caso general
    else:
        theta = np.arccos((tr - 1) / 2)
        s_theta = np.sin(theta)
        # Filtrar posibles divisiones por cero
        if np.abs(s_theta) < 1e-6:
            s_theta = 1e-6
        log_R = (R - R.T) / (2 * s_theta)
        return theta, antisimetrica_vector(log_R)

# Rotar un vector selecion automática
def RotarVector(v, eje, θ):
    """
    Automatiza la rotación de un vector v entorno a un eje específico o genérico.
    \nNota: Se decide como rotar el vector según el tipo de dato de eje y su valor.
    \nParámetros: v (array), eje (str) o (array), θ (float) en radianes
    \nRetorna: v' (array)
    """
    if isinstance(eje, np.ndarray) or isinstance(eje, list):
        print("RotarVector: Rotación entorno a un eje genérico.")
        eje_norm = np.array(eje) / np.linalg.norm(eje)                 # Normalizar el vector
        #R = fr.RotGen(eje_norm, θ)                                    # Cálculo mediante la fórmula general
        R = RotRodrigues(eje_norm, θ)                                  # Cálculo mediante la fórmula de Rodrigues
    elif eje == "x":
        R = Rotx(θ)
    elif eje == "y":
        R = Roty(θ)
    elif eje == "z":
        R = Rotz(θ)
    else:
        print("Eje no válido.")
        return None
    return np.dot(R, v) # Producto punto, rotación de v: v'=Rv

# Visualización de rotación
def Visualizar_Rotacion(vector, eje):
        """
        Visualizar la rotación de un vector entorno a un eje especificado.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Dibujar el vector original en rojo
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label='Original')

        # Visualizar el eje de rotación con una línea verde
        if isinstance(eje, np.ndarray) or isinstance(eje, list):
            eje_norm = np.array(eje) / np.linalg.norm(eje)
            ax.quiver(0, 0, 0, eje_norm[0]*5, eje_norm[1]*5, eje_norm[2]*5,  color='g', label='Eje de rotación')
        elif eje in ["x", "y", "z"]:
            if eje == "x":
                ax.quiver(0, 0, 0, 5, 0, 0, color='g', label='Eje X')
            elif eje == "y":
                ax.quiver(0, 0, 0, 0, 5, 0, color='g', label='Eje Y')
            elif eje == "z":
                ax.quiver(0, 0, 0, 0, 0, 5, color='g', label='Eje Z')

        # Rotar el vector en incrementos de 5 grados y dibujar cada vector rotado con gradiente de color
        total_steps = 360 // 5
        cmap = plt.cm.rainbow  # Use rainbow for a full color cycle
        
        for i, angulo in enumerate(range(0, 360, 5)):
            angulo_rad = np.radians(angulo)
            vector_rotado = RotarVector(vector, eje, angulo_rad)
            
            # Calculate color based on progress through rotation (0 to 1)
            color = cmap(i / total_steps)
            
            # Dibujar el vector rotado
            ax.quiver(0, 0, 0, vector_rotado[0], vector_rotado[1], vector_rotado[2], color=color)

        # Establecer los límites de la gráfica
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 8])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()  # Mostrar leyenda
        plt.title("Rotación de vector")
        plt.show()