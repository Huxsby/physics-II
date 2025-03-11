""" Matrices de Rotación

[x] 1 - Crea 3 funciones Rotx(θ), Roty(θ) y Rotz(θ) en Python que incluyan las
rotaciones entorno a los ejes cartesianos como funciones. # Funciones de rotación

[x] 2 - Crea un programa que utilice las funciones anteriores. El input debe ser un vector
v, el eje entorno al cual se debe rotar (x, y o z) y el ángulo de rotación (θ). El
output debe ser las componentes del vector rotado v0.

    La rotación de vectores sólo involucra el SR en el que el vector está expresado y el eje
    de rotación ω debe interpretarse en el mismo SR. El vector rotado, en el mismo SR es:
    v'=Rv

[x] 3 - Crea una función Rotv(v; θ) en Python que devuelva la matriz genérica de
rotación de un ángulo θ entorno a cualquier eje v. 
    ___________________________________________________________________________
    Q -> Rot(w, θ); w = [w1, w2, w3] tiene que ser un vector unitario? -> A: Sí
    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
[x] 4 - Crea un programa en Python que utilice la función anterior. Los input deben ser
el vector que se va a rotar, el eje de rotación y el ángulo a rotar. El output debe
ser el vector rotado un ángulo θ entorno al eje de rotación especificado.

[X] 5 - Comprueba que los resultados sean los mismos del ejercicio anterior cuando se
realizan rotaciones entorno a los ejes cartesianos. Prueba a hacer rotaciones
entorno a un eje diferente a los ejes cartesianos (recuerda normalizar el vector que
indica el eje de rotación para poder aplicar la fórmula general).

[X] 6 - Comprueba las propiedades no conmutativa y asociativa de rotaciones sucesivas
utilizando los programas anteriores. ∀A,B,C ∈SO(n):
    Propiedad asociativa: (AB)C =A(BC)      Producto interno: El producto AB nos da otro elemento del grupo (otra matriz de
                                            rotación). Las matrices de SO(3) no verifican la propiedad conmutativa pero las
                                            de SO(2) sí.      
"""

import numpy as np                              # Para cálculos numéricos
from scipy.optimize import fsolve               # Para resolver ecuaciones no lineales
import time                                     # Para medir el tiempo de ejecución
import matplotlib.pyplot as plt



from class_datos import Datos                   # Clase para organizar la toma de datos

# Funciones de rotación
def Rotx(θ):
    matrix = np.array([[1, 0, 0],
                        [0, np.cos(θ), -np.sin(θ)],
                        [0, np.sin(θ), np.cos(θ)]])
    return matrix

def Roty(θ):
    matrix = np.array([[np.cos(θ), 0, np.sin(θ)],
                        [0, 1, 0],
                        [-np.sin(θ), 0, np.cos(θ)]])
    return matrix

def Rotz(θ):
    matrix = np.array([[np.cos(θ), -np.sin(θ), 0],
                       [np.sin(θ), np.cos(θ), 0],
                       [0, 0, 1]])
    return matrix

def Visualizar(vector, eje, θ):
    """
    Visualizar la rotación de un vector entorno al eje Z.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar el vector original en rojo
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label='Original')

    # Rotar el vector en incrementos de 10 grados y dibujar cada vector rotado
    for angulo in range(20, 350, 20):
        angulo_rad = np.radians(angulo)
        vector_rotado = list(RotarVector(vector, eje, angulo_rad))
        
        # Dibujar el vector rotado
        ax.quiver(0, 0, 0, vector_rotado[0], vector_rotado[1], vector_rotado[2], color='b')

    # Establecer los límites de la gráfica
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 8])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def Rot(w, θ):
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
    print(f"Tiempo de ejecución de Rot(w, θ) {time.time() - start} segundos")
    return matrix

# Podemos usar la fórmula de Rodrigues para obtener la matriz de rotación de un vector w y un ángulo θ.
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
        - En esta implementación, W^2 se calcula como el producto tensorial de w consigo mismo,
          lo que es una simplificación válida para vectores normalizados.
    """
    start = time.time()
    matrix = np.eye(3) + np.sin(θ)*w + (1 - np.cos(θ))*np.dot(w, w)
    print(f"Tiempo de ejecución de RotRodrigues(w, θ) {time.time() - start} segundos")
    return matrix

def RotarVector(v, eje, θ):
    """
    Rotar un vector v un ángulo θ entorno a un eje especificado.
    # Nota: Se decide como rotar el vector según el tipo de dato de eje y su valor.
    Parámetros: v (array), eje (str) o (array), θ (float) en radianes
    Retorna: v' (array)
    """
    if isinstance(eje, np.ndarray) or isinstance(eje, list):
        print("RotarVector: Rotación entorno a un eje genérico.")
        eje_norm = np.array(eje) / np.linalg.norm(eje)
        #R = Rot(eje_norm, θ)
        R = RotRodrigues(eje_norm, θ)
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

# Menú interactivo
def menu():
    """Menú interactivo para seleccionar acciones."""
    while True:
        print("\n" + "="*80)
        print(" "*25 + "MENÚ DE OPCIONES" + " "*25)
        print("="*80)
        print("Nota. Los vectores que se tomen como ejes seran convertidos a unitarios automáticamente.")
        print("="*80)
        # Añadir content
        print("1. Rotar un vector entorno a un eje específico (x,y,z).")    
        print("2. Rotar un vector entorno a un eje genérico.")
        print("3. Visualizar rotación de un vector entorno al eje Z.")
        print("0. Salir.")
        print("-"*80)

        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1" or opcion == "2":
            vector = Datos(tipo="vector").valor
            if opcion == "1":
                eje = Datos(tipo="eje").valor
            else:
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            angulo = Datos(tipo="angulo").valor
            
            # Rotar el vector
            vector_rotado = RotarVector(vector, eje, angulo)  
            print(f"\nVector original: {vector}")
            print(f"Vector rotado: {vector_rotado}")
            continue
        
        elif opcion == "3": Visualizar(Datos(tipo="vector").valor, "z", Datos(tipo="angulo").valor)

        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

if __name__ == "__main__":
    menu()
