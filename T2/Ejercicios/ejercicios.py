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

5 - Comprueba que los resultados sean los mismos del ejercicio anterior cuando se
realizan rotaciones entorno a los ejes cartesianos. Prueba a hacer rotaciones
entorno a un eje diferente a los ejes cartesianos (recuerda normalizar el vector que
indica el eje de rotación para poder aplicar la fórmula general).

6 - Comprueba las propiedades no conmutativa y asociativa de rotaciones sucesivas
utilizando los programas anteriores.
"""

import numpy as np                              # Para cálculos numéricos
from scipy.optimize import fsolve               # Para resolver ecuaciones no lineales
import time                                     # Para medir el tiempo de ejecución

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

def Rot(w, θ):
    """
    Matriz de rotación de un ángulo θ entorno a un eje genérico w.
    Parámetros: w (array), θ (float) en radianes
    Retorna: matrix (Matriz de rotación de un eje génerico w)
    """

    w1, w2, w3 = w      # Componentes del vector de rotación   # Convertir a radianes
    s = np.sin(θ)       # Seno de θ
    c = np.cos(θ)       # Coseno de θ
    t = 1 - c           # 1 - cos(θ)
    
    matrix = np.array([[c + w1**2*t, w1*w2*t - w3*s, w1*w3*t + w2*s],
                       [w1*w2*t + w3*s, c + w2**2*t, w2*w3*t - w1*s],
                       [w1*w3*t - w2*s, w2*w3*t + w1*s, c + w3**2*t]])
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
        R = Rot(eje_norm, θ)
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
        # Añadir content
        print("1. Rotar un vector entorno a un eje específico.")
        print("2. Rotar un vector entorno a un eje genérico.")
        print("0. Salir.")
        print("-"*80)

        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            vector = Datos(tipo="vector").valor
            eje = Datos(tipo="eje").valor
            angulo = Datos(tipo="angulo").valor
            
            # Rotar el vector
            vector_rotado = RotarVector(vector, eje, angulo)  
            print(f"\nVector original: {vector}")
            print(f"Vector rotado: {vector_rotado}")
            break
        
        elif opcion == "2": 
            vector = Datos(tipo="vector").valor
            eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas): ").valor
            angulo = Datos(tipo="angulo").valor
            
            # Rotar el vector
            vector_rotado = RotarVector(vector, eje, angulo)
            print(f"\nVector original: {vector}")
            print(f"Vector rotado: {vector_rotado}")
            break

        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

if __name__ == "__main__":
    menu()
