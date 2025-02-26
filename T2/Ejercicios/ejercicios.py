""" Matrices de Rotación

[x] - Crea 3 funciones Rotx(θ), Roty(θ) y Rotz(θ) en Python que incluyan las
rotaciones entorno a los ejes cartesianos como funciones. # Funciones de rotación

 - Crea un programa que utilice las funciones anteriores. El input debe ser un vector
v, el eje entorno al cual se debe rotar (x, y o z) y el ángulo de rotación (θ). El
output debe ser las componentes del vector rotado v0.

 - Crea una función Rotv(v; θ) en Python que devuelva la matriz genérica de
rotación de un ángulo θ entorno a cualquier eje v.
Crea un programa en Python que utilice la función anterior. Los input deben ser
el vector que se va a rotar, el eje de rotación y el ángulo a rotar. El output debe
ser el vector rotado un ángulo θ entorno al eje de rotación especificado.

 - Comprueba que los resultados sean los mismos del ejercicio anterior cuando se
realizan rotaciones entorno a los ejes cartesianos. Prueba a hacer rotaciones
entorno a un eje diferente a los ejes cartesianos (recuerda normalizar el vector que
indica el eje de rotación para poder aplicar la fórmula general).

 - Comprueba las propiedades no conmutativa y asociativa de rotaciones sucesivas
utilizando los programas anteriores.
"""

import numpy as np                              # Para cálculos numéricos
from scipy.optimize import fsolve               # Para resolver ecuaciones no lineales
import time  

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

"""
# Menú interactivo
def menu():
    ""Menú interactivo para seleccionar acciones.""
    while True:
        print("\n" + "="*80)
        print(" "*25 + "MENÚ DE OPCIONES" + " "*25)
        print("="*80)
        # Añadir content
        print("-"*80)

        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1": break
        
        elif opcion == "2": break

        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

if __name__ == "__main__":
    menu()
"""