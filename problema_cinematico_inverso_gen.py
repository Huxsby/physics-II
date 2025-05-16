#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import time

from class_robot_structure import Robot, cargar_robot_desde_yaml
from class_helicoidales import calcular_M_generalizado, calcular_T_robot
from class_jacobian import calcular_jacobiana, mostrar_jacobiana_resumida
from class_rotaciones import Rp2Trans, Euler2R, R2Euler

# 8.2. Funciones utilizadas en el código que resuelve el problema cinemático inverso del Niryo One

def VecToso3(w): # convierte un eje de rotación en una matriz antisimétrica 3x3
    return np.array([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

def VecTose3(V): # convierte un vector giro o eje helicoidal en matriz 4x4 se3
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]], np.zeros((1, 4))]

def so3ToVec(so3mat): # extrae un vector de 3 componentes de una matriz antisimétrica so3
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def se3ToVec(se3mat): # Convierte una matriz se3 en un vector giro 1x6
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def MatrixExp6(se3mat): # convierte un vector giro en forma matricial 4x4 se3 en una MTH a través de la exponencial
    se3mat = np.array(se3mat) # vector giro en representación matricial se3 (4x4)
    v=se3mat[0: 3, 3] # extraemos el vector v*theta (velocidad lineal)
    omgmattheta=se3mat[0: 3, 0: 3] # extraemos omega*theta en forma matricial 3x3 (so3)
    omgtheta = so3ToVec(omgmattheta) # lo pasamos a forma vectorial
    if (np.linalg.norm(omgtheta))<1.e-6: # en el caso de que no haya giro (omega despreciable)
        return np.r_[np.c_[np.eye(3), v], [[0, 0, 0, 1]]] # concatena columnas y filas. Sólo traslación
    else: # caso general
        theta = np.linalg.norm(omgtheta)
        omgmat = omgmattheta / theta # omega en forma matricial 3x3 (so3) Normalizada
        # a continuación aplicamos la definición de matriz exponencial que vimos en clase (slide 42, tema 2)
        G_theta=np.eye(3)*theta+(1-np.cos(theta))*omgmat+(theta-np.sin(theta))*np.dot(omgmat,omgmat)
        R=np.eye(3)+np.sin(theta)*omgmat+(1.-np.cos(theta))*np.dot(omgmat,omgmat)
        return np.r_[np.c_[R,np.dot(G_theta,v)/theta],[[0, 0, 0, 1]]]

def MatrixLog3(R): # Calcula la matriz logaritmo de una matriz de rotación
    acosinput = (np.trace(R) - 1) *0.5
    if np.trace(R) >= 3: return np.zeros((3, 3))
    elif np.trace(R) <= -1:
        if abs(1 + R[2][2])>1.e-6: omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif abs(1 + R[1][1])>1.e-6: omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else: omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return (theta*0.5)/np.sin(theta) * (R-np.array(R).T)

def MatrixLog6(T): # Calcula la matriz logaritmo de una MTH
    R=T[0: 3, 0: 3]; p = T[0: 3, 3] # separa la MTH en matriz de rotación y vector traslación
    omgmat = MatrixLog3(R) # coordenadas exponenciales de la matriz de rotación
    # o sea, un vector de rotación como matriz antisimétrica so3 (3x3)
    if np.array_equal(omgmat, np.zeros((3, 3))): # Si no hay rotación, es una matriz de ceros
        return np.r_[np.c_[np.zeros((3, 3)),p],[[0, 0, 0, 0]]]
    else:
        omgvec= so3ToVec(omgmat) # expresa la rotación como un vector en la dirección del eje por el ángulo
        omgmat=omgmat/np.linalg.norm(omgvec) # el vector en el eje de rotación normalizado y en forma matricial
        theta = np.linalg.norm(omgvec) # también se puede calcular como np.arccos((np.trace(R)-1)/2.0)
        # a continuación aplicamos la definición que vimos en clase (ver diapositivas)
        invG_theta=np.eye(3)/theta-omgmat*0.5+(1.0/theta-0.5/np.tan(theta*0.5))*np.dot(omgmat,omgmat)
        v=np.dot(invG_theta,p)
        return np.r_[np.c_[omgmat,v],[[0, 0, 0, 0]]]*theta # primero concatena columnas y luego filas

def Adjunta(T): # Calcula la matriz adjunta de una MTH
    R=T[0: 3, 0: 3]; p = T[0: 3, 3]
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(VecToso3(p), R), R]]

# Esta versión de CinematicaDirecta tiene  un decimal más de potencia 1: [13.507635282249645, -3.4211564466914752, -2.421332532622626, 23.86320013290479*, -7.292415453429542, -17.806945894800737]
                                                                    # 2: [13.507635282249646, -3.4211564466914752, -2.421332532622629, 23.863200132904787, -7.292415453429542, -17.80694589480073*]
# def CinematicaDirecta(robot, M,S,t):
#     T=np.eye(4)
#     for i in range(0,len(robot.links),1): T=np.dot(T,MatrixExp6(VecTose3(S[i]*t[i])))
#     return np.dot(T,M)

def CinematicaDirecta(ejes, thetas, M):
    return calcular_T_robot(ejes, thetas, M)

def CinematicaInversa(robot: Robot, Jacobiana_tuple: tuple, thetas_actuales=None, p_xyz=[0.1, 0.1, 0.1], RPY=[0, 0, 0], error_oet=1.00000000e-10, error_pet=1.00000000e-10, error_vel_lineal=1.00000000e-10):
    """Resolución del problema cinemático inverso para el Robot Niryo One."""
    tiempo = time.time()
    if robot is None:
        raise ValueError("El robot no está definido. Por favor, carga un robot válido.")
    if thetas_actuales is None:
        thetas_actuales = [0]*len(robot.links)

    # Casting inputs a tipos apropiados
    thetas_actuales = np.array([np.float64(theta) for theta in thetas_actuales])
    p_xyz = np.array([np.float64(coord) for coord in p_xyz])
    orientation = Euler2R(RPY[0], RPY[1], RPY[2])
    error_oet=float(error_oet)
    error_vel_lineal=float(error_vel_lineal)

    # Matriz de transformación homogénea en la posición cero del robot
    M = calcular_M_generalizado(robot)

    # Calculamos la Matriz de Transformación Homogénea a partir de posiciones y ángulos
    Tsd = Rp2Trans(orientation, p_xyz)
    print("\nMatriz de transformación homogénea inical Tsd:\n", Tsd)
    print(f"\nVectores oritentation y p_xyz (distancia al objetivo):\n{np.round(orientation, 8)}\n{np.round(p_xyz, 8)}")
   
    print(f"\nExtrayendo dastos del robot:")
    S = robot.ejes_helicoidales; print("Ejes helicoidales del robot:", S)
    print("\nMatriz Jacobiana del robot:")
    J, thetas_s = Jacobiana_tuple; mostrar_jacobiana_resumida(J)
    
    thetas_follower = []                                    # Lista para almacenar los ángulos de las articulaciones por los que ha pasado el robot en cada iteración.
    Tsb = CinematicaDirecta(S, thetas_actuales, M)          # Resuelve la Cinemática Directa para thetas_actuales
    Vb = MatrixLog6(np.dot(np.linalg.inv(Tsb), Tsd))        # vector Giro para ir a la posición deseada en {b}
    Vs = np.dot(Adjunta(Tsb), se3ToVec(Vb))                 # vector Giro en el SR de la base {s}
    
    # Condición de convergencia: módulo de velocidad angular < error_oet y velocidad lineal < error_vel_lineal
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > error_oet or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > error_vel_lineal
    i = 0
    MAXITERATIONS = 20

    # Bucle principal del algoritmo de cinemática inversa iterativo.
    print("\nIteraciones de la cinemática inversa:")
    while err and i < MAXITERATIONS:                        # Continúa mientras el error 'err' sea verdadero (es decir, el error supera los umbrales) y el número de iteraciones 'i' sea menor que 'MAXITERATIONS'.
        thetalist_s = {thetas_s[i]: np.float64(thetas_actuales[i]) for i in range(len(thetas_s))} # Crea un diccionario que asigna cada variable simbólica a su valor actual.
        Jp = J.subs(thetalist_s)                            # Sustituye los valores actuales de los ángulos de las articulaciones (thetalist_s).
        Jp = np.array(Jp.tolist()).astype(np.float64)       # Convierte la Jacobiana numérica (SymPy) a un array NumPy de tipo float64.

        thetas_follower.append(thetas_actuales.tolist())    # Almacena los ángulos por los que ha pasado el robot en cada iteración.

        # Actualiza los ángulos de las articulaciones. Esta es la fórmula central del método de Newton-Raphson para cinemática inversa usando la pseudo-inversa de la Jacobiana.
        # np.linalg.pinv(Jp): Calcula la pseudo-inversa de Moore-Penrose de la Jacobiana numérica. Se usa porque la Jacobiana puede no ser cuadrada o invertible.
        # np.dot(..., Vs): Multiplica la pseudo-inversa por el vector de giro espacial 'Vs' (error de velocidad). El resultado es el cambio necesario en los ángulos de las articulaciones (delta_theta).
        # thetas_actuales + ...: Suma el cambio calculado a los ángulos actuales para obtener los nuevos ángulos.
        thetas_actuales = thetas_actuales + np.dot(np.linalg.pinv(Jp), Vs)

        i = i + 1 # Incrementa el contador de iteraciones.

        # Calcula la cinemática directa con los *nuevos* ángulos de las articulaciones ('thetas_actuales').
        # 'M' es la configuración inicial (home), 'S' son los ejes de giro (screw axes), 'thetas_actuales' son los ángulos actualizados.
        # El resultado 'Tsb' es la nueva pose (posición y orientación) del efector final en el marco espacial.
        Tsb = CinematicaDirecta(S, thetas_actuales, M)

        # Calcula el error de transformación entre la pose actual y la deseada ('Tsd').
        # np.linalg.inv(Tsb): Calcula la inversa de la matriz de transformación homogénea actual.
        # np.dot(..., Tsd): Multiplica la inversa de la pose actual por la pose deseada (T_error = Tsb^-1 * Tsd).
        # MatrixLog6(...): Calcula el logaritmo matricial de la matriz de error. Esto convierte la matriz de transformación de error (SE(3)) en su representación de vector de giro (twist) 'Vb' (se(3)) en el marco del cuerpo (body frame).
        Vb = MatrixLog6(np.dot(np.linalg.inv(Tsb), Tsd))

        # Convierte el vector de giro del marco del cuerpo 'Vb' al marco espacial 'Vs'.
        # Adjunta(Tsb): Calcula la matriz Adjunta de la pose actual 'Tsb'. La Adjunta transforma vectores de giro (twists) entre marcos.
        # se3ToVec(Vb): Convierte la matriz se(3) 'Vb' a un formato de vector 6x1 (si no lo está ya).
        # np.dot(..., ...): Multiplica la Adjunta por el vector de giro del cuerpo 'Vb' para obtener el vector de giro equivalente 'Vs' en el marco espacial (fixed frame).
        # print(Vs): Imprime el vector de giro espacial actual (útil para depuración).
        Vs = np.dot(Adjunta(Tsb), se3ToVec(Vb))

        # Update the error condition 'err' for the next loop iteration.
        # The loop continues if the norm of the angular velocity (Vs[0:3]) exceeds error_oet
        # OR the norm of the linear velocity (Vs[3:6]) exceeds error_vel_lineal.
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > error_oet or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > error_vel_lineal
        # Print the error status with color: red if True (error exists), green if False (converged)
        print(f"\tIter ({i:02d}) Vector giro: \033[36m{np.round(Vs.tolist(), 8)}\033[0m", f"Error: {(f'\033[31m{err}' if err else f'\033[32m{err} ⭢  Solución valida')}\033[0m")
    print(f"\t\033[92mTiempo de cálculo total de la cinemática inversa: {time.time() - tiempo:.4f} segundos\033[0m")
    
    # Imprime el resultado final de la cinemática inversa.
    print(f"\nCoordenadas de las articulaciones:\n {thetas_actuales.tolist()}")
    print("Error en w:", np.round(np.linalg.norm([Vs[0], Vs[1], Vs[2]]),8))
    print("Error en v:", np.round(np.linalg.norm([Vs[3], Vs[4], Vs[5]]),8))
    print("Número de iteraciones:", i)

    Tsd_re = calcular_T_robot(robot.ejes_helicoidales, thetas_actuales, M)
    print("\nMatriz de transformación homogénea final Tsd re-calculada:\n", np.round(Tsd_re, 3))
    print("\nMatriz de transformación homogénea final Tsd original:\n", np.round(Tsd, 3))
    
    print(f"\nLas thetas por las que ha pasado el robot son:")
    for i in range(len(thetas_follower)):
        print(f"\t{np.round(thetas_follower[i], 4).tolist()}")
    
    return thetas_follower

def main():
    """Resolución del problema cinemático inverso para el Robot Niryo One."""
    
    robot = cargar_robot_desde_yaml('robot.yaml')
    Jacobiana_tuple = calcular_jacobiana(robot)
    CinematicaInversa(robot, Jacobiana_tuple)

if __name__=="__main__" :
    main()