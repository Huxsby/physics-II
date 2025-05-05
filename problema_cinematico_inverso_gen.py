#!/usr/bin/env python
#
# By Angel.Pineiro at usc.es
# Version April, 2021
#
# EJEMPLO DE USO:
# python CinematicaInversaNiryo_03.py -a '0 0 90' -r '0.2 0 0.1' -j '0 -0.8 -0.8 0 -1.5 0'
#
#

import numpy as np
import sympy as sp

from class_robot_structure import Robot, cargar_robot_desde_yaml
from class_helicoidales import calcular_M_generalizado, calcular_T_robot
from class_jacobian import calcular_jacobiana

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

#____________________________________________________________________________________________________________

def getR(w,RPY): # Formula de Rodrigues para obtener matriz de Rotación
    wmat=VecToso3(w)
    return np.eye(3)+np.sin(RPY)*wmat+(1.-np.cos(RPY))*np.dot(wmat,wmat)

def getqs(): # Definimos vectores q para la posición cero del Robot Niryo One
    # vectores que van de cada eje al siguiente
    q=[]; scalefactor=0.001
    L=np.array([103.0, 80.0, 210.0, 30.0, 41.5, 180.0, 23.7, -5.5])*scalefactor
    q.append(np.array([0,0,L[0]]))
    q.append(np.array([0,0,L[1]]))
    q.append(np.array([0,0,L[2]]))
    q.append(np.array([L[4],0,L[3]]))
    q.append(np.array([L[5],0,0]))
    q.append(np.array([L[6],0,L[7]]))
    return q

def getT(orientation,r): # Devuelve la matriz de transformación homogénea
    i=np.array([1,0,0]); j=np.array([0,1,0]); k=np.array([0,0,1]);
    Ri=getR(i,orientation[0]); Rj=getR(j,orientation[1]); Rk=getR(k,orientation[2]);
    R=np.matmul(Rk,np.matmul(Rj,Ri))
    aux=np.array([[0,0,0,1]])
    return np.r_[np.c_[R,r],aux]

#____________________________________________________________________________________________________________

def CinematicaDirecta(M,S,t):
    T=np.eye(4)
    for i in range(0,6,1): T=np.dot(T,MatrixExp6(VecTose3(S[i]*t[i])))
    return np.dot(T,M)


def main():
    """Resolución del problema cinemático inverso para el Robot Niryo One."""
    
    robot = cargar_robot_desde_yaml('robot.yaml')
    
    # Por defecto:
    thethas_actuales = '0 0 0 0 0 0'    # Coordenadas iniciales de las articulaciones en rad
    p_xyz = '0.1 0.1 0.1'               # Coordenadas p_xyz finales para el elemento terminal
    RPY = '0 0 0'                       # ángulos de Euler finales para el elemento terminal en grados
    error_oet = 0.01                    # Error en la orientación del elemento terminal (0.01 por defecto)
    error_pet = 0.001                   # Error en la posición del elemento terminal (0.001 por defecto)
    error_vel_lineal = 0.001            # Error en la velocidad lineal del elemento terminal (0.001 por defecto)

    """
    *Manejar entradas de usuario para los parámetros del problema cinemático inverso.
    """

    # Convertir thethas_actuales y p_xyz a una lista de np.float64 directamente
    thethas_actuales = np.array([np.float64(theta) for theta in str(thethas_actuales).split()])
    p_xyz = [np.float64(coord) for coord in str(p_xyz).split()]

    # Convertir RPY a una lista de radianes directamente
    orientation = [np.deg2rad(np.float64(angle)) for angle in str(RPY).split()]
    
    error_oet=float(error_oet)
    error_vel_lineal=float(error_vel_lineal)

    # Matriz de transformación homogénea en la posición cero del robot
    M = calcular_M_generalizado(robot)

    # Calculamos la Matriz de Transformación Homogénea a partir de posiciones y ángulos
    T = getT(orientation, p_xyz)
    print("\nMatriz de transformación homogénea original T:\n", T)

    print("\nValores de las articulaciones iniciales:", [0,0,0,0,0,0])
    Tm = calcular_T_robot(robot.ejes_helicoidales, [0,0,0,0,0,0], M)
    print("\nMatriz de transformación homogénea mia T:", Tm)
    print("\nOrientacion y p_xyz comparation", '\n', orientation, '\n', p_xyz)

    S = robot.ejes_helicoidales

    J, thetas_s = calcular_jacobiana(robot)
    i = 0
    maxiterations = 20
    Tsb = CinematicaDirecta(M,S, thethas_actuales) # Resuelve la Cinemática Directa para thethas_actuales
    Vb = MatrixLog6(np.dot(np.linalg.inv(Tsb), T)) # vector Giro para ir a la posición deseada en {b}
    Vs = np.dot(Adjunta(Tsb), se3ToVec(Vb)) # vector Giro en el SR de la base {s}
    
    # condición de convergencia: módulo de velocidad angular < error_oet y velocidad lineal < error_vel_lineal
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > error_oet or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > error_vel_lineal
    
    # Bucle principal del algoritmo de cinemática inversa iterativo.
    while err and i < maxiterations: # Continúa mientras el error 'err' sea verdadero (es decir, el error supera los umbrales) y el número de iteraciones 'i' sea menor que 'maxiterations'.
        thetalist_s = {thetas_s[i]: np.float64(thethas_actuales[i]) for i in range(len(thetas_s))}
        Jp = J.subs(thetalist_s)     # Sustituye los valores actuales de los ángulos de las articulaciones (thetalist_s).

        Jp = np.array(Jp.tolist()).astype(np.float64)   # Convierte la Jacobiana numérica (SymPy) a un array NumPy de tipo float64.

        # Actualiza los ángulos de las articulaciones. Esta es la fórmula central del método de Newton-Raphson para cinemática inversa usando la pseudo-inversa de la Jacobiana.
        # np.linalg.pinv(Jp): Calcula la pseudo-inversa de Moore-Penrose de la Jacobiana numérica. Se usa porque la Jacobiana puede no ser cuadrada o invertible.
        # np.dot(..., Vs): Multiplica la pseudo-inversa por el vector de giro espacial 'Vs' (error de velocidad). El resultado es el cambio necesario en los ángulos de las articulaciones (delta_theta).
        # thethas_actuales + ...: Suma el cambio calculado a los ángulos actuales para obtener los nuevos ángulos.
        thethas_actuales = thethas_actuales + np.dot(np.linalg.pinv(Jp), Vs)

        i = i + 1 # Incrementa el contador de iteraciones.

        # Calcula la cinemática directa con los *nuevos* ángulos de las articulaciones ('thethas_actuales').
        # 'M' es la configuración inicial (home), 'S' son los ejes de giro (screw axes), 'thethas_actuales' son los ángulos actualizados.
        # El resultado 'Tsb' es la nueva pose (posición y orientación) del efector final en el marco espacial.
        Tsb = CinematicaDirecta(M, S, thethas_actuales)

        # Calcula el error de transformación entre la pose actual y la deseada ('T').
        # np.linalg.inv(Tsb): Calcula la inversa de la matriz de transformación homogénea actual.
        # np.dot(..., T): Multiplica la inversa de la pose actual por la pose deseada (T_error = Tsb^-1 * T).
        # MatrixLog6(...): Calcula el logaritmo matricial de la matriz de error. Esto convierte la matriz de transformación de error (SE(3)) en su representación de vector de giro (twist) 'Vb' (se(3)) en el marco del cuerpo (body frame).
        Vb = MatrixLog6(np.dot(np.linalg.inv(Tsb), T))

        # Convierte el vector de giro del marco del cuerpo 'Vb' al marco espacial 'Vs'.
        # Adjunta(Tsb): Calcula la matriz Adjunta de la pose actual 'Tsb'. La Adjunta transforma vectores de giro (twists) entre marcos.
        # se3ToVec(Vb): Convierte la matriz se(3) 'Vb' a un formato de vector 6x1 (si no lo está ya).
        # np.dot(..., ...): Multiplica la Adjunta por el vector de giro del cuerpo 'Vb' para obtener el vector de giro equivalente 'Vs' en el marco espacial (fixed frame).
        # print(Vs): Imprime el vector de giro espacial actual (útil para depuración).
        Vs = np.dot(Adjunta(Tsb), se3ToVec(Vb)); print (Vs)

        # Update the error condition 'err' for the next loop iteration.
        # The loop continues if the norm of the angular velocity (Vs[0:3]) exceeds error_oet
        # OR the norm of the linear velocity (Vs[3:6]) exceeds error_vel_lineal.
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > error_oet or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > error_vel_lineal
    
    print ("\n\nCoordenadas de las articulaciones:\n", thethas_actuales)
    print ("Error en w:", np.round(np.linalg.norm([Vs[0], Vs[1], Vs[2]]),8))
    print ("Error en v:", np.round(np.linalg.norm([Vs[3], Vs[4], Vs[5]]),8))
    print ("Número de iteraciones:", i)

if __name__=="__main__" :
    main()