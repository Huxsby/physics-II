#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
import time

EPSILON = 1e-8  # Small threshold for floating point comparisons

from core import Robot, cargar_robot_desde_yaml, print_ejes_helicoidales, str_config, filtrar_configuraciones
from calculations.class_helicoidales import CinematicaDirecta, calcular_posiciones_articulaciones
from calculations.class_jacobian import calcular_jacobiana, mostrar_jacobiana_resumida, calcular_volumen_elipsoides
from calculations.class_rotaciones import Rp2Trans, Euler2R, R2Euler, imprimir_matriz
from animation.visualize_fabrik import visualizar_iteraciones_fabrik

# 8.2. Funciones utilizadas en el código que resuelve el problema cinemático inverso

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
#     for i in range(0,robot.num_links,1): T=np.dot(T,MatrixExp6(VecTose3(S[i]*t[i])))
#     return np.dot(T,M)

def CinematicaInversa(robot: Robot, Jacobiana_tuple: tuple, thetas_actuales=None, p_xyz=[0.1, 0.1, 0.1], RPY=[0, 0, 0], error_oet=1.00000000e-10, error_vel_lineal=1.00000000e-10, show=True):
    """
    Resuelve el problema cinemático inverso para un robot utilizando el método iterativo de Newton-Raphson
    con la pseudo-inversa de la matriz Jacobiana.

    Args:
        robot (Robot): Objeto robot que contiene la definición de sus enlaces y parámetros.
        Jacobiana_tuple (tuple): Tupla que contiene (J, thetas_s), donde J es la matriz Jacobiana simbólica y thetas_s las variables simbólicas correspondientes.
        thetas_actuales (list, optional): Lista de ángulos iniciales para las articulaciones. Por defecto: [0,...,0].
        p_xyz (list, optional): Coordenadas [x, y, z] de la posición deseada del efector final. Por defecto: [0.1, 0.1, 0.1].
        RPY (list, optional): Ángulos de Euler [roll, pitch, yaw] para la orientación deseada. Por defecto: [0, 0, 0].
        error_oet (float, optional): Umbral de error para velocidad angular. Por defecto: 1e-10.
        error_vel_lineal (float, optional): Umbral de error para velocidad lineal. Por defecto: 1e-10.
        show (bool, optional): Mostrar información detallada del proceso. Por defecto: True.

    Returns:
        thetas_follower (list): Lista de listas con los ángulos de las articulaciones en cada iteración del algoritmo.

    Algoritmo:
        1. Calcula la matriz de transformación homogénea objetivo (Tsd) a partir de posición y orientación.
        2. Obtiene la matriz de transformación homogénea inicial (M) y los ejes helicoidales (S).
        3. Inicializa el vector de giro espacial (Vs) que representa el error de velocidad.
        4. En cada iteración:
            - Sustituye los valores actuales en la Jacobiana simbólica para obtener la Jacobiana numérica.
            - Actualiza los ángulos de las articulaciones usando la pseudo-inversa de la Jacobiana.
            - Recalcula la cinemática directa y actualiza el vector de giro (error).
            - Verifica convergencia basada en la norma de las velocidades angular y lineal.
            - Se detiene cuando el error es menor que los umbrales establecidos o alcanza máximo de iteraciones.

    Notas:
        - El método está optimizado para el Robot Niryo One, pero funciona con cualquier robot definido correctamente.
        - La precisión está limitada por los parámetros de error y el número máximo de iteraciones.
        - Utiliza la formulación moderna de la cinemática basada en teoría de Lie con matriz logarítmica y ejes helicoidales.
        - Muestra información detallada del proceso si 'show' es True, incluyendo los valores intermedios y errores.
    """
    tiempo = time.time()
    if robot is None:
        raise ValueError("El robot no está definido. Por favor, carga un robot válido.")
    if thetas_actuales is None:
        thetas_actuales = [0]*robot.num_links

    # Casting inputs a tipos apropiados
    thetas_actuales = np.array([np.float64(theta) for theta in thetas_actuales])
    p_xyz = np.array([np.float64(coord) for coord in p_xyz])
    orientation = Euler2R(RPY[0], RPY[1], RPY[2])
    error_oet=float(error_oet)
    error_vel_lineal=float(error_vel_lineal)

    # Matriz de transformación homogénea en la posición cero del robot
    M = robot.M

    # Calculamos la Matriz de Transformación Homogénea a partir de posiciones y ángulos
    Tsd = Rp2Trans(orientation, p_xyz)
    S = robot.ejes_helicoidales
    J, thetas_s = Jacobiana_tuple
    
    if show:
        # print("\nMatriz de transformación homogénea inical Tsd:\n", Tsd)
        imprimir_matriz(Tsd, "Matriz de transformación homogénea objetivo Tsd")
        print(f"\nVectores oritentation y p_xyz (distancia al objetivo):\n{str_config(orientation, 8)}\n{np.round(p_xyz, 8)}")
        print(f"\nExtrayendo dastos del robot:")
        print_ejes_helicoidales(robot)
        print("\nMatriz Jacobiana del robot:")
        mostrar_jacobiana_resumida(J)
        print("\nIteraciones de la cinemática inversa:")
        cero_umbral = min(error_oet, error_vel_lineal) # Precalcula el umbral de cero para la impresión del vector de giro.
    
    thetas_follower = []                                    # Lista para almacenar los ángulos de las articulaciones por los que ha pasado el robot en cada iteración.
    Tsb = CinematicaDirecta(S, thetas_actuales, M)          # Resuelve la Cinemática Directa para thetas_actuales
    Vb = MatrixLog6(np.dot(np.linalg.inv(Tsb), Tsd))        # vector Giro para ir a la posición deseada en {b}
    Vs = np.dot(Adjunta(Tsb), se3ToVec(Vb))                 # vector Giro en el SR de la base {s}
    
    # Condiciones del bucle: err = True (error) y i < MAXITERATIONS (máximo de iteraciones)
    i = 0; MAXITERATIONS = 20
    # Condición de convergencia: módulo de velocidad angular < error_oet y velocidad lineal < error_vel_lineal
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > error_oet or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > error_vel_lineal

    while err and i < MAXITERATIONS:                        # Continúa mientras el error 'err' sea verdadero (es decir, el error supera los umbrales) y el número de iteraciones 'i' sea menor que 'MAXITERATIONS'.
        thetalist_s = {thetas_s[i]: np.float64(thetas_actuales[i]) for i in range(len(thetas_s))} # Crea un diccionario que asigna cada variable simbólica a su valor actual.
        Jp = J.subs(thetalist_s)                            # Sustituye los valores actuales de los ángulos de las articulaciones (thetalist_s).
        Jp = np.array(Jp, dtype=np.float64)                 # Convierte la Jacobiana numérica (SymPy) a un array NumPy de tipo float64

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
        # Format each value in Vs based on its magnitude, with scientific notation for small values
        
        if show:
            ancho = 12  # Ancho fijo para todos los elementos, funciona como complemento de la notación científica
            formatted_values = []
            for element in Vs:
                # Usar el error de convergencia (cero_umbral) como umbral para imprimir ceros
                if abs(element) < cero_umbral:  # Cero según el error de convergencia
                    formatted_values.append(f"\033[90m{'0.00000000':>{ancho}}\033[36m") # Color gris para ceros
                elif abs(element) < 1e-4:  # Pequeño pero no cero - notación científica alineada
                    formatted_values.append(f"{element:>{ancho}.4e}")
                else:  # Números grandes - notación fija alineada
                    formatted_values.append(f"{element:>{ancho}.8f}")
            vector_str = "[" + ", ".join(formatted_values) + "]"
            print(f"\tIter ({i:02d}) Vector giro: \033[36m{vector_str}\033[0m", f"Error: {(f'\033[31m{err}' if err else f'\033[32m{err} ⭢  Solución valida')}\033[0m")
    
    print(f"\t\033[92mTiempo de cálculo total de la cinemática inversa: {time.time() - tiempo:.4f} segundos\033[0m")
    
    if show: # Imprime el resultado final de la cinemática inversa.
        Tsd_re = CinematicaDirecta(robot.ejes_helicoidales, thetas_actuales, M)
        R = Tsd_re[:3,:3]; p = Tsd_re[:3,3]; RPY = R2Euler(R)
        print(f"\nCoordenadas de las articulaciones:\n {thetas_actuales.tolist()}")
        print(f"\nCoordenadas (x,y,z) del TCP:  {p} (Objetivo: {Tsd[:3,3]})")
        print(f"Los angulos de Euler (Roll Pitch Yaw) son: {RPY} (Objetivo: {R2Euler(Tsd[:3,:3])})") 
        print("\nError en w:", np.round(np.linalg.norm([Vs[0], Vs[1], Vs[2]]), 8))
        print("Error en v:", np.round(np.linalg.norm([Vs[3], Vs[4], Vs[5]]), 8))
        print("Número de iteraciones:", i)
        
        # Recalcular la matriz de transformación homogénea final Tsd_re
        # print("\nMatriz de transformación homogénea final Tsd re-calculada:\n", np.round(Tsd_re, 3))
        imprimir_matriz(Tsd_re, "Matriz de transformación homogénea final Tsd re-calculada")
        # print("\nMatriz de transformación homogénea final Tsd original:\n", np.round(Tsd, 3))
        imprimir_matriz(Tsd, "Matriz de transformación homogénea final Tsd original")
        
        print(f"\nLas thetas por las que ha pasado el robot son:")
        for i in range(len(thetas_follower)):
            J_vol = J.subs(thetalist_s)
            vol_EM, vol_EF = calcular_volumen_elipsoides(J_vol)  # Guardamos los elipsoides para ver si se cruza cerca de una singularidad.
            print(f"\t{str_config(thetas_follower[i], 4)}\t Volumen elipsoide: {vol_EM}")
            if vol_EM < 1e-20: print("\t\t\033[91mCuidado, el elipsoide es muy pequeño, puede haber una singularidad\033[0m"); input("Presione Enter para continuar...")
    
    return thetas_follower

def CinematicaInversa_FABRIK(robot: Robot, p_xyz_objetivo: list, thetas_iniciales: list = None, tol: float = 1e-3, max_iter: int = 100, show: bool = True):
    """
    Resuelve la cinemática inversa utilizando el algoritmo FABRIK (Forward And Backward Reaching Inverse Kinematics).

    Args:
        robot (Robot): El objeto robot con su estructura.
        p_xyz_objetivo (list): La posición objetivo [x, y, z] para el efector final.
        thetas_iniciales (list, optional): Configuración inicial de los ángulos. Si es None, se usa [0,...,0].
        tol (float, optional): Tolerancia para la convergencia.
        max_iter (int, optional): Número máximo de iteraciones.
        show (bool, optional): Si es True, muestra información de depuración.

    Returns:
        list: La lista de ángulos de las articulaciones (thetas) que alcanzan el objetivo, o None si no converge.
        list: Historial de posiciones de las articulaciones en cada iteración.
    """
    p_xyz_objetivo = np.array(p_xyz_objetivo, dtype=float)
    num_eslabones = robot.num_links
    if thetas_iniciales is None:
        thetas_iniciales = np.zeros(num_eslabones)
    else:
        thetas_iniciales = np.array(thetas_iniciales, dtype=float)

    # 1. Calcular posiciones iniciales de las articulaciones y longitudes de los eslabones
    p = calcular_posiciones_articulaciones(robot, thetas_iniciales)
    p = [np.array(pi, dtype=float) for pi in p]
    longitudes = [np.linalg.norm(p[i+1] - p[i]) for i in range(len(p) - 1)]
    dist_total = sum(longitudes)

    if show:
        print("--- Cinematica Inversa FABRIK ---")
        print(f"Objetivo: {p_xyz_objetivo}")
        print(f"Posiciones iniciales de las articulaciones (p):")
        for i, pos in enumerate(p):
            print(f"  p[{i}]: {pos}")
        print(f"Longitudes de eslabones (d): {longitudes}")
        print(f"Longitud total del brazo: {dist_total:.4f}")

    # --- Algoritmo FABRIK ---
    p_history = [ [pi.copy() for pi in p] ]  # Historial de posiciones

    # 1.1-1.3: Distancia raíz-objetivo y verificación de alcance
    dist = np.linalg.norm(p[0] - p_xyz_objetivo)
    if show: print(f"Distancia de la raíz al objetivo: {dist:.4f}")

    # 1.4: Comprobar si el objetivo está fuera de alcance
    if dist > dist_total:
        if show: print("\033[93mAdvertencia: El objetivo está fuera del alcance del robot.\033[0m")
        # Estirar el brazo hacia el objetivo
        for i in range(num_eslabones):
            r = np.linalg.norm(p_xyz_objetivo - p[i])
            if r < EPSILON:
                p[i+1] = p[i]
            else:
                lambda_i = longitudes[i] / r

    if dist > dist_total:
        if show: print("\033[93mAdvertencia: El objetivo está fuera del alcance del robot.\033[0m")
        # Estirar el brazo hacia el objetivo
        for i in range(num_eslabones):
            r = np.linalg.norm(p_xyz_objetivo - p[i])
            if r < EPSILON:
                p[i+1] = p[i]
            else:
                lambda_i = longitudes[i] / r
                p[i+1] = (1 - lambda_i) * p[i] + lambda_i * p_xyz_objetivo
        p_history.append([pi.copy() for pi in p])
    else:
        b = p[0].copy()  # Guardar la posición inicial de la raíz
        difA = np.linalg.norm(p[-1] - p_xyz_objetivo)
        iteracion = 0
        while difA > tol and iteracion < max_iter:
            if show:
                print(f"\n--- Iteración {iteracion + 1} ---")
                print(f"Distancia al objetivo (difA): {difA:.6f}")

            # FORWARD: mover efector final al objetivo y propagar hacia la base
            p[-1] = p_xyz_objetivo.copy()
            for i in range(num_eslabones - 1, 0, -1):
                r = np.linalg.norm(p[i] - p[i-1])
                if r < EPSILON:
                    p[i-1] = p[i]
                else:
                    lambda_i = longitudes[i-1] / r
                    p[i-1] = (1 - lambda_i) * p[i] + lambda_i * p[i-1]
            # BACKWARD: fijar base y propagar hacia el efector
            p[0] = b.copy()
            for i in range(num_eslabones):
                r = np.linalg.norm(p[i+1] - p[i])
                if r < EPSILON:
                    p[i+1] = p[i]
                else:
                    lambda_i = longitudes[i] / r
                    p[i+1] = (1 - lambda_i) * p[i] + lambda_i * p[i+1]
            p_history.append([pi.copy() for pi in p])
            difA = np.linalg.norm(p[-1] - p_xyz_objetivo)
            iteracion += 1

        if show:
            print("\n--- Convergencia ---")
            print(f"Iteraciones: {iteracion}")
            print(f"Distancia final al objetivo: {difA:.6f}")
            if difA > tol:
                print("\033[93mAdvertencia: No se alcanzó la tolerancia después del número máximo de iteraciones.\033[0m")

    # --- Conversión de posiciones a ángulos (thetas) ---
    if show:
        print("\nPosiciones finales de las articulaciones (p):")
        for i, pos in enumerate(p):
            print(f"  p[{i}]: {pos}")

    # Extraer los ángulos (thetas) de las posiciones finales de las articulaciones 'p'
    thetas_finales = extraer_thetas_desde_posiciones(robot, p)

    if show:
        print(f"Ángulos finales (thetas): {np.round(np.rad2deg(thetas_finales), 2)}")

    return thetas_finales, p_history

def extraer_thetas_desde_posiciones(robot: Robot, p_final: list) -> list:
    """
    Extrae los ángulos de las articulaciones (thetas) a partir de la lista de posiciones
    de las articulaciones obtenida por FABRIK.

    Args:
        robot (Robot): El objeto robot.
        p_final (list): La lista de posiciones 3D de las articulaciones [p0, p1, ..., pn].

    Returns:
        list: La lista de ángulos de las articulaciones (thetas).
    """
    thetas = []
    T_acumulada = np.eye(4)
    p_final = [np.array(punto) for punto in p_final]

    for i in range(robot.num_links):
        S_i = robot.ejes_helicoidales[i]
        w_i = S_i[:3]
        v_i = S_i[3:]
        
        # Transformación inversa para llevar p[i+1] al sistema de coordenadas de la articulación i
        T_inv = np.linalg.inv(T_acumulada)
        p_i_local = T_inv @ np.hstack([p_final[i], 1])
        p_i1_local = T_inv @ np.hstack([p_final[i+1], 1])
        
        vector_actual = (p_i1_local - p_i_local)[:3]

        if robot.links[i].tipo == 'revolute':
            # Para articulaciones de revolución, necesitamos encontrar el ángulo theta
            # que alinea el vector de referencia con el vector actual en el plano de rotación.
            
            # Vector de referencia en el plano de rotación (perpendicular a w_i)
            # Tomamos la proyección de v_i sobre el plano normal a w_i
            q_i = robot.links[i].joint_coords
            v_ref_dir = -np.cross(w_i, q_i) # Dirección inicial del eslabón
            
            # Proyectar los vectores sobre el plano normal al eje de rotación w_i
            vector_actual_proy = vector_actual - np.dot(vector_actual, w_i) * w_i
            v_ref_proy = v_ref_dir - np.dot(v_ref_dir, w_i) * w_i

            # Normalizar los vectores proyectados
            if np.linalg.norm(vector_actual_proy) > 1e-6 and np.linalg.norm(v_ref_proy) > 1e-6:
                u_actual = vector_actual_proy / np.linalg.norm(vector_actual_proy)
                u_ref = v_ref_proy / np.linalg.norm(v_ref_proy)
                
                # Calcular el ángulo entre los dos vectores
                cos_theta = np.dot(u_ref, u_actual)
                sin_theta_sign = np.sign(np.dot(w_i, np.cross(u_ref, u_actual)))
                
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * sin_theta_sign
            else:
                theta = 0.0

        elif robot.links[i].tipo == 'prismatic':
            # Para articulaciones prismáticas, theta es la distancia a lo largo del eje v_i
            # La dirección de traslación es v_i
            theta = np.dot(vector_actual, v_i)
        
        else:
            theta = 0.0 # Tipo de articulación no soportado

        thetas.append(theta)
        
        # Actualizar la matriz de transformación acumulada para la siguiente iteración
        from calculations.class_helicoidales import calcular_exp_Sθ
        T_i = calcular_exp_Sθ(S_i, theta)
        T_acumulada = T_acumulada @ T_i

    return thetas

def menu_cinematica_inversa():
    """Resolución del problema cinemático inverso generalizado."""

    robot = cargar_robot_desde_yaml('config/robot.yaml')

    # Prueba de Cinemática Inversa con el método de FABRIK (Forward And Backward Reaching Inverse Kinematics)
    # NOTA: Se usa una configuración inicial ligeramente flexionada para evitar la singularidad en thetas = [0,0,0...]
    thetas_iniciales_fabrik = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    p_xyz_objetivo = [0.1, 0.1, 0.1]
    thetas_finales, p_history = CinematicaInversa_FABRIK(robot, p_xyz_objetivo=p_xyz_objetivo, thetas_iniciales=thetas_iniciales_fabrik, tol=1e-3, max_iter=15, show=True)
    
    if thetas_finales is not None:
        # La función de filtrado espera una lista de configuraciones
        filtrar_configuraciones(robot, [thetas_finales])
        
        # Llamar a la función de visualización con el historial de posiciones
        visualizar_iteraciones_fabrik(robot, p_history, p_xyz_objetivo, save_gif=True)

    # Jacobiana_tuple = calcular_jacobiana(robot)
    # thetas_follower = CinematicaInversa(robot, Jacobiana_tuple, p_xyz=[0.1, 0.1, 0.1], RPY=[0, 0, 0])
    # filtrar_configuraciones(robot, thetas_follower)

    # print("\nConfiguraciones equivalentes entre (-π, π):")
    # # thetas_follower = [[np.mod(theta, 2 * np.pi) for theta in thetas] for thetas in thetas_follower] # PASAR VALORES A VALORES ENTRE -2PI Y 2PI
    # thetas_follower = [[(theta + np.pi) % (2 * np.pi) - np.pi for theta in thetas] for thetas in thetas_follower] # PASAR VALORES A VALORES ENTRE -π Y π
    # filtrar_configuraciones(robot, thetas_follower) 
if __name__ == "__main__":
    menu_cinematica_inversa()