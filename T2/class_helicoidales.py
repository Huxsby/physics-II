import numpy as np
import os
from class_datos import Datos
from class_rotaciones import *
import class_robot_structure as robot_structure

## Función validada
def calcular_Sθ(S, theta):
    """
    Crea la matriz [S]θ a partir del eje helicoidal S y el ángulo/distancia θ.
    
    Parámetros:
    - S (numpy.ndarray): Vector 6D del eje helicoidal, S = [ω, v]
        donde ω es el vector de rotación (3D) y v es el vector de traslación (3D)
    - theta (float): Ángulo de rotación (si ||ω|| = 1) o distancia (si ||ω|| = 0)
    
    Retorna:
    - numpy.ndarray: Matriz [S]θ ∈ se(3) (4x4)
    """
    # Extraer componentes del eje helicoidal
    omega = S[:3]  # Vector de rotación (3D)
    v = S[3:]      # Vector de velocidad lineal (3D)
    
    # Calcular la norma de omega
    omega_norm = np.linalg.norm(omega)
    
    # Crear la matriz [S]θ
    S_theta = np.zeros((4, 4))
    
    if omega_norm > 1e-6:  # Si hay rotación (paso finito)
        # Normalizar omega si no es unitario
        if abs(omega_norm - 1.0) > 1e-6:
            omega = omega / omega_norm
        
        # Parte superior izquierda: [ω]θ (matriz antisimétrica por theta)
        S_theta[:3, :3] = antisimetrica(omega) * theta
        
        # Parte superior derecha: vθ
        S_theta[:3, 3] = v * theta
    else:  # Caso de traslación pura (paso infinito)
        # Normalizar v si no es unitario
        v_norm = np.linalg.norm(v)
        if abs(v_norm - 1.0) > 1e-6 and v_norm > 1e-6:
            v = v / v_norm
            
        # Parte superior derecha: vθ (solo traslación)
        S_theta[:3, 3] = v * theta
    
    return S_theta

## Función validada
def calcular_expSθ(S, theta):
    """
    Calcula la exponencial de la matriz e^([S]θ) para obtener la transformación homogénea. Nota T = e^([S]θ)
    
    Parámetros:
    - S (numpy.ndarray): Vector 6D del eje helicoidal, S = [ω, v]
    - theta (float): Ángulo de rotación o distancia según sea el caso
    
    Retorna:
    - numpy.ndarray: Matriz de transformación homogénea e^([S]θ) ∈ SE(3) (4x4)
    """
    # Extraer componentes
    omega = S[:3]
    v = S[3:]
    
    # Calcular la norma del vector de rotación
    omega_norm = np.linalg.norm(omega)
    
    # Matriz de transformación homogénea (4x4)
    T = np.eye(4)
    
    if omega_norm > 1e-6:  # Caso con rotación
        # Normalizar omega si no es unitario
        if abs(omega_norm - 1.0) > 1e-6:
            omega = omega / omega_norm
            theta = theta * omega_norm  # Ajustar theta si se normaliza omega
        
        # Calcular la matriz de rotación usando Rodrigues
        omega_hat = antisimetrica(omega)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # R = I + sin(θ)·[ω] + (1-cos(θ))·[ω]²
        R = np.eye(3) + sin_theta * omega_hat + (1 - cos_theta) * np.dot(omega_hat, omega_hat)
        
        # Calcular la parte de traslación
        # p = (Iθ + (1-cos(θ))[ω] + (θ-sin(θ))[ω]²)v
        temp1 = np.eye(3) * theta
        temp2 = (1 - cos_theta) * omega_hat
        temp3 = (theta - sin_theta) * np.dot(omega_hat, omega_hat)
        p = np.dot(temp1 + temp2 + temp3, v)
        
        # Formar la matriz de transformación homogénea
        T[:3, :3] = R
        T[:3, 3] = p
    
    else:  # Caso de traslación pura
        # Normalizar v si no es unitario
        v_norm = np.linalg.norm(v)
        if abs(v_norm - 1.0) > 1e-6 and v_norm > 1e-6:
            v = v / v_norm
            
        # T = [I, vθ; 0, 1]
        T[:3, 3] = v * theta
    
    return T

## Función validada
def logaritmo_transformacion(T):
    """
    Calcula el logaritmo de una matriz de transformación homogénea T.
    Encuentra el eje helicoidal S y la distancia θ tales que e^([S]θ) = T.
    
    Parámetros:
    - T (numpy.ndarray): Matriz de transformación homogénea 4x4
    
    Retorna:
    - theta (float): Ángulo/distancia a lo largo del eje helicoidal
    - S (numpy.ndarray): Vector 6D representando el eje helicoidal [ω, v]
    """
    # Extraer la matriz de rotación y el vector de traslación
    R = T[:3, :3]
    p = T[:3, 3]
    
    # Calcular el logaritmo de la matriz de rotación
    theta, omega = LogRot(R)  # Usando la función LogRot del módulo class_rotaciones
    
    # Caso especial: rotación nula o muy pequeña
    if theta < 1e-6:
        # Traslación pura
        v_norm = np.linalg.norm(p)
        if v_norm < 1e-6:
            # Identidad (sin rotación ni traslación)
            return 0.0, np.zeros(6)
        else:
            # Traslación pura normalizada
            theta = v_norm
            v = p / v_norm
            S = np.concatenate([np.zeros(3), v])
            return theta, S
    
    # Calcular el vector v para el eje helicoidal
    omega_hat = antisimetrica(omega)
    
    # Para calcular v, usamos la fórmula inversa de la parte traslacional
    # Invertimos: p = (Iθ + (1-cos(θ))[ω] + (θ-sin(θ))[ω]²)v
    
    # Calculamos la matriz A = (Iθ + (1-cos(θ))[ω] + (θ-sin(θ))[ω]²)
    A = np.eye(3) * theta
    A += (1 - np.cos(theta)) * omega_hat
    A += (theta - np.sin(theta)) * np.dot(omega_hat, omega_hat)
    
    # Ahora necesitamos resolver A·v = p para encontrar v
    # Podemos usar la pseudo-inversa para sistemas que podrían no tener solución exacta
    v = np.linalg.lstsq(A, p, rcond=None)[0]
    
    # Formar el vector S = [ω, v]
    S = np.concatenate([omega, v])
    
    return theta, S

def visualizar_eje_helicoidal(S, theta, num_puntos=100):
    """
    Genera puntos para visualizar un eje helicoidal.
    
    Parámetros:
    - S (numpy.ndarray): Vector 6D del eje helicoidal [ω, v]
    - theta (float): Ángulo/distancia a lo largo del eje
    - num_puntos (int): Número de puntos para la visualización
    
    Retorna:
    - numpy.ndarray: Array de puntos 3D a lo largo del eje helicoidal
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extraer componentes
    omega = S[:3]
    v = S[3:]
    
    # Calcular normas
    omega_norm = np.linalg.norm(omega)
    
    # Inicializar figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear puntos a lo largo del eje helicoidal
    t_values = np.linspace(0, theta, num_puntos)
    points = np.zeros((num_puntos, 3))
    
    if omega_norm > 1e-6:  # Caso con rotación (paso helicoidal)
        # Normalizar omega
        if abs(omega_norm - 1.0) > 1e-6:
            omega = omega / omega_norm
        
        for i, t in enumerate(t_values):
            # Calcular matriz de transformación para ese punto
            T_t = calcular_expSθ(S, t)
            # El origen transformado es el punto en el eje helicoidal
            points[i] = T_t[:3, 3]
        
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, label='Eje Helicoidal')
        
        # Visualizar el vector de rotación
        origin = points[0]
        ax.quiver(origin[0], origin[1], origin[2], 
                 omega[0], omega[1], omega[2], 
                 length=2.0, color='r', label='Vector ω')
        
    else:  # Caso de traslación pura
        # Normalizar v
        v_norm = np.linalg.norm(v)
        if abs(v_norm - 1.0) > 1e-6 and v_norm > 1e-6:
            v = v / v_norm
        
        # Una línea recta en dirección v
        for i, t in enumerate(t_values):
            points[i] = t * v
        
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'g-', linewidth=2, label='Eje de Traslación')
    
    # Configurar gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualización del Eje Helicoidal')
    ax.legend()
    
    # Mostrar el sistema de coordenadas original
    ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, color='b', label='Z')
    
    plt.show()
    
    return points

def calcular_M_generalizado(robot):
    """
    Calcula la matriz M 4x4, matriz de transformación correspondiente a la posición y orientación del
    elemento terminal cuando las coordenadas de todas las articulaciones son nulas (posición “cero” del Robot)
    
    Parámetros:
    - robot: Objeto robot con información de sus eslabones y articulaciones
    
    Retorna:
    - M: Matriz de transformación homogénea 4x4 desde la base hasta el efector final
         cuando todas las articulaciones están en posición cero
    """
    M = np.eye(4)  # Matriz identidad inicial
    
    for link in robot.links:
        M[:3, 3] += np.dot(link.length,abs(link.joint_axis))
        print(M[:3, 3])

        #Pofesor: # M = np.array([[1,0,0,L[6]+L[5]+L[4]],[0,1,0,0],[0,0,1,L[0]+L[1]+L[2]+L[3]+L[7]],[0,0,0,1]])

        # # Obtener parámetros del eslabón
        # joint_coords = link.joint_coords  # Posición de la articulación
        # axis = link.joint_axis  # Eje de articulación
        # length = link.length  # Longitud del eslabón
        
        # # Crear matriz de transformación para configuración cero
        # R = np.eye(3)  # Sin rotación para posición de referencia
        # p = joint_coords
        # if hasattr(link, 'end_effector_coords') and link.end_effector_coords is not None:
        #     p = link.end_effector_coords
        # elif length > 0:
        #     # Si no hay coordenadas del efector final, usar la longitud del eslabón
        #     if np.linalg.norm(axis) > 1e-6:
        #         axis_norm = axis / np.linalg.norm(axis)
        #         p = joint_coords + length * axis_norm
        #     else:
        #         p = joint_coords
        
        # # Crear matriz de transformación homogénea del eslabón
        # T = Rp2Trans(R, p)
        
        # # Acumular transformación
        # M = M @ T
    
    return M

def calcular_ejes_helicoidales_body_frame(robot, M=None):
    """
    Calcula los ejes helicoidales en el sistema de referencia del efector final (body frame)
    
    Parámetros:
    - robot: Objeto robot con información de sus eslabones y articulaciones
    - M: Matriz de transformación homogénea 4x4 opcional. Si no se proporciona, se calcula.
    
    Retorna:
    - beta: Lista de ejes helicoidales en el sistema de referencia del efector final
    """
    if M is None:
        M = calcular_M_generalizado(robot)
    
    beta = []
    M_inv = np.linalg.inv(M)
    
    for link in robot.links:
        # Crear eje helicoidal S = [ω, v] para cada articulación en el frame base
        if link.tipo == "revolute":
            # Articulación de rotación: S = [ω, -ω × q]
            # donde ω es el eje de rotación y q es un punto en el eje
            omega = np.array(link.joint_axis)
            if np.linalg.norm(omega) > 1e-6:
                omega = omega / np.linalg.norm(omega)
            q = np.array(link.joint_coords)
            v = -np.cross(omega, q)
            S = np.concatenate([omega, v])
        
        elif link.tipo == "prismatic":
            # Articulación prismática: S = [0, v]
            # donde v es la dirección de traslación
            v = np.array(link.joint_axis)
            if np.linalg.norm(v) > 1e-6:
                v = v / np.linalg.norm(v)
            S = np.concatenate([np.zeros(3), v])
        
        # Transformar S al sistema de referencia del efector final usando la adjunta
        S_matrix = calcular_Sθ(S, 1.0)  # Crear matriz asociada al eje
        # Aplicar transformación adjunta: β = Ad_{M^{-1}}(S)
        beta_matrix = M_inv @ S_matrix @ M
        
        # Extraer eje helicoidal β del resultado
        omega_beta = np.array([beta_matrix[2, 1], beta_matrix[0, 2], beta_matrix[1, 0]])
        v_beta = beta_matrix[:3, 3]
        beta_i = np.concatenate([omega_beta, v_beta])
        beta.append(beta_i)
    
    return beta

def calcular_T_robot_body_frame(robot, thetas):
    """
    Calcula la matriz de transformación T usando la fórmula del producto de exponenciales
    con el sistema de referencia en el efector final.
    
    Parámetros:
    - robot: Objeto robot con información de sus eslabones y articulaciones
    - thetas: Lista de ángulos/desplazamientos para cada articulación
    
    Retorna:
    - T: Matriz de transformación homogénea 4x4 desde la base hasta el efector final
    """
    # Calcular matriz M (configuración de referencia)
    M = calcular_M_generalizado(robot)
    
    # Calcular ejes helicoidales en el sistema de referencia del efector final
    beta = calcular_ejes_helicoidales_body_frame(robot, M)
    
    # Calcular transformación usando la fórmula: T(θ) = M * e^([β1]θ1) * ... * e^([βn]θn)
    T = M.copy()
    
    # Multiplicar por las matrices exponenciales en orden (de la primera a la última articulación)
    for i, beta_i in enumerate(beta):
        theta_i = thetas[i]
        exp_beta_i = calcular_expSθ(beta_i, theta_i)
        T = T @ exp_beta_i
    
    return T

def calcular_T_robot(ejes, thetas, M):
    """ Calcula la matriz de transformación homogénea T usando la fórmula del producto de exponenciales. """
    T = np.eye(4)
    for S, theta in zip(ejes, thetas): # ejes,thetas -> eje,theta
        T_i = calcular_expSθ(S, theta) # e^[Sθ]
        T = T @ T_i  # Multiplicación en orden: e^[S1θ1] * e^[S2θ2] * e^[S3θ3]
    T = T @ M  # Multiplicar por M al final
    return T

# Función para calcular la matriz de transformación homogénea del robot 
def calcular_T_met_sit(robot, thetas):
    """
    Un método más sistemático de resolver el problema consiste en definir un SR en cada
    articulación y de esa manera la cinemática directa se resuelve simplemente
    multiplicando las matrices: T04 =T01T12T23T34.
    """
    
    links = robot.links # Obtener los eslabones del robot
    ejes_helicoidales = robot.get_ejes_helicoidales() # Obtener los ejes helicoidales del robot en una lista de ejes
    #thetas = thetas

    # Asegurarse de que el número de ángulos coincide con el número de articulaciones
    if len(thetas) != len(links):
        raise ValueError("El número de ángulos debe coincidir con el número de articulaciones.")
    
    T = np.eye(4)  # Inicializar la matriz de transformación homogénea
    for i in range(len(links)):
        # Obtener el eje helicoidal y el ángulo de la articulación
        S = ejes_helicoidales[i]
        theta = thetas[i]

        T = np.dot(T, Exp2Trans(S, theta))  # Calcular la matriz de transformación homogénea
        imprimir_matriz(T, f"T0_{i+1}")  # Imprimir la matriz de transformación para cada eslabón

    return T

# Función de validación
def validar_transformaciones_helicoidales():
    """ Valida las funciones de transformación con ejes helicoidales usando casos de prueba. """
    print("\n" + "="*80)
    print("VALIDACIÓN DE TRANSFORMACIONES HELICOIDALES")
    print("="*80)
    
    # Casos de prueba
    casos_prueba = [
        # [omega, v, theta, descripción]
        [[1, 0, 0, 0, 0, 0], np.pi/2, "Rotación pura en X 90°"],
        [[0, 1, 0, 0, 0, 0], np.pi/3, "Rotación pura en Y 60°"],
        [[0, 0, 1, 0, 0, 0], np.pi/4, "Rotación pura en Z 45°"],
        [[0, 0, 0, 1, 0, 0], 2.0, "Traslación pura en X"],
        [[1, 0, 0, 1, 0, 0], np.pi/2, "Movimiento helicoidal (rotación X + traslación X)"],
        [[0, 1, 0, 0, 0, 1], np.pi/2, "Movimiento helicoidal (rotación Y + traslación Z)"],
        [[1, 1, 1, 0, 0, 0], np.pi/4, "Rotación en eje arbitrario"]
    ]
    
    for i, (S, theta, descripcion) in enumerate(casos_prueba, 1):
        print(f"\n{'='*50}")
        print(f"CASO {i}: {descripcion}")
        print(f"{'='*50}")
        
        # Convertir a array numpy
        S = np.array(S)
        
        # 1. Crear matriz helicoidal [S]θ
        print("\n1. Matriz helicoidal [S]θ:")
        S_theta = calcular_Sθ(S, theta)
        imprimir_matriz(S_theta, "Matriz [S]θ")
        
        # 2. Calcular la exponencial para obtener T
        print("\n2. Matriz exponencial e^([S]θ):")
        T = calcular_expSθ(S, theta)
        imprimir_matriz(T, "Matriz T = e^([S]θ)")
        
        # 3. Verificar propiedades de T
        print("\n3. Verificación de propiedades:")
        R = T[:3, :3]
        det_R = np.linalg.det(R)
        is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-10)
        
        print(f"Determinante de R: {det_R:.6f} (debe ser ≈ 1)")
        print(f"R es ortogonal: {is_orthogonal}")
        
        # 4. Calcular logaritmo para recuperar S y theta
        print("\n4. Logaritmo de la transformación:")
        theta_recovered, S_recovered = logaritmo_transformacion(T)
        
        print(f"θ original: {theta:.6f}")
        print(f"θ recuperado: {theta_recovered:.6f}")
        print(f"S original: {S}")
        print(f"S recuperado: {S_recovered}")
        
        # 5. Verificar error de reconstrucción
        print("\n5. Error de reconstrucción:")
        T_reconstructed = calcular_expSθ(S_recovered, theta_recovered)
        error = np.linalg.norm(T - T_reconstructed)
        print(f"Error de reconstrucción: {error:.6e}")
        
        # Verificar que el error es pequeño
        assert error < 1e-10, f"Error de reconstrucción significativo en caso {i}"
    
    print("\n" + "="*50)
    print("VALIDACIÓN COMPLETA: Todos los casos pasaron las pruebas.")
    print("="*50)

# Añadir al menú principal
def menu_helicoidales():
    """Menú interactivo para operaciones con ejes helicoidales."""
    def limpiar_pantalla():
        """Limpia la pantalla de la consola."""
        input("\033[93mPresione Enter para continuar...\033[0m")
        os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        print("\n" + "="*90)
        print(" "*20 + "MENÚ DE OPERACIONES CON EJES HELICOIDALES" + " "*20)
        print("="*90)
        print("1. Crear eje helicoidal y calcular su matriz exponencial")
        print("2. Calcular logaritmo de una matriz de transformación")
        print("3. Visualizar eje helicoidal")
        print("4. Validar transformaciones helicoidales")
        print("5. Calcular T con el método sistemático")
        print("0. Volver al menú principal")
        print("-"*90)
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            # Crear eje helicoidal y calcular su matriz exponencial
            print("\nCreación de eje helicoidal:")
            omega = Datos(tipo="vector", mensaje="Ingrese el vector de rotación omega (3 componentes): ").valor
            v = Datos(tipo="vector", mensaje="Ingrese el vector de velocidad v (3 componentes): ").valor
            theta = Datos(tipo="angulo").valor
            
            # Formar el vector S
            S = np.concatenate([omega, v])
            
            # Calcular matriz helicoidal y su exponencial
            S_theta = calcular_Sθ(S, theta)
            T = calcular_expSθ(S, theta)
            
            print("\nEje helicoidal S:")
            print(f"omega = {omega}")
            print(f"v = {v}")
            print(f"theta = {theta}")
            
            imprimir_matriz(S_theta, "Matriz [S]θ")
            imprimir_matriz(T, "Matriz de transformación T = e^([S]θ)")
            limpiar_pantalla()

        elif opcion == "2":
            # Calcular logaritmo de una matriz de transformación
            print("\nCálculo del logaritmo de una matriz de transformación:")
            print("Para crear una matriz de transformación, ingrese:")
            
            # Crear matriz de rotación
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            if eje_input in ["x", "y", "z"]:
                if eje_input == "x":
                    eje = np.array([1, 0, 0])
                elif eje_input == "y":
                    eje = np.array([0, 1, 0])
                else:  # z
                    eje = np.array([0, 0, 1])
            else:
                eje = Datos(tipo="vector", mensaje="Ingrese el eje de rotación (3 componentes): ").valor
                eje = eje / np.linalg.norm(eje)  # Normalizar
            
            angulo = Datos(tipo="angulo").valor
            R = RotRodrigues(eje, angulo)
            
            # Crear vector de traslación
            p = Datos(tipo="vector", mensaje="Ingrese el vector de traslación (3 componentes): ").valor
            
            # Formar matriz de transformación
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p
            
            imprimir_matriz(T, "Matriz de transformación T")
            
            # Calcular logaritmo
            theta, S = logaritmo_transformacion(T)
            
            print("\nLogaritmo de la transformación:")
            print(f"theta = {theta}")
            print(f"S = {S}")
            print(f"omega = {S[:3]}")
            print(f"v = {S[3:]}")
            limpiar_pantalla()

        elif opcion == "3":
            # Visualizar eje helicoidal
            print("\nVisualización de eje helicoidal:")
            omega = Datos(tipo="vector", mensaje="Ingrese el vector de rotación omega (3 componentes): ").valor
            v = Datos(tipo="vector", mensaje="Ingrese el vector de velocidad v (3 componentes): ").valor
            theta = Datos(tipo="angulo").valor
            
            # Formar el vector S
            S = np.concatenate([omega, v])
            
            # Visualizar
            print("Generando visualización...")
            visualizar_eje_helicoidal(S, theta)
            limpiar_pantalla()

        elif opcion == "4":
            # Validar transformaciones helicoidales
            validar_transformaciones_helicoidales()
            limpiar_pantalla()

        elif opcion == "5":
            # Calcular T con el método sistemático
            print("\nCálculo de T usando el método sistemático:")
            robot = robot_structure.cargar_robot_desde_yaml()
            
            pi = np.pi
            print("Matriz de transformación T calculada. Configuración nula:")
            T = calcular_T_met_sit(robot, [0, 0, 0, 0, 0, 0])
            imprimir_matriz(T)
            input("Presione Enter para continuar...")
            print("Matriz de transformación T calculada. Configuración pi:")
            T2 = calcular_T_met_sit(robot, [pi, pi, pi, pi, pi, pi])
            imprimir_matriz(T2)
            input("Presione Enter para continuar...")

        elif opcion == "0":
            print("Volviendo al menú principal...", end=" ")
            limpiar_pantalla()
            break
            
        else:
            print("Opción no válida, intente nuevamente.")
            limpiar_pantalla()

if __name__ == "__main__":
    menu_helicoidales()
