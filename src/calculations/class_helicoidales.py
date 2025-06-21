import numpy as np
from .class_rotaciones import RotRodrigues, LogRot, antisimetrica, imprimir_matriz
from ..core import Robot, cargar_robot_desde_yaml

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
def calcular_exp_Sθ(S, theta):
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
            T_t = calcular_exp_Sθ(S, t)
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

# Dentro de 'robot.M':
# def calcular_M_generalizado(robot: Robot):
#     """ Calcula la matriz de transformación homogénea M del robot. """
#     M = np.eye(4)  # Matriz identidad inicial
#     for link in robot.links:
#         M[:3, 3] += link.joint_coords  # Sumar la posición de la articulación
        
#     return M

def calcular_ejes_helicoidales_body_frame(robot: Robot, M=None):
    """
    Calcula los ejes helicoidales en el sistema de referencia del efector final (body frame)
    usando los ejes helicoidales almacenados en el robot.
    """
    import numpy as np
    from class_helicoidales import calcular_Sθ

    if M is None:
        M = robot.M
    beta = []
    M_inv = np.linalg.inv(M)
    for S in robot.ejes_helicoidales:  # Usar los ejes helicoidales almacenados
        S_matrix = calcular_Sθ(S, 1.0)
        beta_matrix = M_inv @ S_matrix @ M
        omega_beta = np.array([beta_matrix[2, 1], beta_matrix[0, 2], beta_matrix[1, 0]])
        v_beta = beta_matrix[:3, 3]
        beta_i = np.concatenate([omega_beta, v_beta])
        beta.append(beta_i)
    return beta

# Funcion para calcular la matriz de transformación homogénea del robot
def calcular_T_robot(ejes, thetas, M):
    """ Calcula la matriz de transformación homogénea T con los ejes helicoidales y la matriz de transformación M, usando la fórmula del producto de exponenciales. """
    T = np.eye(4)
    for S, theta in zip(ejes, thetas): # ejes,thetas -> eje,theta
        T_i = calcular_exp_Sθ(S, theta) # e^[Sθ]
        T = T @ T_i  # Multiplicación en orden: e^[S1θ1] * e^[S2θ2] * e^[S3θ3]
    T = T @ M  # Multiplicar por M al final
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
        print(f"\n{'='*90}")
        print(f"CASO {i}: {descripcion}")
        print(f"{'='*90}")
        
        # Convertir a array numpy
        S = np.array(S)
        
        # 1. Crear matriz helicoidal [S]θ
        print("\n1. Matriz helicoidal [S]θ:")
        S_theta = calcular_Sθ(S, theta)
        imprimir_matriz(S_theta, "Matriz [S]θ")
        
        # 2. Calcular la exponencial para obtener T
        print("\n2. Matriz exponencial e^([S]θ):")
        T = calcular_exp_Sθ(S, theta)
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
        T_reconstructed = calcular_exp_Sθ(S_recovered, theta_recovered)
        error = np.linalg.norm(T - T_reconstructed)
        print(f"Error de reconstrucción: {error:.6e}")
        
        # Verificar que el error es pequeño
        assert error < 1e-10, f"Error de reconstrucción significativo en caso {i}"
    
    print("\n" + "="*90)
    print("VALIDACIÓN COMPLETA: Todos los casos pasaron las pruebas.")
    print("="*90)

if __name__ == "__main__":
    validar_transformaciones_helicoidales()
