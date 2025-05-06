""""
Clase para la creación de matrices de rotación en 3D utilizando diferentes métodos.
"""

import numpy as np                              # Para operaciones matemáticas
import matplotlib.pyplot as plt                 # Para visualización
import time                                     # Para medir el tiempo de ejecución

""" Funciones de rotación en torno a los ejes de coordenadas """

# Funciones de rotación en torno a los ejes de coordenadas
def Rotx(θ):
    """ Matriz de rotación en torno al eje X """
    return np.array([[1, 0, 0],
                    [0, np.cos(θ), -np.sin(θ)],
                    [0, np.sin(θ), np.cos(θ)]])

def Roty(θ):
    """ Matriz de rotación en torno al eje Y """
    return np.array([[np.cos(θ), 0, np.sin(θ)],
                    [0, 1, 0],
                    [-np.sin(θ), 0, np.cos(θ)]])

def Rotz(θ):
    """ Matriz de rotación en torno al eje Z """
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
    print(f"\t\033[92mTiempo de ejecución de Rot(w, θ) {time.time() - start} segundos\033[0m")
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
    
    print(f"\t\033[92mTiempo de ejecución de RotRodrigues(w, θ) {time.time() - start} segundos\033[0m")
    return matrix

# Función de rotación con logaritmica
def LogRot(R):
    """
    Calcula el logaritmo de la matriz de rotación R.
    Parámetros:
    R : matriz de rotación 3x3 (R SO(3))
    Retorna:
    θ : ángulo de rotación en radianes.
    log_R : matriz antisimétrica [hat_omega ]*θ que representa el logaritmo de R.
    Se aplican las siguientes condiciones para robustez numérica:
    - Si trace(R) >= 3, se asume R I y θ = 0.
    - Si trace(R) <= -1, se maneja el caso especial con θ = .
    """
    tr = np.trace(R)
    
    # Caso: R casi es la identidad
    if tr >= 3.0:
        θ = 0.0
        return θ, np.zeros(3)
   
    # Caso especial: θ cercano a
    elif tr <= -1.0:
        θ = np.pi
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
        # Usar la fórmula general; aunque en este caso , s_θ 0,
        # la expresión (R - R.T)/(2* sin(θ)) es válida para θ = .
        log_R = (R - R.T) / (2 * np.sin(θ))
        return θ, antisimetrica_vector(log_R)
    
    # Caso general
    else:
        θ = np.arccos((tr - 1) / 2)
        s_θ = np.sin(θ)
        # Filtrar posibles divisiones por cero
        if np.abs(s_θ) < 1e-6:
            s_θ = 1e-6
        log_R = (R - R.T) / (2 * s_θ)
        return θ, antisimetrica_vector(log_R)

# Función para convertir una matriz de rotación en ángulos de Euler
def R2Euler(R):
    """ Convierte una matriz de rotación R en ángulos de Euler (roll, pitch, yaw). """
    sy=np.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
    singular=sy<1.e-6
    if not singular:
        x=np.arctan2(R[2,1],R[2,2])
        y=np.arctan2(-R[2,0],sy)
        z=np.arctan2(R[1,0],R[0,0])
    else:
        x=np.arctan2(-R[1,2],R[1,1])
        y=np.arctan2(-R[2,0],sy)
        z=0.
    return np.array([x,y,z])

# Función para convertir ángulos de Euler en una matriz de rotación
def Euler2R(roll, pitch, yaw):
    """ Convierte ángulos de Euler (roll, pitch, yaw) en una matriz de rotación R. """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return np.dot(R_z, np.dot(R_y, R_x)) # Producto matricial de las rotaciones

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

# Función para validar el sistema de rotaciones
def validar_rotaciones():
    """
    Función para validar rotaciones usando diferentes métodos:
    1. Rodrigues vs Rot
    2. LogRot para recuperar ángulo y eje de rotación
    3. Rotaciones inversas
    4. Casos predefinidos con diferentes ejes y ángulos
    """
    print("\n" + "="*90)
    print("VALIDACIÓN DETALLADA DE ROTACIONES")
    print("="*90)
    
    # Casos de prueba predefinidos
    casos_prueba = [
        # [vector, eje, ángulo]
        [[1, 0, 0], [1, 0, 0], np.pi/2],      # 90° rotación alrededor de eje X
        [[0, 1, 0], [0, 1, 0], np.pi/4],      # 45° rotación alrededor de eje Y
        [[0, 0, 1], [0, 0, 1], np.pi/3],      # 60° rotación alrededor de eje Z
        [[1, 1, 1], [1, 1, 1], np.pi/6],      # 30° rotación alrededor de eje genérico diagonal
        [[1, 2, 3], [0, 1, 0], np.pi/2]       # 90° rotación alrededor de eje Y
    ]
    
    for i, (vector, eje, angulo) in enumerate(casos_prueba, 1):
        print(f"\n{'='*90}")
        print(f"CASO {i}:")
        print(f"{'='*90}")
        
        # Información inicial
        print(f"Vector original: {vector}")
        print(f"Eje de rotación: {eje}")
        print(f"Ángulo de rotación: {np.degrees(angulo):.2f}°")
        
        # Normalizar eje de rotación
        eje_norm = np.array(eje) / np.linalg.norm(eje)
        print(f"Eje de rotación normalizado: {eje_norm}")
        
        # Método 1: Comparar Rot vs RotRodrigues
        print("\n1. Comparación de métodos de rotación:")
        R1 = RotGen(eje_norm, angulo)
        R2 = RotRodrigues(eje_norm, angulo)
        
        print("\nMatriz de rotación (Método Explícito - Rot):")
        imprimir_matriz(R1, "R1")
        
        print("\nMatriz de rotación (Método Rodrigues - RotRodrigues):")
        imprimir_matriz(R2, "R2")
        
        diff_metodos = np.linalg.norm(R1 - R2)
        print(f"Diferencia entre matrices de rotación: {diff_metodos:.2e}")
        
        # Método 2: Rotar vector y verificar
        print("\n2. Rotación de vector:")
        vector_np = np.array(vector)
        vector_rotado1 = np.dot(R1, vector_np)
        vector_rotado2 = np.dot(R2, vector_np)
        
        print(f"Vector original:      {vector_np}")
        print(f"Vector rotado (R1):   {vector_rotado1}")
        print(f"Vector rotado (R2):   {vector_rotado2}")
        
        diff_vectores = np.linalg.norm(vector_rotado1 - vector_rotado2)
        print(f"Diferencia entre vectores rotados: {diff_vectores:.2e}")
        
        # Método 3: Recuperar ángulo con LogRot
        print("\n3. Recuperación de logaritmo de rotación:")
        angulo_recuperado, eje_recuperado = LogRot(R1)
        
        print(f"Ángulo original:     {np.degrees(angulo):.2f}°")
        print(f"Ángulo recuperado:   {np.degrees(angulo_recuperado):.2f}°")
        
        print(f"\nVector original:  {eje_norm}")
        print(f"Vector recuperado: {eje_recuperado}")
        
        # Método 4: Aplicar rotación inversa para verificar simetría
        print("\n4. Verificación de rotación inversa:")
        R_inversa = R1.T  # Matriz transpuesta = inversa en matrices de rotación
        vector_recuperado = np.dot(R_inversa, vector_rotado1)
        
        print(f"Vector original:      {vector_np}")
        print(f"Vector rotado:        {vector_rotado1}")
        print(f"Vector recuperado:    {vector_recuperado}")
        
        diff_recuperado = np.linalg.norm(vector_np - vector_recuperado)
        print(f"Diferencia al recuperar vector original: {diff_recuperado:.2e}")
        
        print("\n5. Verificación de propiedades de rotación:")
        # Verificar propiedades de matrices de rotación
        print("Determinante de R1:  {:.2f}".format(np.linalg.det(R1)))
        print("Transpuesta de R1 == Inversa de R1: {}".format(np.allclose(R1.T, np.linalg.inv(R1))))
        
        print("\n" + "-"*90)
        
        # Criterios de validación
        assert diff_metodos < 1e-10, f"Error: Diferencia significativa entre métodos de rotación en Caso {i}"
        assert diff_vectores < 1e-10, f"Error: Diferencia significativa en vectores rotados en Caso {i}"
        assert np.abs(angulo - angulo_recuperado) < 1e-10, f"Error: Ángulo no recuperado correctamente en Caso {i}"
        assert diff_recuperado < 1e-10, f"Error: Vector no recuperado correctamente en Caso {i}"
    
    print("\n" + "="*90)
    print("VALIDACIÓN COMPLETA: Todos los casos pasaron las pruebas.")
    print("="*90)

""" Funciones de matrices generales """

# Matriz antisimétrica
def antisimetrica(w):
    """ Matriz antisimétrica asociada a un vector w de 3 o 6 elementos. """
    shape = w.shape
    if shape == (3,):
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])
    if shape == (6,):
        return np.array([[  0, -w[0], -w[1], -w[2]],
                         [ w[0],   0, -w[3], -w[4]],
                         [ w[1],  w[3],   0, -w[5]],
                         [ w[2],  w[4],  w[5],   0]])
    else:
        raise ValueError(f"El vector debe tener dimensión 3 o 6. Dimension ({shape}) incorrecta.")
    
# Vector antisimétrico
def antisimetrica_vector(W):
    """ Vector asociado a una matriz antisimétrica W. """
    shape = W.shape
    if shape == (3, 3):
        return np.array([W[2,1], W[0,2], W[1,0]])
    if shape == (4, 4):
        return np.array([W[2,1], W[0,2], W[1,0], W[3,0]])
    else:
        raise ValueError(f"Matriz de dimensiones incorrectas ({shape}). Debe ser 3x3 o 4x4.")

# Funciones de visualización y comparación
def imprimir_matriz(M, nombre="Matriz"):
    """ Imprime la matriz M redondeada a 3 decimales con un encabezado. """
    print(f"\n{nombre} =\n{np.round(M, 3)}\n")

""" Funciones de transformación homogénea """

# Función para convertir un vector de 6 elementos en una matriz de transformación homogénea
def Exp2Trans(S, θ):
    """
    Convierte un vector de 6 elementos en una matriz de transformación homogénea 6x6.
    (La matriz exponencial del vector S corresponde con la de transformación homogénea).
    """
    if S.shape != (6,):
        raise ValueError("El vector S debe tener tamaño 6.")
    
    # Extraer la parte angular y la parte de traslación del vector
    w = S[:3]  # w Parte angular (vector de rotación)
    v = S[3:]  # v Parte de traslación (vector de posición)
    modulo_w = np.linalg.norm(w)
    
    if modulo_w == 0 and np.linalg.norm(v) == 1: # Articulación prísmaticas
        # Si el vector de rotación es cero y el vector de traslación es unitario:
        T = np.eye(4)
        T[:3, 3] = v * θ    # Asignar la traslación
        return T
        """
        return np.array([[1, 0, 0, v[0]]*θ,
                        [0, 1, 0, v[1]]*θ,
                        [0, 0, 1, v[2]]*θ,
                        [0, 0, 0, 1]])
        """
    elif modulo_w == 1 and θ.imag == 0:          # Articulación rotativa
        # Si el vector de rotación es unitario y el ángulo es real, se aplica la fórmula de Rodrigues
        R = RotRodrigues(w, θ)
        T = np.eye(4)
        T[:3, :3] = R       # Asignar la rotación (Se sobrescribe la matriz identidad)
        T[:3, 3] = v * θ    # Asignar la traslación
        return T
    else:
        raise ValueError("El vector S no es válido para la transformación homogénea.")

# Función para convertir una matriz de rotación y un vector de traslación en una matriz de transformación homogénea
def Rp2Trans(R, p):
    """ Convierte una matriz de rotación R y un vector de traslación p en una matriz de transformación homogénea 4x4. """
    if R.shape != (3, 3) or p.shape != (3,):
        raise ValueError("La matriz de rotación debe ser de tamaño 3x3 y el vector de traslación debe ser de tamaño 3.")
    
    T = np.eye(4)  # Matriz identidad 4x4
    T[:3, :3] = R  # Asignar la rotación
    T[:3, 3] = p   # Asignar la traslación
    
    return T

# Función para convertir una matriz de transformación homogénea en una matriz de rotación y un vector de traslación
def Trans2Rp(T):
    """ Convierte una matriz de transformación homogénea T en una matriz de rotación R y un vector de traslación p. """
    if T.shape != (4, 4):
        raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
    return T[:3, :3], T[:3, 3]

# Función para calcular el logaritmo de una matriz de transformación homogénea devolviendo S = (w, p)
def LogTrans(T, mode='rp'):
    """Calcula el logaritmo de una matriz de transformación homogénea T. 
        Args:
            T (np.ndarray): Matriz de transformación homogénea de 4x4.
            mode (str, optional): Modo de retorno. Si es 'rp', devuelve la matriz de rotación y el vector de traslación por separado. Si es 's', devuelve el vector S = (w,v), que es la concatenación del vector de rotación y del vector de traslación. Defaults to 'rp'.
        Raises:
            ValueError: Si la matriz de transformación no es de tamaño 4x4.
        Returns:
            tuple or np.ndarray: Dependiendo del modo, devuelve la matriz de rotación y el vector de traslación por separado, o el vector S = (w,v).
    """
    if T.shape != (4, 4):
        raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
    
    R = T[:3, :3]       # Extraer la matriz de rotación
    p = T[:3, 3]        # Extraer el vector de traslación
    θ, w = LogRot(R)    # Calcular el logaritmo de la matriz de rotación
    
    if mode.lower == "rp": return R, p # Devuelve la matriz de rotación y el vector de traslación por separado 
    elif mode.lower == "s": return np.concatenate((w, p)) # Devuelve el vector S = (w,v), que es la concatenación del vector de rotación y del vector de traslación

# Función para calcular la inversa de una matriz de transformación homogénea
def TransInv(Tr):
    """ Calcula la inversa de una matriz de transformación homogénea T. """
    if Tr.shape != (4, 4):
        raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
    Inv = np.eye(4)  # Matriz identidad 4x4
    Inv[:3, :3] = Tr[:3, :3].T  # Transponer la matriz de rotación
    Inv[:3, 3] = -np.dot(Tr[:3, :3].T, Tr[:3, 3])  # Calcular la traslación inversa
    return Inv

# Función para calcular la matriz adjunta de una matriz de transformación homogénea
def TransAdj(T):
    """
    Calcula la matriz AdjT asociada a una matriz de transformación homogénea T. 
    La matriz AdjT se utiliza para transformar el producto de dos matrices de transformación homogénea"
    """
    if T.shape != (4, 4):
        raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
    
    R = T[:3, :3]  # Extracción de la matriz de rotación
    p = T[:3, 3]   # Extracción del vector de traslación
    
    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[:3, 3:] = np.dot(antisimetrica(p), R)
    
    return AdT

