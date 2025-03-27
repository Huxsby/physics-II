# Contenido previo de
""" Matrices de transformación homogénea y rotaciones en 3D.    
"""
"""
 - Importar las funciones en el archivo SEM3_1_Rotaciones.py (ejercicios.py)
 - 
"""

import numpy as np                              # Para cálculos numéricos
import matplotlib.pyplot as plt                 # Para visualización
import time                                     # Para medir el tiempo de ejecución
from class_datos import Datos                   # Clase para organizar la toma de datos

# Funciones de rotación
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
        eje_norm = np.array(eje) / np.linalg.norm(eje)              # Normalizar el vector
        #R = RotGen(eje_norm, θ)                                       # Cálculo mediante la fórmula general
        R = RotRodrigues(eje_norm, θ)                               # Cálculo mediante la fórmula de Rodrigues
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

# Matriz antisimétrica
def antisimetrica(w):
    """ Matriz antisimétrica asociada a un vector w. """
    return np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])

# Vector antisimétrico
def antisimetrica_vector(W):
    """ Vector asociado a una matriz antisimétrica W. """
    return np.array([W[2,1], W[0,2], W[1,0]])

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

# Funciones de visualización y comparación
def imprimir_matriz(M, nombre="Matriz"):
    """
    Imprime la matriz M redondeada a 3 decimales con un encabezado.
    """
    print(f"\n{nombre} =\n{np.round(M, 3)}\n")

# Visualización de rotación
def Visualizar(vector, eje):
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

# Comparar rotaciones
def comparar_rotaciones(w , θ):
    w = np.array(w) / np.linalg.norm(w) # Normalizar el vector
    R1 = RotGen(w , θ)
    R2 = RotRodrigues(w , θ)
    diferencia = np.linalg.norm(R1 - R2)
    return R1 , R2 , diferencia

# Validación de rotaciones
def validar_rotaciones():
    """
    Función para validar rotaciones usando diferentes métodos:
    1. Rodrigues vs Rot
    2. LogRot para recuperar ángulo y eje de rotación
    3. Rotaciones inversas
    4. Casos predefinidos con diferentes ejes y ángulos
    """
    print("\n" + "="*80)
    print("VALIDACIÓN DETALLADA DE ROTACIONES")
    print("="*80)
    
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
        print(f"\n{'='*50}")
        print(f"CASO {i}:")
        print(f"{'='*50}")
        
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
        
        print(f"\nVector original):  {eje_norm}")
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
        
        print("\n" + "-"*50)
        
        # Criterios de validación
        assert diff_metodos < 1e-10, f"Error: Diferencia significativa entre métodos de rotación en Caso {i}"
        assert diff_vectores < 1e-10, f"Error: Diferencia significativa en vectores rotados en Caso {i}"
        assert np.abs(angulo - angulo_recuperado) < 1e-10, f"Error: Ángulo no recuperado correctamente en Caso {i}"
        assert diff_recuperado < 1e-10, f"Error: Vector no recuperado correctamente en Caso {i}"
    
    print("\n" + "="*50)
    print("VALIDACIÓN COMPLETA: Todos los casos pasaron las pruebas.")
    print("="*50)

# Menú interactivo
def menu():
    """Menú interactivo para seleccionar acciones."""
    while True:
        print("\n" + "="*90)
        print(" "*30 + "MENÚ DE OPCIONES" + " "*30)
        print("="*90)
        print("Nota. Los vectores que se tomen como ejes serán convertidos a unitarios automáticamente.")
        print("="*90)
        # Añadir content
        print("1. Rotar un vector entorno a un eje específico (x,y,z).")    
        print("2. Rotar un vector entorno a un eje genérico.")
        print("3. Comparar rotaciones con fórmula generar vs Rodrigues.")
        print("4. Visualizar rotación de un vector entorno a un eje específico.")
        print("5. Aplicar logaritmo de una matriz de rotación.")
        print("6. Validar rotaciones y funciones (casos predefinidos).")
        print("0. Salir.")
        print("-"*90)

        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1" or opcion == "2":              # 1. y 2. Rotar un vector
            vector = Datos(tipo="vector").valor
            if opcion == "1":                           # 1. Rotar entorno a un eje específico
                eje = Datos(tipo="eje").valor
            else:                                       # 2. Rotar entorno a un eje genérico
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            angulo = Datos(tipo="angulo").valor
            
            # Rotar el vector
            vector_rotado = RotarVector(vector, eje, angulo)  
            print(f"\nVector original: {vector}")
            print(f"Vector rotado: {vector_rotado}")
            continue
        
        elif opcion == "3":                             # 3. Comparar rotaciones
            R1 , R2 , diff = comparar_rotaciones(Datos(tipo="vector").valor, Datos(tipo="angulo").valor)
            imprimir_matriz(R1 , "R (Definición Explícita)")
            imprimir_matriz(R2 , "R (Rodrigues)")
            print("Diferencia entre métodos:", round(diff , 4))
            
        elif opcion == "4":                             # 4. Visualizar rotación
            vector = Datos(tipo="vector").valor
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            if eje_input in ["x", "y", "z"]:
                eje = eje_input
            else:
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            
            Visualizar(vector, eje)

        elif opcion == "5":                             # 5. Calcular logaritmo de una matriz de rotación
            # Obtener matriz de rotación para cálculo del logaritmo
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            
            # Convertir eje de tipo string a vector unitario o normalizar eje genérico
            if eje_input == "x":
                eje = np.array([1, 0, 0])
            elif eje_input == "y":
                eje = np.array([0, 1, 0])
            elif eje_input == "z":
                eje = np.array([0, 0, 1])
            else:
                # Obtener vector del usuario y normalizarlo
                eje = np.array(Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor)
                eje = eje / np.linalg.norm(eje)  # Normalizar a vector unitario
            
            angulo = Datos(tipo="angulo").valor
            R = RotRodrigues(eje, angulo)
            
            # Calcular logaritmo de la matriz de rotación
            angulo_result, eje_resultado = LogRot(R)
            
            print(f"\nÁngulo original (rads): {round(angulo, 3)}")
            print(f"Ángulo recuperado (rads): {round(angulo_result, 3)}")
            print(f"Eje de rotación original: {eje}")
            print(f"Eje de rotación recuperado: {eje_resultado}")                                 

        elif opcion == "6":                             # 6. Validación del sistema de calculo
            validar_rotaciones()

        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

#AdT 6x6
def AdT(T):
    """
    Calcula la matriz AdT asociada a una matriz de transformación homogénea T.
    
    La matriz AdT se utiliza para transformar el producto de dos matrices de transformación homogénea"
    """
    R = T[:3, :3]  # Extracción de la matriz de rotación
    p = T[:3, 3]   # Extracción del vector de traslación
    
    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[:3, 3:] = np.dot(antisimetrica(p), R)
    
    return AdT

# Vector giro


if __name__ == "__main__":
    menu()
