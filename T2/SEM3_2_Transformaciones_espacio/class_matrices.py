import numpy as np

# Matriz antisimétrica
def antisimetrica(w):
    """ Matriz antisimétrica asociada a un vector w. """
    return np.array([[0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0]])

# Vector antisimétrico
def antisimetrica_vector(W):
    """ Vector asociado a una matriz antisimétrica W. """
    shape = W.shape
    if shape == (3, 3):
        return np.array([W[2,1], W[0,2], W[1,0]])
    if shape == (4, 4):
        return np.array([W[2,1], W[0,2], W[1,0], W[3,0]])
    if shape != ((3, 3) or (4,4)):
        raise ValueError("Matriz de dimensiones incorrectas. {W.shape}")
    
    return 



# Funciones de visualización y comparación
def imprimir_matriz(M, nombre="Matriz"):
    """
    Imprime la matriz M redondeada a 3 decimales con un encabezado.
    """
    print(f"\n{nombre} =\n{np.round(M, 3)}\n")

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
