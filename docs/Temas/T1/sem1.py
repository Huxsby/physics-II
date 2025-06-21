""" Función del documento 'seminario1.pdf' """
import numpy as np

### 1.1. Segmento rectilíneo
def create_line(center, dir_vector, length, steps):
    """
    Crea un segmento de línea en el espacio 3D.
    Parameters:
    center (array-like): Coordenadas del punto central del segmento. Se espera un array o lista de 3 elementos [x, y, z].
    dir_vector (array-like): Vector de dirección del segmento. Se espera un array o lista de 3 elementos [dx, dy, dz].
    length (float): Longitud total del segmento.
    steps (int): Número de pasos o puntos en los que se divide el segmento. Define la resolución del segmento.
    Returns:
    numpy.ndarray: Un array de forma (steps+1, 3) que contiene las coordenadas de los puntos del segmento.
    """
    v = np.array(dir_vector)
    v = v / np.linalg.norm(v)
    start = np.array(center) - length * 0.5 * v # punto inicial del segmento
   
    # El número de steps define la resolución del segmento
    points = np.zeros((steps+1, 3))
   
    # Generamos los puntos del segmento
    for i in range(steps+1):
        x = length / steps * i
        points[i,:] = x * v + start
   
    # Devolvemos el array que contiene los puntos del segmento
    return points

### 1.2. Paralelogramo
def create_paralelogram(center, v1, v2, l1, l2, steps):
    """
    Crea un paralelogramo en el espacio 3D y devuelve los puntos que forman sus bordes.
    Parameters:
    center (array-like): Coordenadas del centro del paralelogramo. Se espera una lista o array de longitud 3.
    v1 (array-like): Vector director del primer lado del paralelogramo. Se espera una lista o array de longitud 3.
    v2 (array-like): Vector director del segundo lado del paralelogramo. Se espera una lista o array de longitud 3.
    l1 (float): Longitud del primer lado del paralelogramo.
    l2 (float): Longitud del segundo lado del paralelogramo.
    steps (int): Número de pasos para discretizar cada lado del paralelogramo.
    Returns:
    numpy.ndarray: Un array de forma (4*(steps+1)+1, 3) que contiene los puntos que forman los bordes del paralelogramo.
    """
    v1 = np.array(v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array(v2)
    v2 = v2 / np.linalg.norm(v2)
    
    # vértices del paralelogramo
    s1 = np.array(center) - l1 * 0.5 * v1 - l2 * 0.5 * v2
    s2 = np.array(center) + l1 * 0.5 * v1 - l2 * 0.5 * v2
    s3 = np.array(center) + l1 * 0.5 * v1 + l2 * 0.5 * v2
    s4 = np.array(center) - l1 * 0.5 * v1 + l2 * 0.5 * v2
    points = np.zeros((4*(steps+1)+1, 3))
    
    # Generamos los puntos de cada segmento
    for i in range(steps+1):
        x1 = l1 / steps * i; x2 = l2 / steps * i
        points[i,:] = x1 * v1 + s1
        points[steps+1+i,:] = x2 * v2 + s2
        points[2*steps+2+i,:] = -x1 * v1 + s3
        points[3*steps+3+i,:] = -x2 * v2 + s4
    points[-1,:] = s1 #repetimos el primer punto porque es una figura cerrada
    
    # Devolvemos el array que contiene los puntos del segmento
    return points

### 1.3. Polígono Regular
def create_polygon(center, normal_vector, radius, steps):
    """
    Crea un polígono en 3D dado un centro, un vector normal, un radio y el número de pasos.
    Parameters:
    center (array-like): Coordenadas del centro del polígono. Se espera un array o lista de longitud 3.
    normal_vector (array-like): Vector normal al plano del polígono. Se espera un array o lista de longitud 3.
    radius (float): Radio de la circunferencia que contiene al polígono.
    steps (int): Número de lados del polígono.
    Returns:
    numpy.ndarray: Un array de forma (steps+1, 3) que contiene las coordenadas de los puntos del polígono.
    """
    n = np.array(normal_vector)
    n = n / np.linalg.norm(n)
    c = np.array(center)
    
    # Generamos un vector aleatorio no nulo y que no esté alineado con n
    while True:
        temp = np.random.rand(3)
        if np.linalg.norm(temp) < 1e-5: continue
        temp = temp / np.linalg.norm(temp)
        if np.dot(temp, n) < 0.8: break
    
    # El producto vectorial de n y temp es un vector que va en la
    # dirección radial de la circunferencia que contiene al polígono
    u = np.cross(n, temp)
    
    # El producto vectorial del vector normal y del vector
    # radial es un vector tangente a la circunferencia
    v = np.cross(n, u)
    points = np.zeros((steps+1, 3))
    
    # Generamos los puntos del polígono
    for i in range(steps):
        angle = 2.0 * np.pi / steps * i
        points[i,:] = radius * (np.cos(angle) * u + np.sin(angle) * v) + c
    points[-1] = radius * u + c
    
    # Devolvemos el array que contiene los puntos del polígono
    return points
