import numpy as np

def create_line(center, dir_vector, length, steps):
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
