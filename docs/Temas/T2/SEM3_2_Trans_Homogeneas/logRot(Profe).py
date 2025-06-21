import numpy as np

# Función de rotación con logaritmica
def LogRot(R):
    """
    Calcula el logaritmo de la matriz de rotación R.
    Parámetros:
    R : matriz de rotación 3x3 (R SO(3))
    Retorna:
    theta : ángulo de rotación en radianes.
    log_R : matriz antisimétrica [hat_omega ]*theta que representa el
    logaritmo de R.
    Se aplican las siguientes condiciones para robustez numérica:
    - Si trace(R) >= 3, se asume R I y theta = 0.
    - Si trace(R) <= -1, se maneja el caso especial con theta = .
    """
    tr = np.trace(R)
    
    # Caso: R casi es la identidad
    if tr >= 3.0:
        theta = 0.0
        return theta , np.zeros ((3,3))
   
    # Caso especial: theta cercano a
    elif tr <= -1.0:
        theta = np.pi
        # Se selecciona el índice con mayor valor diagonal para evitar indeterminaciones
        diag = np.diag(R)
        idx = np.argmax(diag)
        hat_omega = np.zeros(3)
        hat_omega[idx] = np.sqrt((R[idx , idx] - R[(idx+1) %3, (idx+1) %3] + R[(idx +2) %3, (idx +2) %3] + 1) / 2)
        
        # Evitar división por cero
        if np.abs(hat_omega[idx]) < 1e-6:
            hat_omega[idx] = 1e-6
        hat_omega = hat_omega / np.linalg.norm(hat_omega)
        # Usar la fórmula general; aunque en este caso , s_theta 0,
        # la expresión (R - R.T)/(2* sin(theta)) es válida para theta = .
        log_R = (R - R.T) / (2 * np.sin(theta))
        return theta , log_R
    
    # Caso general
    else:
        theta = np.arccos ((tr - 1) / 2)
        s_theta = np.sin(theta)
        # Filtrar posibles divisiones por cero
        if np.abs(s_theta) < 1e-6:
            s_theta = 1e-6
        log_R = (R - R.T) / (2 * s_theta)
        return theta , log_R