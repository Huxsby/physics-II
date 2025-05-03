import sympy as sp
import numpy as np
import time

from class_robot_structure import Robot, cargar_robot_desde_yaml

""" Funciones de calculo simbólico de la Jacobiana"""

def VecToso3sp(v: sp.Matrix):
    v = sp.Matrix(v)
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def MatrixExp3sp(w: sp.Matrix, theta):
    """ Calcula la matriz exponencial de un vector w en SO(3) usando la fórmula de Rodrigues. """
    w = sp.Matrix(w).normalized()
    omgmat = VecToso3sp(w)
    return sp.eye(3) + sp.sin(theta) * omgmat + (1 - sp.cos(theta)) * (omgmat * omgmat)

def MatrixExp6sp(S: sp.Matrix, theta):
    """ Calcula la matriz exponencial de un elemento de Lie en SE(3) usando la fórmula de Rodrigues. """
    w = sp.Matrix(S[0:3])
    v = sp.Matrix(S[3:6])
    omgmat = VecToso3sp(w)

    # Check for pure translation
    if w.norm() < 1e-5: # Use a small tolerance for floating point comparison
        # Simplified matrix exponential for pure translation
        return sp.eye(3).row_join(v * theta).col_join(sp.Matrix([[0, 0, 0, 1]]))
    else:
        # Standard matrix exponential for general screw axis
        R = MatrixExp3sp(w, theta)
        G_theta = sp.eye(3) * theta + (1 - sp.cos(theta)) * omgmat + (theta - sp.sin(theta)) * (omgmat * omgmat)
        p = G_theta * v
        return R.row_join(p).col_join(sp.Matrix([[0, 0, 0, 1]]))

def Adjunta(T: sp.Matrix):
    """ Calcula la matriz adjunta de una transformación homogénea T en el espacio SE(3). """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_skew = VecToso3sp(p)

    upper = R.row_join(sp.zeros(3))
    lower = (p_skew * R).row_join(R)
    return upper.col_join(lower)

def calcular_jacobiana(robot: Robot):
    """
    Calcula la matriz Jacobiana simbólica del robot proporcionado usando la
    formulación del espacio (Space Jacobian).

    Parámetros:
    - robot: objeto Robot cargado (de tu clase Robot)

    Devuelve:
    - Jacobian: matriz simbólica 6xn de la Jacobiana del Espacio
    - thetas: variables simbólicas correspondientes a las articulaciones
    """
    time_i = time.time()  # Iniciar temporizador
    # Obtener ejes helicoidales en la posición cero (referencia espacial {s})
    ejes_helicoidales = robot.get_ejes_helicoidales()
    n = len(ejes_helicoidales)

    # Crear variables simbólicas para los ángulos/desplazamientos de las articulaciones
    thetas = sp.symbols(f't0:{n}')  # t0, t1, ..., tn-1

    # Calcular las transformaciones homogéneas acumuladas T_i = exp([S1]t1)*...*exp([Si]ti)
    Ts = []
    T = sp.eye(4)
    for i in range(n):
        S = ejes_helicoidales[i]
        T_i = MatrixExp6sp(S, thetas[i])
        T = T * T_i
        Ts.append(T) # Ts[i] = T_1 * T_2 * ... * T_{i+1}

    # Calcular las columnas de la Jacobiana del Espacio J_s
    J_cols = []
    # La primera columna es simplemente el primer eje helicoidal S1
    J_cols.append(sp.Matrix(ejes_helicoidales[0]))

    # Las columnas subsiguientes J_si = Ad(T_{i-1}) * Si
    # donde T_{i-1} = exp([S1]t1)*...*exp([S_{i-1}]t_{i-1})
    for i in range(1, n):
        # T_prev = T_0 * T_1 * ... * T_{i} -> Ts[i-1]
        T_prev = Ts[i-1]
        Ad = Adjunta(T_prev)
        S = sp.Matrix(ejes_helicoidales[i])
        J_cols.append(Ad * S)

    Jacobian = sp.Matrix.hstack(*J_cols)

    print(f"\t\033[92mTiempo de cálculo de la Jacobiana del robot {robot.name}: {time.time() - time_i:.4f} segundos\033[0m")
    return Jacobian, thetas

""" Funciones de visualización de la Jacobiana"""

def mostrar_jacobiana_resumida(Jacobian: sp.Matrix, max_chars=30):
    """ Muestra la matriz Jacobiana de forma resumida, limitando el número de caracteres por elemento. """
    # Si la entrada es numérica (numpy array), usarla directamente
    if isinstance(Jacobian, np.ndarray):
        matrix_data = Jacobian
        rows, cols = matrix_data.shape
        is_symbolic = False
    # Si es simbólica (sympy Matrix), convertir a texto
    elif isinstance(Jacobian, sp.Matrix):
        matrix_data = Jacobian
        rows, cols = matrix_data.shape
        is_symbolic = True
    else:
        print("Error: La entrada debe ser una matriz SymPy o un array NumPy.")
        return

    # Convertir elementos a texto y truncar si es necesario
    matrix_text = []
    for i in range(rows):
        row_text = []
        for j in range(cols):
            if is_symbolic:
                elem = str(matrix_data[i, j])
            else:
                # Formatear números flotantes
                elem = f"{matrix_data[i, j]:.3f}" # 3 decimales por defecto
            if len(elem) > max_chars:
                elem = elem[:max_chars] + "..."
            row_text.append(elem)
        matrix_text.append(row_text)

    # Calcular el ancho máximo de cada columna para alinear
    col_widths = [max(len(matrix_text[r][c]) for r in range(rows)) for c in range(cols)]

    # Imprimir la matriz formateada con bordes
    print()
    for i in range(rows):
        formatted_row = []
        for j in range(cols):
            elem = matrix_text[i][j]
            # Alinear a la derecha para números, izquierda para simbólicos/truncados
            if not is_symbolic or "..." in elem:
                 formatted_row.append(elem.ljust(col_widths[j]))
            else:
                 formatted_row.append(elem.rjust(col_widths[j]))

        row_str = "  ".join(formatted_row)
        if i == 0:
            print(f"⎡ {row_str} ⎤")
        elif i == rows - 1:
            print(f"⎣ {row_str} ⎦")
        else:
            print(f"⎢ {row_str} ⎥")

def prueba_jacobiana():
    """ Función de prueba para calcular y mostrar la Jacobiana de un robot. """
    robot = cargar_robot_desde_yaml("robot.yaml") # Carga del robot
    Jacobian, thetas = calcular_jacobiana(robot) # Calcular Jacobiana Simbólica

    # Mostrar Jacobiana simbólica de forma resumida
    print("\n--- Jacobiana Simbólica (Resumida) ---")
    mostrar_jacobiana_resumida(Jacobian)
    input("Presiona Enter para continuar con las evaluaciones numéricas...")

    # --- Prueba 1: Valores parciales (ejemplo: solo t0 varía, el resto 0) ---
    print("\n--- Prueba con valores parciales (t0 variable, resto 0) ---")
    # Crea un diccionario donde todas las thetas excepto la primera (thetas[0]) se ponen a 0
    valores_parciales_t0 = {theta: 0 for i, theta in enumerate(thetas) if i != 0}
    Jacobian_parcial_t0 = Jacobian.subs(valores_parciales_t0)
    print(f"Sustituyendo: {valores_parciales_t0}")
    mostrar_jacobiana_resumida(Jacobian_parcial_t0) # Mostrar simbólicamente con t0
    input("Presiona Enter...")

    # --- Prueba 2: Valores parciales (ejemplo: solo t1 varía, el resto 0) ---
    print("\n--- Prueba con valores parciales (t1 variable, resto 0) ---")
    # Crea un diccionario donde todas las thetas excepto la segunda (thetas[1]) se ponen a 0
    valores_parciales_t1 = {theta: 0 for i, theta in enumerate(thetas) if i != 1}
    Jacobian_parcial_t1 = Jacobian.subs(valores_parciales_t1)
    print(f"Sustituyendo: {valores_parciales_t1}")
    mostrar_jacobiana_resumida(Jacobian_parcial_t1) # Mostrar simbólicamente con t1
    input("Presiona Enter...")

    # --- Prueba 3: Todos los valores a cero ---
    print("\n--- Prueba con todos los valores a cero ---")
    valores_cero = {theta: 0 for theta in thetas}
    # Sustituir y evaluar numéricamente. chop=True elimina pequeños errores numéricos.
    Jacobian_num_cero = Jacobian.subs(valores_cero).evalf(chop=True)
    print(f"Sustituyendo: {valores_cero}")
    # Convertir a NumPy array para mostrar con formato numérico
    mostrar_jacobiana_resumida(np.array(Jacobian_num_cero).astype(np.float64))
    input("Presiona Enter...")

    # --- Prueba 4: Todos los valores a pi ---
    print("\n--- Prueba con todos los valores a pi ---")
    valores_pi = {theta: sp.pi for theta in thetas}
    Jacobian_num_pi = Jacobian.subs(valores_pi).evalf(chop=True)
    print(f"Sustituyendo: {valores_pi}")
    mostrar_jacobiana_resumida(np.array(Jacobian_num_pi).astype(np.float64))
    input("Presiona Enter...")

    # --- Prueba 5: Todos los valores a pi/2 ---
    print("\n--- Prueba con todos los valores a pi/2 ---")
    valores_pi_half = {theta: sp.pi/2 for theta in thetas}
    Jacobian_num_pi_half = Jacobian.subs(valores_pi_half).evalf(chop=True)
    print(f"Sustituyendo: {valores_pi_half}")
    mostrar_jacobiana_resumida(np.array(Jacobian_num_pi_half).astype(np.float64))
    input("Presiona Enter...")

# Función principal para ejecutar la prueba de Jacobiana
if __name__ == "__main__":
    prueba_jacobiana()