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

# Función que convierte un eje de rotación en matriz antisim ́etrica 3x3 (so3)
def VecToso3(w): return sp.Matrix([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

def calcular_jacobiana(robot: Robot):
    tiempo = time.time()  # Iniciar temporizador
    w = []; q = []
    for link in robot.links:
        w.append(link.joint_axis)   # Definimos ejes de rotación de las articulaciones en la posición cero del robot
        q.append(link.joint_coords) # Definimos los vectores que van del centro de cada eje al centro del siguiente

    # Coordenadas de las articulaciones
    n = len(w)  # Número de articulaciones
    thetas_s = sp.symbols(f't0:{n}')  # t0, t1, ..., tn-1

    # Calculamos las matrices de rotación a partir de los ejes w, utilizando la fórmula de Rodrigues
    R = []
    for i in range(0,6,1):
        wmat = VecToso3(w[i])
        R.append(sp.eye(3)+sp.sin(thetas_s[i])*wmat+(1-sp.cos(thetas_s[i]))*(wmat*wmat))

    # Aplicamos rotaciones a los vectores q y w para llevarlos a la configuración deseada
    qs = []; ws = []; Ri = R[0]
    qs.append(sp.Matrix(q[0]))
    ws.append(sp.Matrix(w[0]))
    for i in range(1,6,1):
        ws.append(Ri*sp.Matrix(w[i]))
        qs.append(Ri*sp.Matrix(q[i])+qs[i-1])
        Ri = Ri*R[i]

    # Calculamos las velocidades lineales, los vectores giro correspondientes y la matriz Jacobiana
    vs = []; Ji = []    # Ji equivale a Si (cada eje helicoidal)
    i = 0
    vs.append(qs[i].cross(ws[i]))
    Ji.append(ws[i].row_insert(3,vs[i]))
    Jacobian = Ji[0]
    for i in range(1,6,1):
        vs.append(qs[i].cross(ws[i]))
        Ji.append(ws[i].row_insert(3,vs[i]))
        Jacobian = Jacobian.col_insert(i,Ji[i])
    print(f"\t\033[92mTiempo de cálculo de la Jacobiana del robot {robot.name}: {time.time() - tiempo:.4f} segundos\033[0m")
    return Jacobian, thetas_s

def find_singular_configurations(jacobian: sp.Matrix, substitutions: dict):
    """
    Calcula las configuraciones singulares para una Jacobiana dada
    con ciertas restricciones en los ángulos.
    """
    try:
        determinant = jacobian.subs(substitutions).det()
        solutions = sp.solve(determinant)
        return solutions
    except Exception as e:
        print(f"\033[91mError al calcular configuraciones singulares con {substitutions}:\033[0m {e}")
        return None

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
    Jacobian, thetas_s = calcular_jacobiana(robot) # Calcular Jacobiana Simbólica

    # Mostrar Jacobiana simbólica de forma resumida
    print("\n--- Jacobiana Simbólica (Resumida) ---")
    mostrar_jacobiana_resumida(Jacobian)
    input("Presiona Enter para continuar con las evaluaciones numéricas...")

    # --- Prueba 1: Valores parciales (ejemplo: solo t0 varía, el resto 0) ---
    print("\n--- Prueba con valores parciales (t0 variable, resto 0) ---")
    # Crea un diccionario donde todas las thetas excepto la primera (thetas[0]) se ponen a 0
    valores_parciales_t0 = {theta: 0 for i, theta in enumerate(thetas_s) if i != 0}
    Jacobian_parcial_t0 = Jacobian.subs(valores_parciales_t0)
    print(f"Sustituyendo: {valores_parciales_t0}")
    mostrar_jacobiana_resumida(Jacobian_parcial_t0) # Mostrar simbólicamente con t0
    input("Presiona Enter...")

    # --- Prueba 2: Valores parciales (ejemplo: solo t1 varía, el resto 0) ---
    print("\n--- Prueba con valores parciales (t1 variable, resto 0) ---")
    # Crea un diccionario donde todas las thetas excepto la segunda (thetas[1]) se ponen a 0
    valores_parciales_t1 = {theta: 0 for i, theta in enumerate(thetas_s) if i != 1}
    Jacobian_parcial_t1 = Jacobian.subs(valores_parciales_t1)
    print(f"Sustituyendo: {valores_parciales_t1}")
    mostrar_jacobiana_resumida(Jacobian_parcial_t1) # Mostrar simbólicamente con t1
    input("Presiona Enter...")

    # --- Prueba 3: Todos los valores a cero ---
    print("\n--- Prueba con todos los valores a cero ---")
    valores_cero = {theta: 0 for theta in thetas_s}
    # Sustituir y evaluar numéricamente. chop=True elimina pequeños errores numéricos.
    Jacobian_num_cero = Jacobian.subs(valores_cero).evalf(chop=True)
    print(f"Sustituyendo: {valores_cero}")
    # Convertir a NumPy array para mostrar con formato numérico
    mostrar_jacobiana_resumida(np.array(Jacobian_num_cero).astype(np.float64))
    input("Presiona Enter...")

    # --- Prueba 4: Todos los valores a pi ---
    print("\n--- Prueba con todos los valores a pi ---")
    valores_pi = {theta: sp.pi for theta in thetas_s}
    Jacobian_num_pi = Jacobian.subs(valores_pi).evalf(chop=True)
    print(f"Sustituyendo: {valores_pi}")
    mostrar_jacobiana_resumida(np.array(Jacobian_num_pi).astype(np.float64))
    input("Presiona Enter...")

    # --- Prueba 5: Todos los valores a pi/2 ---
    print("\n--- Prueba con todos los valores a pi/2 ---")
    valores_pi_half = {theta: sp.pi/2 for theta in thetas_s}
    Jacobian_num_pi_half = Jacobian.subs(valores_pi_half).evalf(chop=True)
    print(f"Sustituyendo: {valores_pi_half}")
    mostrar_jacobiana_resumida(np.array(Jacobian_num_pi_half).astype(np.float64))
    input("Presiona Enter...")

    # --- Prueba 6: Buscar singularidades ---
    print("\n--- Búsqueda de configuraciones singulares ---")
    
    # Restricciones para el primer caso
    subs1 = {thetas_s[2]:0, thetas_s[3]:0, thetas_s[4]:0}
    sol1 = find_singular_configurations(Jacobian, subs1)
    # Resultado esperado: [{t1: -1.57079632679490}, {t1: 1.57079632679490}]

    # Restricciones para el segundo caso
    subs2 = {thetas_s[1]:0, thetas_s[3]:0, thetas_s[4]:0}
    sol2 = find_singular_configurations(Jacobian, subs2)
    # Resultado esperado: [{t2: -1.70541733137745},
                # {t2: -1.57079632679490},
                # {t2: 1.43617532221234},
                # {t2: 1.57079632679490}]

    print("\nConfiguraciones singulares:")
    print(f"Caso 1 (t2=t3=t4=0): {sol1}")
    print(f"Caso 2 (t1=t3=t4=0): {sol2}")

# Función principal para ejecutar la prueba de Jacobiana
if __name__ == "__main__":
    prueba_jacobiana()