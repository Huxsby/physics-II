import sympy as sp
import numpy as np
import time

from class_robot_structure import Robot

def VecToso3sp(v):
    v = sp.Matrix(v)
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def MatrixExp3sp(w, theta):
    w = sp.Matrix(w).normalized()
    omgmat = VecToso3sp(w)
    return sp.eye(3) + sp.sin(theta) * omgmat + (1 - sp.cos(theta)) * (omgmat * omgmat)

def MatrixExp6sp(S, theta):
    w = sp.Matrix(S[0:3])
    v = sp.Matrix(S[3:6])
    omgmat = VecToso3sp(w)

    if w.norm() < 1e-5:
        return sp.eye(3).row_join(v * theta).col_join(sp.Matrix([[0, 0, 0, 1]]))
    else:
        R = MatrixExp3sp(w, theta)
        G_theta = sp.eye(3) * theta + (1 - sp.cos(theta)) * omgmat + (theta - sp.sin(theta)) * (omgmat * omgmat)
        p = G_theta * v
        return R.row_join(p).col_join(sp.Matrix([[0, 0, 0, 1]]))

def Adjunta(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_skew = VecToso3sp(p)

    upper = R.row_join(sp.zeros(3))
    lower = (p_skew * R).row_join(R)
    return upper.col_join(lower)

def calcular_jacobiana(robot: Robot):
    """
    Calcula la matriz Jacobiana simbólica del robot proporcionado.

    Parámetros:
    - robot: objeto Robot cargado (de tu clase Robot)

    Devuelve:
    - Jacobian: matriz simbólica 6xn de la Jacobiana
    - thetas: variables simbólicas correspondientes a las articulaciones
    """
    time_i = time.time()  # Iniciar temporizador
    # Obtener ejes helicoidales
    screw_axes = robot.get_ejes_helicoidales()
    n = len(screw_axes)

    # Crear variables simbólicas
    thetas = sp.symbols(f't0:{n}')  # t0, t1, ..., tn-1

    # Inicializar transformaciones
    Ts = []
    T = sp.eye(4)

    for i in range(n):
        S = screw_axes[i]
        T_i = MatrixExp6sp(S, thetas[i])
        T = T * T_i
        Ts.append(T)

    # Calcular columnas de la Jacobiana
    J_cols = []
    T_prev = sp.eye(4)

    for i in range(n):
        Ad = Adjunta(T_prev)
        S = sp.Matrix(screw_axes[i])
        J_cols.append(Ad * S)
        T_prev = Ts[i]

    Jacobian = sp.Matrix.hstack(*J_cols)

    print(f"\t\033[92mTiempo de cálculo de la Jacobiana del robot {robot.name}: {time.time() - time_i:.4f} segundos\033[0m")
    return Jacobian, thetas

def evaluar_jacobiana(Jacobian, thetas, valores):
    """
    Evalúa la matriz Jacobiana simbólica con los valores numéricos proporcionados para las articulaciones.

    Parámetros:
    - Jacobian: matriz simbólica de la Jacobiana (sympy.Matrix).
    - thetas: lista o tupla de variables simbólicas (sympy.Symbol) correspondientes a las articulaciones.
    - valores: lista, tupla o np.array de valores numéricos para las articulaciones,
               en el mismo orden que las variables en `thetas`.

    Devuelve:
    - Jacobian_num: matriz numérica de la Jacobiana evaluada (sympy.Matrix con valores flotantes),
                    o None si ocurre un error.
    """
    try:
        n_thetas = len(thetas)
        n_valores = len(valores)

        if n_valores > n_thetas:
            raise ValueError(f"Se proporcionaron más valores ({n_valores}) que variables theta ({n_thetas}).")

        # Crear el diccionario para la sustitución, rellenando con 0 si faltan valores
        valores_dict = {}
        for i in range(n_thetas):
            if i < n_valores:
                valores_dict[thetas[i]] = valores[i]
            else:
                # Rellenar con 0 los valores faltantes
                valores_dict[thetas[i]] = 0.0
        
        print(f"Valores para la Jacobiana: {valores_dict}")
        # Sustituir y evaluar numéricamente
        # chop=True ayuda a eliminar pequeños errores numéricos (ej. convertir 1e-15 a 0)
        Jacobian_num = Jacobian.subs(valores_dict).evalf(chop=True)
        return Jacobian_num
    
    except Exception as e:
        print(f"Error al evaluar la Jacobiana: {e}")
        print(f"  Thetas: {thetas}")
        print(f"  Valores proporcionados: {valores}")
        return None

def mostrar_jacobiana_resumida(Jacobian, max_chars=30):
    """ Muestra la matriz Jacobiana de forma resumida, limitando el número de caracteres por elemento. """
    rows, cols = Jacobian.shape

    # Primero, convertir todo a texto corto
    matrix_text = []
    for i in range(rows):
        row = []
        for j in range(cols):
            elem = str(Jacobian[i, j])
            if len(elem) > max_chars:
                elem = elem[:max_chars] + "..."
            row.append(elem)
        matrix_text.append(row)

    # Calcular el ancho máximo de cada columna
    col_widths = [max(len(matrix_text[i][j]) for i in range(rows)) for j in range(cols)]

    # Imprimir con bordes
    print()
    for i in range(rows):
        formatted_row = []
        for j in range(cols):
            elem = matrix_text[i][j]
            formatted_row.append(elem.ljust(col_widths[j]))  # Alinear a la izquierda

        row_text = "  ".join(formatted_row)  # Separador entre columnas
        if i == 0:
            print(f"⎡ {row_text} ⎤")
        elif i == rows - 1:
            print(f"⎣ {row_text} ⎦")
        else:
            print(f"⎢ {row_text} ⎥")

def prueba_jacobiana():
    """
    Función de prueba para calcular y mostrar la Jacobiana de un robot.
    """

    from class_robot_structure import cargar_robot_desde_yaml

    # Carga del robot
    robot = cargar_robot_desde_yaml("robot.yaml")

    # Calcular Jacobiana
    Jacobian, thetas = calcular_jacobiana(robot)

    # Mostrar Jacobiana
    #sp.pprint(Jacobian)

    # Mostrar Jacobiana de forma resumida
    print("\n--- Jacobiana ---")
    mostrar_jacobiana_resumida(Jacobian)

    # --- Prueba 1: Valores específicos ---
    print("\n--- Prueba con valores específicos ---")
    n_thetas = len(thetas)
    # Asegúrate de que los índices coincidan con el número de thetas
    valores_especificos = {
        thetas[0]: 0.5,
        thetas[1]: 0.3,
        thetas[2]: -0.1,
        # Añade o quita según n_thetas si es necesario
        # thetas[3]: 0.2,
        # thetas[4]: 0.4,
        # thetas[5]: -0.3
    }
    # Rellenar con 0 si hay más thetas de los especificados
    for i in range(len(valores_especificos), n_thetas):
        if thetas[i] not in valores_especificos:
            valores_especificos[thetas[i]] = 0.0

    print(f"Valores para la Jacobiana: {valores_especificos}")

    try:
        Jacobian_num1 = Jacobian.subs(valores_especificos)
        """
        .subs() sustituye las variables simbólicas en la matriz Jacobiana por los valores numéricos proporcionados.
        En caso de que no haya suficientes valores en valores_especificos, el resto de las variables se mantendrán simbólicas.
        Por ejemplo, si tienes 6 variables y solo proporcionas 3 valores, las otras 3 seguirán siendo simbólicas.
        """
        mostrar_jacobiana_resumida(Jacobian_num1) # Mostrar simbólicamente
        input()
        print("Evaluando numéricamente...")
        mostrar_jacobiana_resumida(Jacobian_num1.evalf(chop=True)) # Evaluar numéricamente
        """
        .evalf() convierte expresiones simbólicas (como sqrt(2) o pi/2) a sus aproximaciones decimales.
        Si tu matriz Jacobian_num1 ya contenía solo números decimales después de la sustitución, llamar a .evalf() no cambiará su representación numérica de forma apreciable.
        Verías una diferencia si Jacobian_num1 contuviera expresiones simbólicas que necesitan ser evaluadas numéricamente.
        """
        input()
        Jacobian_num1_1 = evaluar_jacobiana(Jacobian, thetas, [0.5, 0.3, -0.1])
        mostrar_jacobiana_resumida(Jacobian_num1_1) # Mostrar simbólicamente
        input()

    except Exception as e:
        print(f"Error al sustituir valores específicos: {e}")
        print("Valores intentados:", valores_especificos)

    # --- Prueba 2: Todos los valores a cero ---
    print("\n--- Prueba con todos los valores a cero ---")
    valores_cero = {theta: 0 for theta in thetas}
    try:
        Jacobian_num2 = Jacobian.subs(valores_cero)
        mostrar_jacobiana_resumida(Jacobian_num2.evalf()) # Evaluar numéricamente
        input()

    except Exception as e:
        print(f"Error al sustituir valores cero: {e}")

    # --- Prueba 3: Todos los valores a pi ---
    print("\n--- Prueba con todos los valores a pi ---")
    valores_pi = {theta: np.pi for theta in thetas}
    try:
        Jacobian_num3 = Jacobian.subs(valores_pi)
        # Usar evalf() para obtener valores numéricos aproximados de expresiones como sin(pi), cos(pi)
        mostrar_jacobiana_resumida(Jacobian_num3.evalf(chop=True)) # chop=True para eliminar pequeños errores numéricos
        input()

    except Exception as e:
        print(f"Error al sustituir valores pi: {e}")

if __name__ == "__main__":
    prueba_jacobiana()