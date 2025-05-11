import sympy as sp
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import combinations

from class_robot_structure import Robot, cargar_robot_desde_yaml, thetas_aleatorias, limits

""" Funciones de calculo simbólico de la Jacobiana"""

# Función que convierte un eje de rotación en matriz antisimétrica 3x3 (so3)
def VecToso3(w): return sp.Matrix([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

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

""" Funciones de cálculo de la Jacobiana """

def calcular_jacobiana(robot: Robot):
    """ Calcula la Jacobiana simbólica de un robot dado. """
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
    for i in range(0,len(robot.links),1):
        wmat = VecToso3(w[i])
        R.append(sp.eye(3)+sp.sin(thetas_s[i])*wmat+(1-sp.cos(thetas_s[i]))*(wmat*wmat))

    # Aplicamos rotaciones a los vectores q y w para llevarlos a la configuración deseada
    qs = []; ws = []; Ri = R[0]
    qs.append(sp.Matrix(q[0]))
    ws.append(sp.Matrix(w[0]))
    for i in range(1,len(robot.links),1):
        ws.append(Ri*sp.Matrix(w[i]))
        qs.append(Ri*sp.Matrix(q[i])+qs[i-1])
        Ri = Ri*R[i]

    # Calculamos las velocidades lineales, los vectores giro correspondientes y la matriz Jacobiana
    vs = []; Ji = []    # Ji equivale a Si (cada eje helicoidal)
    i = 0
    vs.append(qs[i].cross(ws[i]))
    Ji.append(ws[i].row_insert(3,vs[i]))
    Jacobian = Ji[0]
    for i in range(1,len(robot.links),1):
        vs.append(qs[i].cross(ws[i]))
        Ji.append(ws[i].row_insert(3,vs[i]))
        Jacobian = Jacobian.col_insert(i,Ji[i])
    print(f"\t\033[92mTiempo de cálculo de la Jacobiana del robot {robot.name}: {time.time() - tiempo:.4f} segundos\033[0m")
    return Jacobian, thetas_s

def find_singular_configurations(jacobian: sp.Matrix, substitutions: dict):
    """
    Calcula las configuraciones singulares de una Jacobiana simbólica
    basándose en el determinante de dos submatrices específicas:
    1. Jacobiana[primeras 6 filas, primeras 6 columnas]
    2. Jacobiana[primeras 6 filas, primeras 5 columnas + última columna]
    """
    tiempo_total_start = time.time()
    J_sub = jacobian.subs(substitutions)
    
    # Ordenar free_vars por nombre para consistencia, útil si las soluciones se comparan/muestran
    free_vars = sorted([s for s in jacobian.free_symbols if s not in substitutions], key=lambda x: str(x.name))

    all_solutions = []

    # Verificar si la Jacobiana tiene suficientes filas
    if J_sub.rows < 6:
        print(f"\t\033[91mLa Jacobiana sustituida tiene {J_sub.rows} filas, se necesitan al menos 6 para este análisis.\033[0m")
        print(f"\t\033[92mTiempo total de procesamiento: {time.time() - tiempo_total_start:.4f}s\033[0m")
        return []

    # --- Determinante 1: Primeras 6 filas, primeras 6 columnas ---
    print(f"\n\t--- Analizando Determinante 1 (submatriz 6x6 de las primeras 6 columnas) ---")
    tiempo_det1_start = time.time()
    if J_sub.cols >= 6:
        try:
            sub_matrix1 = J_sub[:6, :6]
            det1 = sub_matrix1.det()
            # print(f"\tExpresión del Determinante 1: {det1}") # Descomentar para depuración
            solutions1 = sp.solve(det1, free_vars, dict=True)
            
            if not isinstance(solutions1, list): # sp.solve puede devolver un solo dict
                solutions1 = [solutions1] if solutions1 else []
            
            # Filtrar soluciones que no sean diccionarios (ej. True/False para ecuaciones triviales)
            solutions1 = [s for s in solutions1 if isinstance(s, dict)]

            all_solutions.extend(solutions1)
            msg_color = "\033[92m" if solutions1 else "\033[96m"
            print(f"\t{msg_color}Soluciones para Det1 ({len(solutions1)} encontradas): {solutions1 if solutions1 else 'Ninguna'}. Tiempo: {time.time() - tiempo_det1_start:.4f}s\033[0m")
        except Exception as e:
            print(f"\t\033[91mError calculando/resolviendo Det1: {e}. Tiempo: {time.time() - tiempo_det1_start:.4f}s\033[0m")
    else:
        print(f"\t\033[93mOmitiendo Det1: J_sub tiene {J_sub.cols} columnas, se necesitan al menos 6. Tiempo: {time.time() - tiempo_det1_start:.4f}s\033[0m")

    # --- Determinante 2: Primeras 6 filas, primeras 5 columnas + última columna ---
    print(f"\n\t--- Analizando Determinante 2 (submatriz 6x6 de las primeras 5 columnas y la última) ---")
    tiempo_det2_start = time.time()
    if J_sub.cols >= 6: # Se necesitan al menos 6 columnas para formar esta submatriz 6x6
        col_indices_det2 = list(range(5))  # Columnas 0, 1, 2, 3, 4
        last_col_idx = J_sub.cols - 1
        col_indices_det2.append(last_col_idx)
        # col_indices_det2 es ahora [0,1,2,3,4, J_sub.cols-1]
        # Esta lista tiene 6 elementos distintos porque J_sub.cols-1 >= 5 (dado que J_sub.cols >= 6)

        if J_sub.cols == 6:
            # En este caso, col_indices_det2 será [0,1,2,3,4,5], idéntica a la submatriz de Det1
            print(f"\t\033[96mNota: Para la forma de J_sub {J_sub.shape}, Det2 es idéntico a Det1 (columnas: {col_indices_det2}). Se re-evaluará como solicitado.\033[0m")
        else:
            print(f"\t\033[96mFormando Det2 con columnas: {col_indices_det2}.\033[0m")
            
        try:
            sub_matrix2 = J_sub.extract(list(range(6)), col_indices_det2)
            det2 = sub_matrix2.det()
            # print(f"\tExpresión del Determinante 2: {det2}") # Descomentar para depuración
            solutions2 = sp.solve(det2, free_vars, dict=True)

            if not isinstance(solutions2, list):
                solutions2 = [solutions2] if solutions2 else []
            
            solutions2 = [s for s in solutions2 if isinstance(s, dict)]

            all_solutions.extend(solutions2)
            msg_color = "\033[92m" if solutions2 else "\033[96m"
            print(f"\t{msg_color}Soluciones para Det2 ({len(solutions2)} encontradas): {solutions2 if solutions2 else 'Ninguna'}. Tiempo: {time.time() - tiempo_det2_start:.4f}s\033[0m")
        except Exception as e:
            print(f"\t\033[91mError calculando/resolviendo Det2: {e}. Tiempo: {time.time() - tiempo_det2_start:.4f}s\033[0m")
    else: # J_sub.cols < 6
        print(f"\t\033[93mOmitiendo Det2: J_sub tiene {J_sub.cols} columnas, se necesitan al menos 6. Tiempo: {time.time() - tiempo_det2_start:.4f}s\033[0m")

    # Eliminar soluciones duplicadas
    unique_sols_tuples = set()
    final_unique_solutions = []
    if all_solutions:
        for sol_dict in all_solutions:
            if isinstance(sol_dict, dict):
                # Crear una representación canónica (tupla ordenada de items) para la unicidad
                sol_tuple = tuple(sorted(sol_dict.items(), key=lambda item: str(item[0])))
                if sol_tuple not in unique_sols_tuples:
                    unique_sols_tuples.add(sol_tuple)
                    final_unique_solutions.append(sol_dict)
    
    print(f"\n\tSe encontraron {len(final_unique_solutions)} configuraciones singulares únicas de las evaluadas.")
    print(f"\t\033[92mTiempo total de procesamiento para find_singular_configurations: {time.time() - tiempo_total_start:.4f}s\033[0m")
    return final_unique_solutions

def mostrar_jacobiana_resumida(Jacobian: sp.Matrix, msg="", max_chars=20):
    """ Muestra la matriz Jacobiana de forma resumida, limitando el número de caracteres por elemento. """
    print(f"{msg}", sep="")
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

""" Funciones de calculo de elipsoides de manipulabilidad y fuerza"""

def elipsoide_manipulabilidad(J, articulaciones_idx=[1, 2], puntos=100):
    """ Calcula el elipsoide de manipulabilidad a partir de la Jacobiana. """
    J_num = np.array(J).astype(np.float64)      # Convertir la matriz Jacobiana simbólica (SymPy) a un array NumPy para operaciones numéricas.
    u = np.linspace(0, np.pi / 2, puntos)       # Generar un vector de ángulos 'u' desde 0 hasta pi/2, con 'puntos' divisiones. Se usa para parametrizar un cuarto de círculo unitario.
    x = np.cos(u)                               # Calcular las coordenadas X del cuarto de círculo unitario.
    y = np.sin(u)                               # Calcular las coordenadas Y del cuarto de círculo unitario.
    xx = np.concatenate([x, -x, -x, x])         # Extender las coordenadas X para formar un círculo completo (reflejando en los ejes).
    yy = np.concatenate([y, y, -y, -y])         # Extender las coordenadas Y para formar un círculo completo (reflejando en los ejes).
    
    giro = []                                   # Inicializar una lista para almacenar los vectores de velocidad del efector final resultantes.
    for i in range(len(xx)):                    # Iterar sobre cada punto (xx[i], yy[i]) del círculo unitario.
        vjoints = np.zeros(J_num.shape[1])      # Crear un vector de velocidades articulares (vjoints) inicializado a ceros, con la misma longitud que el número de columnas de J (número de articulaciones).
        vjoints[articulaciones_idx[0]] = xx[i]  # Asignar las coordenadas del círculo unitario (xx[i], yy[i]) a las velocidades de las articulaciones especificadas por 'articulaciones_idx'.
        vjoints[articulaciones_idx[1]] = yy[i]  # Esto simula aplicar una velocidad unitaria distribuida entre estas dos articulaciones.
        giro.append(J_num @ vjoints)            # Calcular la velocidad resultante del efector final usando la relación v_efector = J * v_articulaciones.
    giro = np.array(giro)                       # Convertir la lista de vectores de velocidad del efector final a un array NumPy.
    return xx, yy, giro                         # Devolver las coordenadas del círculo unitario (xx, yy) y los puntos del elipsoide de manipulabilidad (giro).

# def elipsoide_fuerza(J, articulaciones_idx=[1, 2], puntos=100):
#     """ Calcula el elipsoide de fuerza a partir de la Jacobiana. """
#     J_num = np.array(J).astype(np.float64)      # Convertir la matriz Jacobiana simbólica (SymPy) a un array NumPy para operaciones numéricas
#     J_T_inv = np.linalg.inv(J_num.T)            # Calcular la inversa de la transpuesta de la Jacobiana numérica. Esta matriz relaciona los pares articulares con las fuerzas/momentos en el efector final.
#     u = np.linspace(0, np.pi / 2, puntos)       # Generar un vector de ángulos 'u' desde 0 hasta pi/2, con 'puntos' divisiones. Se usa para parametrizar un cuarto de círculo unitario.
#     x = np.cos(u)                               # Calcular las coordenadas X del cuarto de círculo unitario.
#     y = np.sin(u)                               # Calcular las coordenadas Y del cuarto de círculo unitario.
#     xx = np.concatenate([x, -x, -x, x])         # Extender las coordenadas X para formar un círculo completo (reflejando en los ejes).
#     yy = np.concatenate([y, y, -y, -y])         # Extender las coordenadas Y para formar un círculo completo (reflejando en los ejes).

#     llave = []                                  # Inicializar una lista para almacenar los vectores de fuerza/momento resultantes (llave de torsión).
#     for i in range(len(xx)):                    # Iterar sobre cada punto (xx[i], yy[i]) del círculo unitario.
#         tau = np.zeros(J.shape[1])              # Crear un vector de pares articulares (tau) inicializado a ceros, con la misma longitud que el número de columnas de J (número de articulaciones).
#         tau[articulaciones_idx[0]] = xx[i]      # Asignar las coordenadas del círculo unitario (xx[i], yy[i]) a los pares de las articulaciones especificadas por 'articulaciones_idx'.
#         tau[articulaciones_idx[1]] = yy[i]      # Esto simula aplicar un par unitario distribuido entre estas dos articulaciones.
#         llave.append(J_T_inv @ tau)             # Calcular la llave de torsión (fuerza/momento) resultante en el efector final usando la relación F = (J^T)^-1 * tau.
#     llave = np.array(llave)                     # Convertir la lista de vectores de llave de torsión a un array NumPy.
#     return xx, yy, llave                        # Devolver las coordenadas del círculo unitario (xx, yy) y los puntos del elipsoide de fuerza (llave).

def elipsoide_fuerza(J, articulaciones_idx=[1, 2], puntos=100):
    """
    Calcula el elipsoide de fuerza a partir de la Jacobiana.
    Es retrocompatible: usa la inversa si J^T es cuadrada e invertible,
    y la pseudo-inversa en el caso general (robots redundantes o no cuadradas).
    """
    J_num = np.array(J).astype(np.float64)
    JT = J_num.T

    # Intentamos usar la inversa clásica si es posible
    try:
        if JT.shape[0] == JT.shape[1]:
            J_T_inv = np.linalg.inv(JT)
        else:
            raise np.linalg.LinAlgError("No cuadrada")  # Forzar pseudo-inversa
    except np.linalg.LinAlgError:
        # Usamos pseudo-inversa si no es cuadrada o no es invertible
        J_T_inv = np.linalg.pinv(JT)

    # Generar círculo unitario
    u = np.linspace(0, np.pi / 2, puntos)
    x = np.cos(u)
    y = np.sin(u)
    xx = np.concatenate([x, -x, -x, x])
    yy = np.concatenate([y, y, -y, -y])

    llave = []
    for i in range(len(xx)):
        tau = np.zeros(J_num.shape[1])
        tau[articulaciones_idx[0]] = xx[i]
        tau[articulaciones_idx[1]] = yy[i]
        llave.append(J_T_inv @ tau)

    return xx, yy, np.array(llave)


def graficar_elipsoides(xx, yy, giro, llave, name=None, indices=(1, 3), limitplot=8, point_size=2):
    """
    Función para graficar los elipsoides de manipulabilidad y fuerza. 
    
    Nota: En la práctica 2 del curso 2024-25 la gráfica tiene el eje X invertido.
    """
    plt.scatter(xx, yy, label=r'${\dot\theta}$ ó $\tau$', s=point_size)
    plt.scatter(giro[:, indices[0]], giro[:, indices[1]], label="manipulabilidad", s=point_size)
    plt.scatter(llave[:, indices[0]], llave[:, indices[1]], label="fuerza", s=point_size)
    plt.ylim(top=limitplot, bottom=-limitplot)
    plt.xlim(left=-limitplot, right=limitplot)
    plt.legend(loc='upper right', fontsize='large')
    plt.grid(True)
    plt.title(f"Elipsoides de Manipulabilidad y Fuerza de {name}")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.show()

def calcular_volumen_elipsoides(J):
    """ Calcula el volumen de los elipsoides de manipulabilidad y fuerza a partir de la Jacobiana. """
    vol_EM = vol_EF =  (J*sp.Transpose(J)).det() # Volumen del elipsoide de manipulabilidad
    return vol_EM, vol_EF

""" Funciones de validación """

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
    subs1 = {thetas_s[2]:0, thetas_s[3]:0, thetas_s[4]:0, thetas_s[5]:0}
    sol1 = find_singular_configurations(Jacobian, subs1)
    # Resultado esperado: [{t1: -1.57079632679490}, {t1: 1.57079632679490}]

    # Restricciones para el segundo caso
    subs2 = {thetas_s[1]:0, thetas_s[3]:0, thetas_s[4]:0, thetas_s[5]:0}
    sol2 = find_singular_configurations(Jacobian, subs2)
    # Resultado esperado: [{t2: -1.70541733137745},
                # {t2: -1.57079632679490},
                # {t2: 1.43617532221234},
                # {t2: 1.57079632679490}]

    print("\nConfiguraciones singulares:")
    print(f"Caso 1 (t2=t3=t4=0): {sol1}")
    print(f"Caso 2 (t1=t3=t4=0): {sol2}")
    return sol1, sol2

def prueba_elipsoides(sol1, sol2):
    """ Función de prueba para calcular y graficar los elipsoides de manipulabilidad y fuerza. """
    robot = cargar_robot_desde_yaml("robot.yaml")
    J_sym, thetas_s = calcular_jacobiana(robot)

    random_config, thetas_dic_random = thetas_aleatorias(robot)

    Jal = J_sym.subs(thetas_dic_random)
    Jp0 = J_sym.subs({thetas_s[0]:0, thetas_s[1]:0, thetas_s[2]:0, thetas_s[3]:0, thetas_s[4]:0, thetas_s[5]:0})

    # Primero obtenemos la matriz Jacobiana. Tomando la configuración cero del robot:
    # A partir de la Jacobiana podemos calcular los elipsoides de manipulabilidad y fuerza en 2 dimensiones, si
    # restringimos las velocidades de las articulaciones a sólo 2 grados de libertad:

    vol_EM, vol_EF = calcular_volumen_elipsoides(Jal)
    print(f"\n\033[93m--- Elipsoides de una configuración aleatoria (validada) ({random_config})---\033[0m")
    # mostrar_jacobiana_resumida(Jal)
    print(f"\tVolumen del elipsoide de manipulabilidad y fuerza: {vol_EM}") # Volumen del elipsoide de fuerza y manipulabilidad son el mismo (volumen de la matriz J*J^T)
    print("\tGraficando elipsoides... configuración aleatoria")
    xx, yy, giro = elipsoide_manipulabilidad(Jal)
    _, _, llave = elipsoide_fuerza(Jal)
    graficar_elipsoides(xx, yy, giro, llave, name=random_config)

    input("\nPresiona Enter para continuar con la búsqueda de configuraciones válidas...")
    # Buscar el primer caso válido en sol1 o sol2
    valid_config = None
    msg = ""

    print(f"\n\033[93m--- Buscando configuraciones válidas (Probaremos a gráficar la primera que sea compatible con {robot.name}) ---\033[0m")
    # Caso 1: t2=t3=t4=0
    for config in sol1:
        complete_config = {theta: 0 for theta in thetas_s}  # Inicializar todas las thetas en 0
        complete_config.update(config)  # Actualizar con la solución actual de sol1
        print(f"\tProbando configuración completa en \033[35mCaso 1: \033[36m{complete_config}\033[0m")
        valid, msg = limits(robot, complete_config)
        if valid:
            valid_config = complete_config
            msg = f"Configuración válida encontrada en \033[35mCaso 1: \033[32m{valid_config}\033[0m {msg}"
            break
        else:
            print(f"\tConfiguración inválida en \033[35mCaso 1: \033[36m{complete_config}\033[0m {msg}")

    # Caso 2: t1=t3=t4=0
    if not valid_config:  # Solo buscar en el caso 2 si no se encontró una configuración válida en el caso 1
        for config in sol2:
            complete_config = {theta: 0 for theta in thetas_s}  # Inicializar todas las thetas en 0
            complete_config.update(config)  # Actualizar con la solución actual de sol2
            print(f"\tProbando configuración completa en \033[35mCaso 2: \033[36m{complete_config}\033[0m")
            valid, msg = limits(robot, complete_config)
            if valid:
                valid_config = complete_config
                msg = f"Configuración válida encontrada en \033[35mCaso 2: \033[32m{valid_config}\033[0m {msg}"
                break
            else:
                print(f"\tConfiguración inválida en \033[35mCaso 2: \033[36m{complete_config}\033[0m {msg}")

    if valid_config:
        print(msg)
        Jal = J_sym.subs(valid_config)
        vol_EM, vol_EF = calcular_volumen_elipsoides(Jal)
        print(f"\n--- Volúmenes de los elipsoides ({valid_config}) ---")
        print(f"\tVolumen del elipsoide de manipulabilidad y fuerza: {vol_EM}") # Volumen del elipsoide de fuerza y manipulabilidad son el mismo (volumen de la matriz J*J^T)
        print(f"\tGraficando elipsoides... configuración singular válida para {robot.name}")
        xx, yy, giro = elipsoide_manipulabilidad(Jal)
        _, _, llave = elipsoide_fuerza(Jal)
        graficar_elipsoides(xx, yy, giro, llave, name=valid_config)
    else:
        print("\tNo se encontró una configuración válida en los casos analizados.")

    print("\n\033[93m--- Elipsoides configuración nula (θs=0) ---\033[0m")
    print(f"\tVolumen del elipsoide de manipulabilidad y fuerza: {vol_EM}") # Volumen del elipsoide de fuerza y manipulabilidad son el mismo (volumen de la matriz J*J^T)
    print(f"\tGraficando elipsoides... configuración θs=0")
    # mostrar_jacobiana_resumida(Jp0)
    xx, yy, giro = elipsoide_manipulabilidad(Jp0)
    _, _, llave = elipsoide_fuerza(Jp0)
    graficar_elipsoides(xx, yy, giro, llave, name="(θs=0)")
    print("\n\n\033[93m--- Fin de la prueba ---\033[0m")

if __name__ == "__main__":
    sol1, sol2 = prueba_jacobiana()
    prueba_elipsoides(sol1, sol2)