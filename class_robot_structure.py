"""
class_robot_structure.py
=====
Este módulo proporciona clases y funciones para definir la estructura de un robot manipulador,
incluyendo sus eslabones y articulaciones, y para cargar esta estructura desde un archivo
de configuración YAML. También incluye funciones para manejar los límites de las articulaciones
y generar configuraciones aleatorias dentro de estos límites.

Clases:
    Robot: Representa un robot manipulador compuesto por varios eslabones.
    Link: Representa un único eslabón del robot con las propiedades de su articulación asociada.

Funciones:
    cargar_robot_desde_yaml: Carga la estructura de un robot desde un archivo YAML especificado.
    limits: Verifica si una configuración de articulaciones dada está dentro de los límites definidos para el robot.
    get_limits_positive: Devuelve los límites superiores de las articulaciones del robot.
    get_limits_negative: Devuelve los límites inferiores de las articulaciones del robot.
    thetas_aleatorias: Genera una configuración de articulaciones aleatoria dentro de los límites del robot.
    thetas_limite: Ajusta una configuración de articulaciones para que se encuentre dentro de los límites del robot,
                   recortando los valores que exceden los límites. Si show=True: Anuncia por la terminal si se realizan ajustes.


Ejemplo:
    Suponiendo que existe un archivo 'robot.yaml' con la siguiente estructura:

    ```yaml
    robot:
      name: simple_robot
      links:
        - id: 0
          length: 0.080
          type: revolute
          link_orientation: [0, 0, 1]
          joint_coords: [0, 0, 0.103]
          joint_axis: [0, 0, 1]
          joint_limits: "(-2.96, 2.96)"
        - id: 1
          length: 0.210
          type: revolute
          link_orientation: [0, 0, 1]
          joint_coords: [0, 0, 0.080]
          joint_axis: [0, 1, 0]
          joint_limits: "(-1.57, 1.57)"
        - id: 2
          length: 0.0415
          type: prismatic
          link_orientation: [0, 0, 1]
          joint_coords: [0, 0, 0.210]
          joint_axis: [1, 0, 0]
          joint_limits: "(0.0, 0.1)" # Ejemplo de límites para prismático
    ```

    Puedes cargar e inspeccionar el robot de esta manera:

    >>> import numpy as np
    >>> from class_robot_structure import Robot, Link, cargar_robot_desde_yaml, limits, thetas_aleatorias
    >>> # Crear un robot.yaml ficticio para el ejemplo
    >>> yaml_content = '''
    ... robot:
    ...   name: example_robot
    ...   links:
    ...     - id: link1
    ...       length: 1.0
    ...       type: revolute
    ...       link_orientation: [0, 0, 1]
    ...       joint_coords: [0, 0, 0]
    ...       joint_axis: [0, 0, 1]
    ...       joint_limits: "(-3.14, 3.14)"
    ...     - id: link2
    ...       length: 0.5
    ...       type: prismatic
    ...       link_orientation: [1, 0, 0]
    ...       joint_coords: [1, 0, 0]
    ...       joint_axis: [1, 0, 0]
    ...       joint_limits: "(0.0, 1.0)"
    ... '''
    >>> with open("robot_example.yaml", "w") as f:
    ...     f.write(yaml_content)
    >>> robot = cargar_robot_desde_yaml("robot_example.yaml") # doctest: +SKIP
    >>> print(robot) # doctest: +SKIP
    >>> print(robot.ejes_helicoidales) # doctest: +SKIP
    >>> print(robot.limits_dict) # doctest: +SKIP
    >>> config_valida, msg = limits(robot, [0.5, 0.2]) # doctest: +SKIP
    >>> print(f"Configuración válida: {config_valida}, Mensaje: {msg}") # doctest: +SKIP
    >>> config_invalida, msg = limits(robot, [4.0, 0.2]) # doctest: +SKIP
    >>> print(f"Configuración válida: {config_invalida}, Mensaje: {msg}") # doctest: +SKIP
    >>> random_thetas, random_thetas_dict = thetas_aleatorias(robot) # doctest: +SKIP
    >>> print(f"Thetas aleatorias: {random_thetas}") # doctest: +SKIP
    >>> print(f"Thetas aleatorias (dict): {random_thetas_dict}") # doctest: +SKIP
"""

import numpy as np
import time
import yaml

MARGEN_LIMITES_THETAS = 5 # Grados de margen para los límites de las articulaciones y evitar colisiones

class Robot:
    """ Clase que representa un robot con eslabones y sus propiedades."""
    def __init__(self, name: str):
        """ Inicializa una nueva instancia de la clase Robot. """
        self.name = name
        self.links = []
        self.ejes_helicoidales = "Los ejes helicoidales se calcularán al crear el robot."
        self.limits_dict = None # Inicializa limits_dict como None, se puede establecer más tarde
        self.num_links = 0
        print(f"\033[92\tmRobot '{self.name}' creado.\033[0m")

    def __str__(self):
        """ Retorna una representación en cadena del objeto Robot, incluyendo su nombre y los eslabones."""
        return f"\nRobot '{self.name}' con {self.num_links} eslabones. \n\t" + "\n\t".join([str(link) for link in self.links])

    def __len__(self):
        """ Retorna la cantidad de eslabones en el robot. """
        return self.num_links

    def add_link(self, new_link):
        """ Agrega un nuevo eslabón al robot.  Verifica que el nuevo eslabón no tenga un ID duplicado y que su tipo """
        if not isinstance(new_link, Link):
            raise TypeError("El objeto debe ser de tipo Link.")
        elif any(l.id == new_link.id for l in self.links):
            raise ValueError(f"Ya existe un eslabón con el id {new_link.id}.")
        elif new_link.tipo not in ["revolute", "prismatic"]:
            raise ValueError("Tipo de eslabón no válido. Debe ser 'revolute' o 'prismatic'.")

        self.links.append(new_link)
        self.num_links += 1

    def get_ejes_helicoidales(self):
        """ Devuelve una lista de los ejes helicoidales de todos los eslabones del robot. """
        tiempo_inicio = time.time() # Para medir el tiempo de ejecución
        links = self.links
        ejes_helicoidales = []
        qs = []
        for i in range(len(links)):
            link = links[i]
            q = link.joint_coords + qs[i-1] if i > 0 else link.joint_coords
            qs.append(q) # This loop calculating qs seems incorrect for standard screw axis definition and is unused below.

        #print(f"qs {len(qs)}: {qs}") # qs se calcula bien

        # Recalculate based on standard screw axis definition in base frame {0} at zero configuration
        # ejes_helicoidales = []
        for i, link in enumerate(self.links):
            w = np.array(link.joint_axis)
            q = qs[i] # Assuming q is a point on the joint axis in {0}

            if link.tipo == "revolute":
                # Assuming w is a unit vector representing the axis direction
                w_norm = np.linalg.norm(w)
                if np.isclose(w_norm, 0):
                     raise ValueError(f"Eje de articulación cero para eslabón revoluto {link.id}")
                w = w / w_norm # Ensure w is unit vector
                v = -np.cross(w, q) # Standard definition: v = -w x q
            elif link.tipo == "prismatic":
                # Assuming joint_axis gives the direction of translation v
                v_norm = np.linalg.norm(w) # Use w (joint_axis) to calculate norm for v
                if np.isclose(v_norm, 0):
                     raise ValueError(f"Eje de articulación cero para eslabón prismático {link.id}")
                v = w / v_norm # v is the unit vector in the direction of translation
                w = np.zeros(3) # w is zero for prismatic joints
            else:
                raise ValueError(f"Tipo desconocido: {link.tipo} para el eslabón {link.id}")

            ejes_helicoidales.append(np.hstack((w, v)))
        print(f"\t\033[92mTiempo de ejecución de get_ejes_helicoidales: {time.time() - tiempo_inicio:.4f} segundos\033[0m")
        return ejes_helicoidales

class Link:
    """
    Clase que representa un eslabón de un robot manipulador.
    Un eslabón se define por su identificador único, longitud, tipo de articulación (revoluta o prismática),
    orientación, coordenadas de la articulación y el eje de la articulación.  Esta clase proporciona
    métodos para acceder a las propiedades del eslabón y calcular su eje helicoidal, que es fundamental
    para la cinemática exponencial.
    Attributes:
        id (str): Identificador único del eslabón.
        length (float): Longitud del eslabón.
        tipo (str): Tipo de articulación asociada al eslabón ("revolute" o "prismatic").
        orientation (numpy.ndarray): Orientación del eslabón representada como un vector.
        joint_coords (numpy.ndarray): Coordenadas de la articulación del eslabón en el espacio.
        joint_axis (numpy.ndarray): Eje de la articulación del eslabón.
    """
    def __init__(self, id, length, tipo, orientation, joint_coords, joint_axis, joint_limits):
        """ Inicializa una nueva instancia de la clase Link. """
        self.id = id
        self.length = length
        self.tipo = tipo
        self.orientation = np.array(orientation)
        self.joint_coords = np.array(joint_coords)
        self.joint_axis = np.array(joint_axis)
        self.joint_limits = joint_limits
        
    def __str__(self):
        """ Retorna una representación en cadena del objeto Link, incluyendo su ID, tipo y eje helicoidal."""
        return(f"El Eslabón '{self.id}' ({self.tipo}), coordenadas: {self.joint_coords}, eje: {self.joint_axis}, longitud: {self.length} y límites: {self.joint_limits}")
    
    def obtener_eje_de_giro(self):
        """
        Devuelve el eje de giro del eslabón.  El eje de giro se define como el eje de la articulación multiplicado
        por la longitud del eslabón.  Este método es útil para calcular la posición y orientación del eslabón en el espacio.
        Pero sobretodo, para verificar que los ejes de giro son correctos.
        """
        eje = self.joint_axis
        eje_abs = np.abs(eje)
        
        print(f"\t{f'Eslabón {self.id}:':<20}", end="") # Ajusta el ancho (e.g., 20) según sea necesario

        if (eje > 0).any():
            signo = "\033[92m+"
        elif (eje < 0).any():
            signo = "\033[91m-"
        else:
            signo = " "

        if np.array_equal(eje_abs, np.array([1, 0, 0])):
            eje_str = "X"
        elif np.array_equal(eje_abs, np.array([0, 1, 0])):
            eje_str = "Y"
        elif np.array_equal(eje_abs, np.array([0, 0, 1])):
            eje_str = "Z"
        else:
            eje_str = "fuera de los ejes X, Y, Z"

        if (eje > 0).any():
            sentido = "\033[92msentido positivo ⭢  == horario ↻"
        elif (eje < 0).any():
            sentido = "\033[91msentido negativo ⭠  == antihorario ↺"
        else:
            sentido = "sin sentido"
        
        print(f"Eje de giro: {signo}{eje_str}\033[0m \t( {sentido} \033[0m)")
        
        return eje*self.length

""" Funciones auxiliares """

def cargar_robot_desde_yaml(path="robot.yaml"):
    """
    Carga la configuración del robot desde un archivo YAML.

    Args:
        path (str, optional): La ruta al archivo YAML que contiene la configuración del robot.
            Por defecto es "robot.yaml".

    Returns:
        Robot: Un objeto Robot si la configuración se carga exitosamente, None si el archivo no se encuentra.

    Raises:
        FileNotFoundError: Si el archivo especificado en 'path' no existe.

    Example:
    ```python
    robot = cargar_robot_desde_yaml(path='mi_robot.yaml')
    if robot:
        print(f"Robot cargado: {robot.name}")
    else:
        print("No se pudo cargar el robot.")
    ```
    """
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"\033[91mEl archivo {path} no existe.\033[0m")
        return None
    
    robot_data = data['robot']
    robot = Robot(robot_data['name'])

    for l in robot_data['links']:
        joint_limits_str = l.get('joint_limits', None)
        if joint_limits_str:
            # Eliminar paréntesis y dividir por la coma
            parts = joint_limits_str.strip('()').split(',')
            # Convertir a flotantes
            initial_min_limit = float(parts[0].strip())
            initial_max_limit = float(parts[1].strip())

            # Definir el margen
            #margen = np.deg2rad(MARGEN_LIMITES_THETAS)  # Convertir a radianes
            margen = 0

            # Aplicar el margen para reducir el rango de operación
            # El límite inferior se incrementa y el límite superior se decrementa
            adjusted_min_limit = initial_min_limit + margen
            adjusted_max_limit = initial_max_limit - margen
            
            # Asegurarse de que el límite inferior no supere al superior después del ajuste.
            # Si el margen es tan grande que los límites se cruzan (o el rango original es muy pequeño),
            # se establece un rango de un solo punto en el centro del rango original.
            # Esto significa que la articulación efectivamente no tiene rango de movimiento
            # después de aplicar el margen.
            if adjusted_min_limit > adjusted_max_limit:
                mid_point = (initial_min_limit + initial_max_limit) / 2.0
                joint_limits = (mid_point, mid_point)
                # Opcionalmente, se podría imprimir una advertencia:
                # print(f"Advertencia: Para el eslabón {l.get('id', 'desconocido')}, el margen de {margen:.3f} rad "
                #       f"es demasiado grande para el rango original ({initial_min_limit:.3f}, {initial_max_limit:.3f}). "
                #       f"Límites ajustados a ({mid_point:.3f}, {mid_point:.3f}).")
            else:
                joint_limits = (adjusted_min_limit, adjusted_max_limit)
        else:
            joint_limits = None

        new_link = Link(
            id=l['id'],
            length=l['length'],
            tipo=l['type'],
            orientation=l['link_orientation'],
            joint_coords=l['joint_coords'],
            joint_axis=l['joint_axis'],
            joint_limits=joint_limits
        )
        robot.add_link(new_link)
        
    robot.ejes_helicoidales = robot.get_ejes_helicoidales() # Guardar los ejes helicoidales en el robot
    robot.limits_dict = {f'joint_{i+1}': link.joint_limits for i, link in enumerate(robot.links) if link.joint_limits is not None}
    return robot

def print_ejes_helicoidales(robot: Robot):
    """
    Imprime los ejes helicoidales de cada eslabón del robot.
    
    Args:
        robot (Robot): El robot cuyas articulaciones se van a imprimir.
    """

    print("\nEjes helicoidales del robot:")
    for i in range(robot.num_links):
        eje_helicoidal = robot.ejes_helicoidales[i]
        array_str = np.array2string(
            eje_helicoidal,
            suppress_small=True,
            separator=', ',
            formatter={'float_kind': lambda x: f"{x:10.4f}"}
        )
        print(f"\tS{i+1}: {array_str}")

def limits(robot: Robot, θs):
    """
    Function to check if joint angles are within the robot's defined limits.
    
    Args:
        robot (Robot): The robot with defined joint limits.
        θ (list or dict): The joint angles to check.
        
    Returns:
        tuple: (bool, str) - Boolean indicating if all joints are within limits and a message.
    """
    if robot.limits_dict is None or len(robot.limits_dict) == 0:
        return True, "No limits defined for the robot."
    
    # If θ is a dictionary, extract its values as a list
    if isinstance(θs, dict):
        θs = list(θs.values())
    
    # Check if each joint angle is within its limits
    for i in range(len(θs)):
        current_value = θs[i]
        joint_key = f'joint_{i+1}'
        
        if joint_key not in robot.limits_dict:
            print(f"Warning: Limits for {joint_key} not found in robot.limits_dict.")
            return False, f"Limits for {joint_key} not defined."

        lower_limit = robot.limits_dict[joint_key][0]
        upper_limit = robot.limits_dict[joint_key][1]

        # Diagnostic print to show values for each joint before the check
        # print(f"DEBUG: Joint {i+1}: Value={current_value}, Limits=({lower_limit}, {upper_limit})")
        # print(f"DEBUG: Checking condition: {current_value} < {lower_limit} (is {current_value < lower_limit}) OR {current_value} > {upper_limit} (is {current_value > upper_limit})")

        if current_value < lower_limit or current_value > upper_limit:
            # This print executes if the joint is out of limits
            # print(f"Details for out-of-limit Joint {i+1}: Value={current_value}, LowerLimit={lower_limit}, UpperLimit={upper_limit}")
            # print(f"Comparison results: Value < LowerLimit is {current_value < lower_limit}, Value > UpperLimit is {current_value > upper_limit}")
            return False, f"Joint {i+1} out of limits: {current_value} not within range of ({lower_limit}, {upper_limit})"
    
    return True, "All joints within limits"

def get_limits_positive(robot: Robot):
    """ Devuelve los límites positivos de las articulaciones del robot. """
    positive_limits = [robot.limits_dict[f'joint_{i+1}'][1] for i in range(robot.num_links)]
    return np.array(positive_limits)

def get_limits_negative(robot: Robot):
    """ Devuelve los límites negativos de las articulaciones del robot. """
    negative_limits = [robot.limits_dict[f'joint_{i+1}'][0] for i in range(robot.num_links)]
    return np.array(negative_limits)

def thetas_aleatorias(robot: Robot):
    """
    robot (Robot): El robot con los límites definidos.
    Returns:
    tuple: Una tupla que contiene:
        - numpy.ndarray: Un array con los ángulos aleatorios generados.
        - dict: Un diccionario donde las claves son los nombres de las articulaciones ('t0', 't1', etc.)
          y los valores son los ángulos correspondientes.
    Notas:
    - Si el robot no tiene límites definidos (robot.limits_dict es None o vacío), 
      se generarán ángulos aleatorios entre -2π y 2π.
    - Si el robot tiene límites definidos, la función seguirá generando configuraciones 
      aleatorias hasta encontrar una que cumpla con todos los límites.
    - La función utiliza las funciones auxiliares get_limits_negative, get_limits_positive y limits
      para validar que la configuración generada sea válida.
    """
    if robot.limits_dict is None or len(robot.limits_dict) == 0:
        random_config = np.zeros(robot.num_links)
        for i in range(robot.num_links):
            random_config[i] = np.random.uniform(-2*np.pi, 2*np.pi)
        return random_config, {f"t{i}": random_config[i] for i in range(robot.num_links)}
    
    negative_limits = get_limits_negative(robot)
    positive_limits = get_limits_positive(robot)
    # print(f"negative_limits: {negative_limits}")
    # print(f"positive_limits: {positive_limits}")
    while True:
        random_config = np.zeros(len(negative_limits))
        for i in range(len(negative_limits)):
            random_config[i] = np.random.uniform(negative_limits[i], positive_limits[i])
        # Validar la configuración generada
        valid, msg = limits(robot, random_config)
        if valid:
            return random_config, {f"t{i}": random_config[i] for i in range(robot.num_links)}
        else:
            print(f"Configuración random {str_config(random_config, 2)} inválida debido a: {msg}. Intentando nuevamente.")

def thetas_limite(robot: Robot, thetas, show=False):
    """
    Limita los ángulos de las articulaciones del robot a sus límites definidos.

    Args:
        robot (Robot): El robot con los límites definidos.
        thetas (list or dict): Los ángulos de las articulaciones a limitar.
        show (bool): Si es True, muestra mensajes cuando se ajustan los valores.
        
    Returns:
        list: Los ángulos de las articulaciones limitados.
    """
    if robot.limits_dict is None or len(robot.limits_dict) == 0:
        return thetas

    # Si thetas es un diccionario, extraer sus valores como una lista
    if isinstance(thetas, dict):
        thetas = list(thetas.values())

    # Limitar cada ángulo a su rango definido
    for i in range(len(thetas)):
        joint_key = f'joint_{i+1}'
        if joint_key in robot.limits_dict:
            lower_limit = robot.limits_dict[joint_key][0]
            upper_limit = robot.limits_dict[joint_key][1]
            
            if thetas[i] < lower_limit:
                if show: print(f"El valor de joint {joint_key} ({thetas[i]}) está por debajo del límite inferior ({lower_limit}). Ajustando a {lower_limit}.")
                thetas[i] = lower_limit
            elif thetas[i] > upper_limit:
                if show: print(f"El valor de joint {joint_key} ({thetas[i]}) está por encima del límite superior ({upper_limit}). Ajustando a {upper_limit}.")
                thetas[i] = upper_limit
                
    return thetas

def filtrar_configuraciones(robot: Robot, configuraciones):
    """
    Filtra las configuraciones de un robot para asegurarse de que estén dentro de los límites definidos.
    """
    print("\nFiltrando configuraciones:")
    configuraciones_validas = []
    for config in configuraciones:
        if limits(robot, config)[0]:
            configuraciones_validas.append(config)
            print(f"\t\033[92mConfiguración válida: {str_config(config, 2)}\033[0m")
        else:
            print(f"\t\033[91mConfiguración inválida: {str_config(config, 2)}\033[0m")
    return configuraciones_validas

def str_config(config, decimales=6):
    """
    Devuelve un string bonito para imprimir configuraciones numéricas con elementos uniformemente espaciados.
    
    Args:
        config: La configuración numérica (array, lista, etc.)
        decimales (int): Número de decimales a mostrar
        
    Returns:
        str: String formateado con elementos uniformemente espaciados
    """
    try:
        arr = np.round(np.array(config, dtype=float), decimales)
        
        # Handle empty array
        if len(arr) == 0:
            return "[]"
        
        # Format each element with fixed decimal places
        formatted = [f"{val:.{decimales}f}" for val in arr]
        
        # Find maximum width of any formatted number
        max_width = max(len(s) for s in formatted)
        
        # Pad each element to have the same width with one space on each side
        aligned = [f" {s.rjust(max_width)} " for s in formatted]
        
        # Join the aligned elements
        return "[" + ", ".join(aligned) + "]"
    except (ValueError, TypeError):
        # Return the original if conversion fails
        return str(config)


# Ejemplo de uso
if __name__ == "__main__":
    robot = cargar_robot_desde_yaml("robot.yaml")
    print(f"\n{robot}")
    print(f"\nLímites de las articulaciones: {robot.limits_dict}")
    print_ejes_helicoidales(robot)
    print("\nObtener_eje_de_giro")
    for i in range(robot.num_links):
        robot.links[i].obtener_eje_de_giro()
