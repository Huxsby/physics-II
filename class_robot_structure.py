"""
robot:
  name: simple_robot
  links:
    - id: 0
      length: 0.080
      type: revolute
      link_orientation: [0, 0, 1]
      joint_coords: [0, 0, 0.103]
      joint_axis: [0, 0, 1]
    - id: 1
      length: 0.210
      type: revolute
      link_orientation: [0, 0, 1]
      joint_coords: [0, 0, 0.080]
      joint_axis: [0, 1, 0]
    - id: 2
      length: 0.0415
      type: prismatic
      link_orientation: [0, 0, 1]
      joint_coords: [0, 0, 0.210]
      joint_axis: [1, 0, 0]

"""

import numpy as np
import yaml

class Robot:
    """ Clase que representa un robot con eslabones y sus propiedades."""
    def __init__(self, name):
        """ Inicializa una nueva instancia de la clase Robot. """
        self.name = name
        self.links = []
        self.ejes_helicoidales = self.get_ejes_helicoidales()

    def __str__(self):
        """ Retorna una representación en cadena del objeto Robot, incluyendo su nombre y los eslabones."""
        return f"Robot '{self.name}' con {len(self.links)} eslabones. \n\t" + "\n\t".join([str(link) for link in self.links])

    def add_link(self, new_link):
        """ Agrega un nuevo eslabón al robot.  Verifica que el nuevo eslabón no tenga un ID duplicado y que su tipo """
        if not isinstance(new_link, Link):
            raise TypeError("El objeto debe ser de tipo Link.")
        elif any(l.id == new_link.id for l in self.links):
            raise ValueError(f"Ya existe un eslabón con el id {new_link.id}.")
        elif new_link.tipo not in ["revolute", "prismatic"]:
            raise ValueError("Tipo de eslabón no válido. Debe ser 'revolute' o 'prismatic'.")

        self.links.append(new_link)

    def get_ejes_helicoidales(self):
        """ Devuelve una lista de los ejes helicoidales de todos los eslabones del robot. """

        links = self.links
        ejes_helicoidales = []
        qs = []
        for i in range(len(links)):
            link = links[i]
            q = link.joint_coords + qs[i-1] if i > 0 else link.joint_coords
            qs.append(q) # This loop calculating qs seems incorrect for standard screw axis definition and is unused below.

        #print(f"qs {len(qs)}: {qs}") # qs se calcula bien

        # Recalculate based on standard screw axis definition in base frame {0} at zero configuration
        ejes_helicoidales = []
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
    def __init__(self, id, length, tipo, orientation, joint_coords, joint_axis):
        """ Inicializa una nueva instancia de la clase Link. """
        self.id = id
        self.length = length
        self.tipo = tipo
        self.orientation = np.array(orientation)
        self.joint_coords = np.array(joint_coords)
        self.joint_axis = np.array(joint_axis)
    
    def obtener_eje_de_giro(self):
        """
        Devuelve el eje de giro del eslabón.  El eje de giro se define como el eje de la articulación
        multiplicado por la longitud del eslabón.  Este método es útil para calcular la posición y
        orientación del eslabón en el espacio.
        """
        eje = self.joint_axis
        eje_abs = np.abs(eje)
        
        print(f"\tEslabón {self.id}:", end=" ")

        if np.array_equal(eje_abs, np.array([1, 0, 0])):
            eje_str = "X"
        elif np.array_equal(eje_abs, np.array([0, 1, 0])):
            eje_str = "Y"
        elif np.array_equal(eje_abs, np.array([0, 0, 1])):
            eje_str = "Z"
        else:
            eje_str = "fuera de los ejes X, Y, Z"

        if (eje > 0).any():
            sentido = "sentido positivo ⭢  y horario ↻"
        elif (eje < 0).any():
            sentido = "sentido negativo ⭠  y antihorario ↺"
        else:
            sentido = "sin sentido"
        
        print(f"Eje de giro: {eje_str} ({sentido})")
        
        return eje*self.length
    
    def __str__(self):
        """ Retorna una representación en cadena del objeto Link, incluyendo su ID, tipo y eje helicoidal."""
        return(f"El Eslabón '{self.id}' ({self.tipo}), coordenadas: {self.joint_coords}, eje: {self.joint_axis}, longitud: {self.length}")

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
        new_link = Link(
            id=l['id'],
            length=l['length'],
            tipo=l['type'],
            orientation=l['link_orientation'],
            joint_coords=l['joint_coords'],
            joint_axis=l['joint_axis']
        )
        robot.add_link(new_link)

    return robot

if __name__ == "__main__":
    robot = cargar_robot_desde_yaml("robot.yaml")
    print("Ejes helicoidales del robot:")
    print(robot.get_ejes_helicoidales())
    print(f"\n{robot}")

    print("\nobtener_eje_de_giro")
    for i in range(len(robot.links)):
        robot.links[i].obtener_eje_de_giro()
