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
        return [link.obtener_eje_helicoidal() for link in self.links]

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
        return(f"El Eslabón '{self.id}' ({self.tipo}), eje helicoidal: {self.obtener_eje_helicoidal()}, coordenadas: {self.joint_coords}, eje: {self.joint_axis}, longitud: {self.length}")

    def obtener_eje_helicoidal(self):
        """
        Calcula y devuelve el eje helicoidal del eslabón.  El eje helicoidal se calcula de manera diferente
        dependiendo del tipo de articulación (revoluta o prismática). Para articulaciones revolutas, el eje
        helicoidal se define como la combinación del eje de rotación y un vector de traslación perpendicular
        a este eje. Para articulaciones prismáticas, el eje helicoidal se define como un vector de traslación
        en la dirección del eje de la articulación.
        Returns:
            numpy.ndarray: El eje helicoidal del eslabón, representado como un vector de 6 elementos
                            (omega, v), donde omega es la componente de rotación y v es la componente
                            de traslación.
        Raises:
            ValueError: Si el tipo de articulación no es "revolute" ni "prismatic".
        """
        w = self.joint_axis
        q = self.joint_coords
        if self.tipo == "revolute":
            v = -np.cross(w, q)     # Vector de traslación perpendicular al eje de rotación
        elif self.tipo == "prismatic":
            v = self.joint_axis * self.length  # Vector de traslación en la dirección del eje de la articulación
        else:
            raise ValueError(f"Tipo desconocido: {self.tipo}")
        return np.hstack((w, v))

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
    for eje in robot.get_ejes_helicoidales():
        print("\t", eje)
    print(f"\n{robot}")

    print("\nobtener_eje_de_giro")
    for i in range(len(robot.links)):
        robot.links[i].obtener_eje_de_giro()
