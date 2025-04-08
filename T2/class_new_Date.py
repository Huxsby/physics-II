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

def yaml_fun(archivo="robot.yaml"):
    import yaml
    
    try:
        with open(archivo, 'r') as file:
            robotdata = yaml.safe_load(file)
    except FileNotFoundError: 
        print(f"El archivo '{archivo}' no se encontró.")
        return

    robotname = robotdata['robot']['name']
    eslabones = robotdata['robot']['links']

    print("Nombre del robot = ", robotname)

    for eslabon in eslabones:
        print("eslabón ID = ", eslabon['id'])
        print("tipo de eslabón = ", eslabon['type'])
        print("coordenadas articulación = ", eslabon['joint_coords'])
        print("Eje rotación (R) o expansión (P) = ", eslabon['joint_axis'])
    #print(robotdata['robot']['links'][0]['type'])

yaml_fun()