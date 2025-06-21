"""
El paquete 'core' contiene las clases y funciones fundamentales para la estructura y
definición del robot.
"""
from .class_robot_structure import (
    Robot,
    Link,
    Datos,
    cargar_robot_desde_yaml,
    print_ejes_helicoidales,
    limits,
    get_limits_positive,
    get_limits_negative,
    thetas_aleatorias,
    thetas_limite,
    filtrar_configuraciones,
    str_config
)