"""
El paquete 'simulation' contiene todas las funciones y clases relacionadas
con la visualización 3D y la animación de los robots.
"""
from .class_robot_plotter import (
    plot_robot,
    graficar_workspace,
    guardar_animacion,
    graficar_limites,
    importar_trayectoria_cartesian,
    exportar_trayectoria_cartesian,
    importar_trayectoria_angular,
    exportar_trayectoria_angular,
)