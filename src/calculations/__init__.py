"""
El paquete 'calculations' agrupa los módulos responsables de la lógica
matemática de la robótica, como la cinemática, jacobianos, etc.

Este __init__.py expone las clases y funciones de cálculo para que puedan
ser utilizadas de forma centralizada.
"""
from .class_helicoidales import (
    calcular_Sθ,
    calcular_exp_Sθ,
    logaritmo_transformacion,
    visualizar_eje_helicoidal,
    validar_transformaciones_helicoidales,
    calcular_T_robot
)
from .class_jacobian import (
    prueba_jacobiana,
    prueba_elipsoides,
    calcular_jacobiana,
    calcular_volumen_elipsoides,
    mostrar_jacobiana_resumida
)
from .class_rotaciones import (
    RotarVector,
    RotGen,
    RotRodrigues,
    Visualizar_Rotacion,
    LogRot,
    validar_rotaciones,
    R2Euler,
    imprimir_matriz,
    Rp2Trans,
    Euler2R,
    antisimetrica
)
from .problema_cinematico_inverso_gen import menu_cinematica_inversa, CinematicaInversa