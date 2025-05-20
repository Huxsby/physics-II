"""
Módulo para la toma y validación de datos específicos para cálculos de física.

Este módulo proporciona la clase `Datos`, diseñada para facilitar la
solicitud de diferentes tipos de datos al usuario (vectores, ejes, ángulos,
coordenadas exponenciales) con validación incorporada y mensajes
personalizables.

Clases Principales:
    Datos: Clase principal para solicitar y almacenar datos validados.

Ejemplo de uso:
    >>> # Solicitar un vector de 3 componentes
    >>> vector_dato = Datos(tipo="vector")
    Ingrese el vector a rotar (separado por comas o espacios en blanco): 1 2 3
    >>> print(vector_dato.obtener_valor())
    [1. 2. 3.]
    >>> print(vector_dato.obtener_tipo())
    vector

    >>> # Solicitar un ángulo en grados (se almacena en radianes)
    >>> angulo_dato = Datos(tipo="angulo", mensaje="Introduce el ángulo deseado (º):")
    Introduce el ángulo deseado (º): 90
    >>> print(angulo_dato.obtener_valor())
    1.5707963267948966

    >>> # Solicitar un eje
    >>> eje_dato = Datos(tipo="eje")
    Ingrese el eje de rotación (x, y o z): y
    >>> print(eje_dato)
    y
"""

import numpy as np
from class_robot_structure import limits, Robot

class Datos:
    """
    Clase para organizar la toma de datos con el formato deseado.
    """
    def __init__(self, tipo, mensaje=None, robot=None):
        """
        Constructor de la clase Datos.

        Parámetros:
        - tipo (str): Tipo de dato a solicitar ("vector", "eje", "angulo", "cordenadas_exponenciales", "configuración").
        - mensaje (str, opcional): Mensaje personalizado para solicitar el dato.
        - robot (Robot, opcional): Instancia de Robot para validar configuraciones.
        """
        # Mensajes predeterminados según el tipo
        if mensaje is None:
            if tipo == "vector":
                mensaje = "Ingrese el vector a rotar (separado por comas o espacios en blanco): "
            elif tipo == "eje":
                mensaje = "Ingrese el eje de rotación (x, y o z): "
            elif tipo == "angulo":
                mensaje = "Ingrese el ángulo de rotación (º): "
            elif tipo == "cordenadas_exponenciales":
                mensaje = "Ingrese las coordenadas exponenciales (separadas por comas o espacios): "
            elif tipo == "configuración":
                mensaje = "Ingrese los valores de las articulaciones (separados por comas o espacios): "
            else:
                mensaje = "Ingrese el dato: "
        
        self.tipo = tipo
        self.mensaje = mensaje
        self.robot = robot
        self.valor = self.tomar_dato()

    def __str__(self):
        """ Método para representar el objeto como una cadena. """
        return str(self.valor)
    
    def __iter__(self):
        """ Método para permitir la iteración sobre el objeto. """
        return iter(self.valor)
    
    def __getitem__(self, key):
        """ Método para permitir el acceso a los elementos del objeto como si fuera una lista. """
        return self.valor[key]
    
    def __len__(self):
        """ Método para permitir el uso de len() en el objeto. """
        return len(self.valor)
    
    def obtener_valor(self):
        """ Devuelve el valor almacenado. """
        return self.valor
    
    def obtener_tipo(self):
        """ Devuelve el tipo de dato. """
        return self.tipo

    def tomar_dato(self):
        """ Toma el dato según el tipo especificado. """
        if self.tipo == "vector":
            while True:
                try:
                    input_vector = input(self.mensaje)
                    vector = np.array([float(x) for x in input_vector.replace(',', ' ').split() if x.strip()])
                    if vector.shape != (3,):
                        raise ValueError("El vector debe tener 3 componentes.")
                    return vector
                except ValueError as e:
                    print(f"Error: {e}. Por favor, ingrese un vector válido (ej: 1,2,3 o 1 2 3).")
        
        elif self.tipo == "eje":
            while True:
                input_eje = input(self.mensaje).lower()
                if input_eje in ["x", "y", "z"]:
                    return input_eje
                else:
                    print("Eje no válido. Debe ser x, y o z.")
        
        elif self.tipo == "angulo":
            while True:
                try:
                    input_θ = np.deg2rad(float(input(self.mensaje)))
                    return input_θ
                except ValueError:
                    print("Ángulo no válido. Debe ser un número.")
        
        elif self.tipo == "cordenadas_exponenciales":
            while True:
                try:
                    input_vector = input(self.mensaje)
                    vector = np.array([float(x) for x in input_vector.replace(',', ' ').split() if x.strip()])
                    if vector.shape != (6,):
                        raise ValueError("El vector debe tener 6 componentes.")
                    return vector
                except ValueError as e:
                    print(f"Error: {e}. Por favor, ingrese un vector válido (ej: 1,2,3,4,5,6 o 1 2 3 4 5 6).")
        
        elif self.tipo == "configuración":
            if self.robot is None:
                raise ValueError("Para el tipo 'configuración' debe proporcionar el argumento 'robot'.")
            while True:
                try:
                    input_thetas = input(self.mensaje)
                    thetas = np.array([float(x) for x in input_thetas.replace(',', ' ').split()])
                    if len(thetas) != len(self.robot.links):
                        print(f"Error: Debe ingresar {len(self.robot.links)} valores (uno por cada articulación)")
                        continue
                    valido, msg = limits(self.robot, thetas)
                    if not valido:
                        print(f"Configuración inválida: {msg}")
                        continue
                    return thetas
                except ValueError:
                    print("Error: Ingrese solo valores numéricos separados por comas o espacios")
        
        else:
            print("Tipo de dato no válido.")
            return None
