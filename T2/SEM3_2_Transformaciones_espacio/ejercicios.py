""" Matrices de transformación homogénea y rotaciones en 3D."""
"""
 - Importar las funciones en el archivo SEM3_1_Rotaciones.py (ejercicios.py)
 - 
"""

import numpy as np                              # Para cálculos numéricos
from class_datos import Datos                   # Clase para organizar la toma de datos
from class_rotaciones import *                  # Funciones de rotación
from class_matrices import *                    # Funciones de matrices

# Comparar rotaciones
def comparar_rotaciones(w , θ):
    w = np.array(w) / np.linalg.norm(w) # Normalizar el vector
    R1 = RotGen(w , θ)
    R2 = RotRodrigues(w , θ)
    diferencia = np.linalg.norm(R1 - R2)
    return R1 , R2 , diferencia

# Validación de rotaciones
def validar_rotaciones():
    """
    Función para validar rotaciones usando diferentes métodos:
    1. Rodrigues vs Rot
    2. LogRot para recuperar ángulo y eje de rotación
    3. Rotaciones inversas
    4. Casos predefinidos con diferentes ejes y ángulos
    """
    print("\n" + "="*80)
    print("VALIDACIÓN DETALLADA DE ROTACIONES")
    print("="*80)
    
    # Casos de prueba predefinidos
    casos_prueba = [
        # [vector, eje, ángulo]
        [[1, 0, 0], [1, 0, 0], np.pi/2],      # 90° rotación alrededor de eje X
        [[0, 1, 0], [0, 1, 0], np.pi/4],      # 45° rotación alrededor de eje Y
        [[0, 0, 1], [0, 0, 1], np.pi/3],      # 60° rotación alrededor de eje Z
        [[1, 1, 1], [1, 1, 1], np.pi/6],      # 30° rotación alrededor de eje genérico diagonal
        [[1, 2, 3], [0, 1, 0], np.pi/2]       # 90° rotación alrededor de eje Y
    ]
    
    for i, (vector, eje, angulo) in enumerate(casos_prueba, 1):
        print(f"\n{'='*50}")
        print(f"CASO {i}:")
        print(f"{'='*50}")
        
        # Información inicial
        print(f"Vector original: {vector}")
        print(f"Eje de rotación: {eje}")
        print(f"Ángulo de rotación: {np.degrees(angulo):.2f}°")
        
        # Normalizar eje de rotación
        eje_norm = np.array(eje) / np.linalg.norm(eje)
        print(f"Eje de rotación normalizado: {eje_norm}")
        
        # Método 1: Comparar Rot vs RotRodrigues
        print("\n1. Comparación de métodos de rotación:")
        R1 = RotGen(eje_norm, angulo)
        R2 = RotRodrigues(eje_norm, angulo)
        
        print("\nMatriz de rotación (Método Explícito - Rot):")
        imprimir_matriz(R1, "R1")
        
        print("\nMatriz de rotación (Método Rodrigues - RotRodrigues):")
        imprimir_matriz(R2, "R2")
        
        diff_metodos = np.linalg.norm(R1 - R2)
        print(f"Diferencia entre matrices de rotación: {diff_metodos:.2e}")
        
        # Método 2: Rotar vector y verificar
        print("\n2. Rotación de vector:")
        vector_np = np.array(vector)
        vector_rotado1 = np.dot(R1, vector_np)
        vector_rotado2 = np.dot(R2, vector_np)
        
        print(f"Vector original:      {vector_np}")
        print(f"Vector rotado (R1):   {vector_rotado1}")
        print(f"Vector rotado (R2):   {vector_rotado2}")
        
        diff_vectores = np.linalg.norm(vector_rotado1 - vector_rotado2)
        print(f"Diferencia entre vectores rotados: {diff_vectores:.2e}")
        
        # Método 3: Recuperar ángulo con LogRot
        print("\n3. Recuperación de logaritmo de rotación:")
        angulo_recuperado, eje_recuperado = LogRot(R1)
        
        print(f"Ángulo original:     {np.degrees(angulo):.2f}°")
        print(f"Ángulo recuperado:   {np.degrees(angulo_recuperado):.2f}°")
        
        print(f"\nVector original):  {eje_norm}")
        print(f"Vector recuperado: {eje_recuperado}")
        
        # Método 4: Aplicar rotación inversa para verificar simetría
        print("\n4. Verificación de rotación inversa:")
        R_inversa = R1.T  # Matriz transpuesta = inversa en matrices de rotación
        vector_recuperado = np.dot(R_inversa, vector_rotado1)
        
        print(f"Vector original:      {vector_np}")
        print(f"Vector rotado:        {vector_rotado1}")
        print(f"Vector recuperado:    {vector_recuperado}")
        
        diff_recuperado = np.linalg.norm(vector_np - vector_recuperado)
        print(f"Diferencia al recuperar vector original: {diff_recuperado:.2e}")
        
        print("\n5. Verificación de propiedades de rotación:")
        # Verificar propiedades de matrices de rotación
        print("Determinante de R1:  {:.2f}".format(np.linalg.det(R1)))
        print("Transpuesta de R1 == Inversa de R1: {}".format(np.allclose(R1.T, np.linalg.inv(R1))))
        
        print("\n" + "-"*50)
        
        # Criterios de validación
        assert diff_metodos < 1e-10, f"Error: Diferencia significativa entre métodos de rotación en Caso {i}"
        assert diff_vectores < 1e-10, f"Error: Diferencia significativa en vectores rotados en Caso {i}"
        assert np.abs(angulo - angulo_recuperado) < 1e-10, f"Error: Ángulo no recuperado correctamente en Caso {i}"
        assert diff_recuperado < 1e-10, f"Error: Vector no recuperado correctamente en Caso {i}"
    
    print("\n" + "="*50)
    print("VALIDACIÓN COMPLETA: Todos los casos pasaron las pruebas.")
    print("="*50)

# Menú interactivo
def menu():
    """Menú interactivo para seleccionar acciones."""
    while True:
        print("\n" + "="*90)
        print(" "*30 + "MENÚ DE OPCIONES" + " "*30)
        print("="*90)
        print("Nota. Los vectores que se tomen como ejes serán convertidos a unitarios automáticamente.")
        print("="*90)
        # Añadir content
        print("1. Rotar un vector entorno a un eje específico (x,y,z).")    
        print("2. Rotar un vector entorno a un eje genérico.")
        print("3. Comparar rotaciones con fórmula generar vs Rodrigues.")
        print("4. Visualizar rotación de un vector entorno a un eje específico.")
        print("5. Aplicar logaritmo de una matriz de rotación.")
        print("6. Validar rotaciones y funciones (casos predefinidos).")
        print("0. Salir.")
        print("-"*90)

        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1" or opcion == "2":              # 1. y 2. Rotar un vector
            vector = Datos(tipo="vector").valor
            if opcion == "1":                           # 1. Rotar entorno a un eje específico
                eje = Datos(tipo="eje").valor
            else:                                       # 2. Rotar entorno a un eje genérico
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            angulo = Datos(tipo="angulo").valor
            
            # Rotar el vector
            vector_rotado = RotarVector(vector, eje, angulo)  
            print(f"\nVector original: {vector}")
            print(f"Vector rotado: {vector_rotado}")
            continue
        
        elif opcion == "3":                             # 3. Comparar rotaciones
            R1 , R2 , diff = comparar_rotaciones(Datos(tipo="vector").valor, Datos(tipo="angulo").valor)
            imprimir_matriz(R1 , "R (Definición Explícita)")
            imprimir_matriz(R2 , "R (Rodrigues)")
            print("Diferencia entre métodos:", round(diff , 4))
            
        elif opcion == "4":                             # 4. Visualizar rotación
            vector = Datos(tipo="vector").valor
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            if eje_input in ["x", "y", "z"]:
                eje = eje_input
            else:
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            
            Visualizar_Rotacion(vector, eje)
            print("Visualización completa.")

        elif opcion == "5":                             # 5. Calcular logaritmo de una matriz de rotación
            # Obtener matriz de rotación para cálculo del logaritmo
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            
            # Convertir eje de tipo string a vector unitario o normalizar eje genérico
            if eje_input == "x":
                eje = np.array([1, 0, 0])
            elif eje_input == "y":
                eje = np.array([0, 1, 0])
            elif eje_input == "z":
                eje = np.array([0, 0, 1])
            else:
                # Obtener vector del usuario y normalizarlo
                eje = np.array(Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor)
                eje = eje / np.linalg.norm(eje)  # Normalizar a vector unitario
            
            angulo = Datos(tipo="angulo").valor
            R = RotRodrigues(eje, angulo)
            
            # Calcular logaritmo de la matriz de rotación
            angulo_result, eje_resultado = LogRot(R)
            
            print(f"\nÁngulo original (rads): {round(angulo, 3)}")
            print(f"Ángulo recuperado (rads): {round(angulo_result, 3)}")
            print(f"Eje de rotación original: {eje}")
            print(f"Eje de rotación recuperado: {eje_resultado}")                                 

        elif opcion == "6":                             # 6. Validación del sistema de calculo
            validar_rotaciones()

        elif opcion == "0":
            print("Saliendo...")
            break
        
        else:
            print("Opción no válida, intente nuevamente.")

if __name__ == "__main__":
    menu()
