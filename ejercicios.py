"""
Programa principal para probar los codigos de rotaciones y cinemática inversa y funciones auxiliares para las prácticas.
"""

#import sympy as sp                                 # Para cálculos simbólicos
import numpy as np                                  # Para cálculos numéricos
import os                                           # (Limpiar pantalla)
from class_datos import Datos                       # Clase para organizar la toma de datos
from class_rotaciones import *                      # Funciones de rotación
from class_helicoidales import *                    # Funciones de helicoidales y su menu()
import class_robot_structure as robot_structure     # Clase y funciones para leer el archivo robot.yaml
from class_jacobian import *                        # Funciones para calcular la Jacobiana
from class_robot_plotter_prueba import menu_plotter # Funcion de mmenu() de las funciones para graficar el robot, casos de prueba y animaciones.
from problema_cinematico_inverso_gen import menu_cinematica_inversa # Funcion de mmenu() de cinemática inversa, casos de prueba y animaciones.

# Menú interactivo
def menu_principal():
    def limpiar_pantalla(stop=True):
        """Limpia la pantalla de la consola."""
        if stop: input("\033[93mPresione Enter para continuar...\033[0m")
        os.system('cls' if os.name == 'nt' else 'clear')

    """Menú interactivo para seleccionar acciones."""
    while True:
        print("\n" + "="*90)    # Separador
        print(" "*37 + "MENÚ DE OPCIONES")
        print("="*90)   # Separador
        print(" Notas:\n - Por defecto el robot cargado será robot.yaml.\n - Los vectores que se tomen como ejes serán convertidos a unitarios automáticamente.")
        print("="*90)   # Separador
        print("1. Rotar un vector entorno a un eje específico (x,y,z).")    
        print("2. Rotar un vector entorno a un eje genérico.")
        print("3. Comparar rotaciones con fórmula generar vs Rodrigues.")
        print("4. Visualizar rotación de un vector entorno a un eje específico.")
        print("5. Aplicar logaritmo de una matriz de rotación.")
        print("6. Validar rotaciones y funciones (casos predefinidos).")
        print("-"*90)   # Separador
        print("7. Pruebas de ejes helicoidales, vectores de 6 elementos y matrices de 4x4.")
        print("-"*90)   # Separador
        print("8. Lectura de archivo YAML (robot.yaml).")
        print("9. Calcular la matriz de transformación homogénea del robot.")
        print("-"*90)   # Separador
        print("10. Calcular la matriz Jacobiana del robot. Singularidades y elipsoides.")
        print("-"*90)   # Separador
        print("11. Comparar configuraciones random.")
        print("12. Múltiples graficaciones y animcaciones para robot.yaml.")
        print("13. Problema cinemático inverso.")
        print("-"*90)   # Separador
        print("0. Salir.")

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
            limpiar_pantalla()
        
        elif opcion == "3":                             # 3. Comparar rotaciones
            w = Datos(tipo="vector", mensaje="Eje de rotación: ").valor; θ = Datos(tipo="angulo").valor
            w = np.array(w) / np.linalg.norm(w)     # Normalizar el vector
            R1 = RotGen(w , θ); R2 = RotRodrigues(w , θ)
            diferencia = np.linalg.norm(R1 - R2)    # Comparar rotaciones
            
            imprimir_matriz(R1 , "R (Definición Explícita)"); imprimir_matriz(R2 , "R (Rodrigues)")
            print("\nDiferencia entre métodos:", round(diferencia , 4))
            limpiar_pantalla()

        elif opcion == "4":                             # 4. Visualizar rotación
            vector = Datos(tipo="vector").valor
            eje_input = input("¿Desea usar un eje cartesiano (x/y/z) o un eje genérico (g)? ").lower()
            if eje_input in ["x", "y", "z"]:
                eje = eje_input
            else:
                eje = Datos(tipo="vector", mensaje="Ingrese el vector de rotación (separado por comas o espacios): ").valor
            
            Visualizar_Rotacion(vector, eje)
            print("Visualización completa.")
            limpiar_pantalla()

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
                u_eje = eje / np.linalg.norm(eje)
                if list(eje) != list(u_eje):
                    print(f"\tEje no unitario, normalizando {eje} -> {u_eje}")  # Normalizar a vector unitario
                    eje = u_eje  # Normalizar a vector unitario
            
            angulo = Datos(tipo="angulo").valor
            R = RotRodrigues(eje, angulo)
            
            # Calcular logaritmo de la matriz de rotación
            angulo_result, eje_resultado = LogRot(R)
            
            print(f"\nÁngulo original (rads): {round(angulo, 3)}")
            print(f"Ángulo recuperado (rads): {round(angulo_result, 3)}")
            print(f"Eje de rotación original: {eje}")
            print(f"Eje de rotación recuperado: {eje_resultado}")                                 
            limpiar_pantalla()

        elif opcion == "6":                             # 6. Validación del sistema de calculo
            validar_rotaciones()
            limpiar_pantalla()

        elif opcion == "7":                             # 7. Pruebas de ejes helicoidales, vectores de 6 elementos y matrices de 4x4
            limpiar_pantalla(stop = False)
            print("Pruebas de ejes helicoidales, vectores de 6 elementos y matrices de 4x4.")
            print("NOTA: Los vectores que se tomen como ejes serán convertidos a unitarios automáticamente.")
            menu_helicoidales()
            # limpiar_pantalla()

        elif opcion == "8":                             # 8. Pruebas de matrices de 4x4
            print("Lectura de archivo YAML (robot.yaml)")
            robot = robot_structure.cargar_robot_desde_yaml("robot.yaml")
            print(robot)
            print_ejes_helicoidales(robot)
            print("\nObtener_eje_de_giro")
            for i in range(robot.num_links):
                robot.links[i].obtener_eje_de_giro()

            limpiar_pantalla()

        elif opcion == "9":                             # 9. Calcular la matriz de transformación homogénea
            print("Calcular la matriz de transformación homogénea del robot.")
            robot = robot_structure.cargar_robot_desde_yaml("robot.yaml")             # Cargar robot y ejes helicoidales
            print_ejes_helicoidales(robot)
            M = calcular_M_generalizado(robot)                                              # Calcular M (posición cero)
            print("Matriz M (posición cero):")
            imprimir_matriz(M, "M")

            # Valores de las articulaciones
            # thetas = [0.7, 0.3, 0.3, 0.4, 0.5, 0.8]
            print("Limites de las articulaciones (theta):\n", robot.limits_dict)
            thetas = Datos(tipo="configuración", robot=robot).valor
            valid, msg = limits(robot, thetas)
            if not valid:
                print(f"Error: {msg}")
                print("Los límites de las articulaciones son: ", robot.limits_dict)
                input("Presione Enter para continuar...")
                continue
            
            # Calcular T
            T = calcular_T_robot(robot.ejes_helicoidales, thetas, M)
            print("\nMatriz de transformación homogénea T:")
            imprimir_matriz(T, "T")
            
            # Descomponer T en R y p, y calcular ángulos de Euler
            R = T[:3,:3]; p = T[:3,3]; RPY = R2Euler(R)
            print("Coordenadas (x,y,z) del TCP:", p)
            print("Los angulos de Euler (Roll Pitch Yaw) son:", RPY,'\n') 
            limpiar_pantalla()

        elif opcion == "10":                            # 10. Calcular la matriz Jacobiana del robot
            print("Calcular la matriz Jacobiana del robot. Singularidades y elipsoides.")
            robot = cargar_robot_desde_yaml("robot.yaml") # Carga del robot
            final_unique_solutions = prueba_jacobiana(robot); limpiar_pantalla()
            prueba_elipsoides(robot, final_unique_solutions); limpiar_pantalla()

        
        elif opcion == "11":                            # 11. Comparar configuraciones random 8 veces
            robot = robot_structure.cargar_robot_desde_yaml("robot.yaml")
            J_sym, thetas_s = calcular_jacobiana(robot)
            print("Comparando configuraciones...")

            # Configuración cero
            zero_config = np.zeros(robot.num_links)
            thetas_dic_zero = {f"t{i}": zero_config[i] for i in range(robot.num_links)}
            J_num_zero = J_sym.subs(thetas_dic_zero).evalf(chop=True)
            vol_EM_zero, vol_EF_zero = calcular_volumen_elipsoides(J_num_zero)
            print(f"Configuración Cero: {zero_config}\tVol EM: {vol_EM_zero:.2e}\tVol EF: {vol_EF_zero:.2e}")

            # Configuración limite positiva
            limit_conf = get_limits_positive(robot)
            thetas_dic_limit = {f"t{i}": limit_conf[i] for i in range(robot.num_links)}
            J_num_limit = J_sym.subs(thetas_dic_limit).evalf(chop=True)
            vol_EM_limit, vol_EF_limit = calcular_volumen_elipsoides(J_num_limit)
            print(f"Configuración Límite Positiva: {str_config(limit_conf, 2)}\tVol EM: {vol_EM_limit:.2e}\tVol EF: {vol_EF_limit:.2e}")

            # Configuración singular propuesta el Niryo One
            singular_config = np.array([0, 0, 1.43617532221234, 0, 0, 0, 0])
            thetas_dic_singular = {f"t{i}": singular_config[i] for i in range(robot.num_links)}
            J_num_singular = J_sym.subs(thetas_dic_singular).evalf(chop=True)
            vol_EM_singular, vol_EF_singular = calcular_volumen_elipsoides(J_num_singular)
            print(f"Configuración Singular Propuesta: {str_config(singular_config, 2)}\tVol EM: {vol_EM_singular:.2e}\tVol EF: {vol_EF_singular:.2e}")

            print("\nComparando configuraciones random 8 veces...")
            for vueltas in range(8):
                random_config, thetas_dic_random = thetas_aleatorias(robot)
                J_num_random = J_sym.subs(thetas_dic_random).evalf(chop=True)
                vol_EM_random, vol_EF_random = calcular_volumen_elipsoides(J_num_random)
                print(f"Random Config {vueltas+1}: {str_config(random_config, 2)}\tVol EM: {vol_EM_random:.2e}\tVol EF: {vol_EF_random:.2e}")
            limpiar_pantalla()
            
        elif opcion == "12":                           # 12. Graficar robot.yaml
            print("Graficando robot.yaml...")
            menu_plotter() #; limpiar_pantalla()
            
        elif opcion == "13":                           # 13. Problema cinemático inverso
            print("Problema cinemático inverso.")
            menu_cinematica_inversa(); limpiar_pantalla()

        elif opcion == "0":                             # 0. Salir
            print("Saliendo...", end=" ")
            limpiar_pantalla()
            break
        
        else:                                           # Opción no válida
            # print("Opción no válida, intente nuevamente.")
            limpiar_pantalla(stop=False)

if __name__ == "__main__":
    menu_principal()
