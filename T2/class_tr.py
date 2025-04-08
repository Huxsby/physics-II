import numpy as np
import class_rotaciones as rt

class TransformacionHomogenea:
    """
    Clase para representar y manipular transformaciones homogéneas en 3D.
    """
    def __init__(self, R, S, θ):
        """
        La idea es que esta clase facilite la creación de matrices de transformación homogénea. Y todas usus operaciones atraves de metodos.
        """
        """ Que contiene mas información?
        
        # 1. S [eje helicoidal] = [w, p] (vector de rotación y vector de traslación)
            w = S[:3]  # w Parte angular (vector de rotación) -> ¿Se saca de R?
            v = S[3:]  # v Parte de traslación (vector de posición) -> ¿De donde se optiene?
            Nota. Más adelante usaremos estos programas con archivos .csv y otros para darnos una lista de datos, ¿Cuales son estos datos? ¿De que partimos?
        
        # 2. θ (ángulo de rotación)                    -> Valor que se recupera facilmente [y Necesaro para obtener S] [y Necesaro para obtener R] 
        
        # 3. eje (eje de rotación)                       -> Dato que se recupera facilmente de R
        
        # 4. p (vector de traslación)                  -> Dato util para calcular T [y Necesaro para obtener S]
       
        # 4. R (matriz de rotación)                    -> Dato util para calcular T y Dato que se recupera facilmente

        # 0. matriz (matriz de transformación homogénea) -> Dato resultante de la clase
        """
        
        # Matriz de rotación
        if R.any() != "Und": self.R = R  
        elif (R.any() == "Und") and (θ != "Und") and (eje.any() != "Und"):
            self.R = rt.RotRodrigues(self.eje, self.θ)          # Matriz de rotación
            self.eje = eje                                      # Eje de rotación
            self.θ = θ                                          # Ángulo de rotación
        #else:
        #    raise ValueError("Se requiere un eje y un ángulo para calcular la matriz de rotación.")
        if p.any() != "Und": self.p = p
        if θ != "Und": self.θ = θ
        if eje.any() != "Und": self.eje = eje
        # Podemos añadir atrivutos a medida que vayan apareciendo?

     
        self.T = "Indefinida"
        # Crear la matriz de transformación homogénea 4x4
        if R.any() != "Und" and p.any() != "Und":
            self.T = self.Rp2Trans(R, p)
        elif S.any() != "Und" and θ != "Und": self.T = self.Exp2Trans(S, θ)  # Matriz de transformación homogénea
        #else:
        #    raise ValueError("Faltan datos para crear la matriz de transformación homogénea.")
    
    def __str__(self):
        """ Método para representar la matriz de transformación homogénea como una cadena. """
        return str(self.T)

    # Función para convertir un vector de 6 elementos en una matriz de transformación homogénea
    def Exp2Trans(S, θ):
        """
        Convierte un vector de 6 elementos en una matriz de transformación homogénea 6x6.
        (La matriz exponencial del vector S corresponde con la de transformación homogénea).
        """
        if S.shape != (6,):
            raise ValueError("El vector S debe tener tamaño 6.")
        
        # Extraer la parte angular y la parte de traslación del vector
        w = S[:3]  # w Parte angular (vector de rotación)
        v = S[3:]  # v Parte de traslación (vector de posición)
        modulo_w = np.linalg.norm(w)
        # 
        if modulo_w == 0 and np.linalg.norm(v) == 1: # Articulación prísmaticas
            # Si el vector de rotación es cero y el vector de traslación es unitario:
            # Se aplica la fórmula de Rodriguest
            T = np.eye(4)
            T[:3, 3] = v * θ    # Asignar la traslación
            return T
            """
            return np.array([[1, 0, 0, v[0]]*θ,
                            [0, 1, 0, v[1]]*θ,
                            [0, 0, 1, v[2]]*θ,
                            [0, 0, 0, 1]])
            """
        elif modulo_w == 1 and θ.imag == 0:
            # Si el vector de rotación es unitario y el ángulo es real, se aplica la fórmula de Rodrigues
            R = rt.RotRodrigues(w, θ)
            T = np.eye(4)
            T[:3, :3] = R       # Asignar la rotación (Se sobrescribe la matriz identidad)
            T[:3, 3] = v * θ    # Asignar la traslación
            return T

    # Función para convertir una matriz de rotación y un vector de traslación en una matriz de transformación homogénea
    def Rp2Trans(R, p):
        """ Convierte una matriz de rotación R y un vector de traslación p en una matriz de transformación homogénea 4x4. """
        if R.shape != (3, 3) or p.shape != (3,):
            raise ValueError("La matriz de rotación debe ser de tamaño 3x3 y el vector de traslación debe ser de tamaño 3.")
        
        T = np.eye(4)  # Matriz identidad 4x4
        T[:3, :3] = R  # Asignar la rotación
        T[:3, 3] = p   # Asignar la traslación
        
        return T

    # Función para convertir una matriz de transformación homogénea en una matriz de rotación y un vector de traslación
    def Trans2Rp(T):
        """ Convierte una matriz de transformación homogénea T en una matriz de rotación R y un vector de traslación p. """
        if T.shape != (4, 4):
            raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
        return T[:3, :3], T[:3, 3]

    # Función para calcular el logaritmo de una matriz de transformación homogénea devolviendo S = (w, p)
    def LogTrans(T):
        """ Calcula el logaritmo de una matriz de transformación homogénea T. """
        if T.shape != (4, 4):
            raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
        
        R = T[:3, :3]  # Extraer la matriz de rotación
        p = T[:3, 3]   # Extraer el vector de traslación
        
        θ, w = rt.LogRot(R)  # Calcular el logaritmo de la matriz de rotación
        return np.concatenate((w, p))  # Concatenar el vector de rotación y el vector de traslación

    # Función para calcular la inversa de una matriz de transformación homogénea
    def TransInv(T):
        """ Calcula la inversa de una matriz de transformación homogénea T. """
        if T.shape != (4, 4):
            raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
        Inv = np.eye(4)  # Matriz identidad 4x4
        Inv[:3, :3] = T[:3, :3].T  # Transponer la matriz de rotación
        Inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])  # Calcular la traslación inversa
        return Inv

    # Función para calcular la matriz adjunta de una matriz de transformación homogénea
    def TransAdj(T):
        """
        Calcula la matriz AdjT asociada a una matriz de transformación homogénea T. 
        La matriz AdjT se utiliza para transformar el producto de dos matrices de transformación homogénea"
        """
        if T.shape != (4, 4):
            raise ValueError("La matriz de transformación debe ser de tamaño 4x4.")
        
        R = T[:3, :3]  # Extracción de la matriz de rotación
        p = T[:3, 3]   # Extracción del vector de traslación
        
        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, 3:] = R
        AdT[:3, 3:] = np.dot(rt.antisimetrica(p), R)
        
        return AdT

t = TransformacionHomogenea(R=np.eye(3),p=np.array([1, 2, 3]))

print(t)