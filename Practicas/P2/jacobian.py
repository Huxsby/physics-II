""" Código de la práctica 2 punto:
3. Matriz Jacobiana
    La metodologia para calcular la matriz Jacobiana de cualquier robot de lazo abierto consiste simplemente en
    determinar los ejes helicoidales para cada una de las articulaciones del robot en la configuración considerada.
    Estos ejes son directamente las columnas de la matriz Jacobiana. El protocolo detallado pod ́eis verlo en este
    video y el cálculo de esta matriz para el Robot Niryo One está descrito aquí.
    Los ejes de referencia y de rotación utizados en este  ́ultimo video no se corresponden con los utilzados en el
    software Niryo One Studio ni tampoco con el sentido de rotación positivo de algunas de las articulaciones. En
    el código que copiamos a continuación se usan exactamente los mismos sistemas de referencia y ejes de rotación
    utilizados en el software del Robot. El código está desarrollado con cálculo simbólico (utilizando la libreria
    sympy) para tener un resultado gen ́erico, parametrizado en función de las coordenadas de las articulaciones.
"""
#!/usr/bin/env python
import numpy as np
import sympy as sp

""" Funcines mías --------------------------------------------------- """
def mostrar_jacobiana_resumida(Jacobian: sp.Matrix, max_chars=30):
    """ Muestra la matriz Jacobiana de forma resumida, limitando el número de caracteres por elemento. """
    rows, cols = np.shape(Jacobian)

    # Primero, convertir todo a texto corto
    matrix_text = []
    for i in range(rows):
        row = []
        for j in range(cols):
            elem = str(Jacobian[i, j])
            if len(elem) > max_chars:
                elem = elem[:max_chars] + "..."
            row.append(elem)
        matrix_text.append(row)

    # Calcular el ancho máximo de cada columna
    col_widths = [max(len(matrix_text[i][j]) for i in range(rows)) for j in range(cols)]

    # Imprimir con bordes
    print()
    for i in range(rows):
        formatted_row = []
        for j in range(cols):
            elem = matrix_text[i][j]
            formatted_row.append(elem.ljust(col_widths[j]))  # Alinear a la izquierda

        row_text = "  ".join(formatted_row)  # Separador entre columnas
        if i == 0:
            print(f"⎡ {row_text} ⎤")
        elif i == rows - 1:
            print(f"⎣ {row_text} ⎦")
        else:
            print(f"⎢ {row_text} ⎥")

""" ----------------------------------------------------------------- """
# Función que convierte un eje de rotación en matriz antisim ́etrica 3x3 (so3)
def VecToso3(w): return sp.Matrix([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

# Definimos ejes de rotación de las articulaciones en la posición cero del robot
w=[]
w.append(np.array([0,0,1]))
w.append(np.array([0,-1,0]))
w.append(np.array([0,-1,0]))
w.append(np.array([1,0,0]))
w.append(np.array([0,-1,0]))
w.append(np.array([1,0,0]))

# Definimos los eslabones
scalefactor=0.001 # para cambiar las unidades a metros
L=np.array([103.0, 80.0, 210.0, 30.0, 41.5, 180.0, 23.7, -5.5])*scalefactor

# Definimos los vectores que van del centro de cada eje al centro del siguiente
q=[]
q.append(np.array([0,0,L[0]]))
q.append(np.array([0,0,L[1]]))
q.append(np.array([0,0,L[2]]))
q.append(np.array([L[4],0,L[3]]))
q.append(np.array([L[5],0,0]))
q.append(np.array([L[6],0,L[7]]))

# Coordenadas de las articulaciones
t=sp.symbols('t0, t1, t2, t3, t4, t5')

# Calculamos las matrices de rotación a partir de los ejes w, utilizando la fórmula de Rodrigues
R=[]
for i in range(0,6,1):
    wmat=VecToso3(w[i])
    R.append(sp.eye(3)+sp.sin(t[i])*wmat+(1-sp.cos(t[i]))*(wmat*wmat))

# Aplicamos rotaciones a los vectores q y w para llevarlos a la configuración deseada
qs=[]; ws=[]; Ri=R[0]
qs.append(sp.Matrix(q[0]))
ws.append(sp.Matrix(w[0]))
for i in range(1,6,1):
    ws.append(Ri*sp.Matrix(w[i]))
    qs.append(Ri*sp.Matrix(q[i])+qs[i-1])
    Ri=Ri*R[i]

# Calculamos las velocidades lineales, los vectores giro correspondientes y la matriz Jacobiana
vs=[]; Ji=[]
i=0
vs.append(qs[i].cross(ws[i]))
Ji.append(ws[i].row_insert(3,vs[i]))
J=Ji[0]
for i in range(1,6,1):
    vs.append(qs[i].cross(ws[i]))
    Ji.append(ws[i].row_insert(3,vs[i]))
    J=J.col_insert(i,Ji[i])
"""
El resultado de este código es una matriz Jacobiana parametrizada en función de los ángulos de las articulaciones.
A continuación se puede ver cómo queda la Matriz Jacobiana de manera general, en función de todas las
coordenadas de las articulaciones, con restricciones en algunas de ellas, y finalmente con valores constantes
para las coordenadas de todas las articulaciones:
"""
# Resultado general parametrizado:
print("\n Matriz Jacobiana parametrizada:")
mostrar_jacobiana_resumida(J)

# Resultados con restricciones en algunos ángulos:
print("\nConfiguración 0 parcial, (t0); t1=t2=t3=t4=0")
mostrar_jacobiana_resumida(J.subs({t[1]:0, t[2]:0, t[3]:0, t[4]:0}))
print("\nConfiguración 0 parcial, (t1); t0=t2=t3=t4=0")
mostrar_jacobiana_resumida(J.subs({t[0]:0, t[2]:0, t[3]:0, t[4]:0}))

# Resultados para varias configuraciones específicas:

Ja=J.subs({t[0]:0, t[1]:0, t[2]:0, t[3]:0, t[4]:0})
print("\nConfiguración 0")
mostrar_jacobiana_resumida(np.array(Ja).astype(np.float64).round(decimals=3))

Ja=J.subs({t[0]:np.pi, t[1]:np.pi, t[2]:np.pi, t[3]:np.pi, t[4]:np.pi})
print("\nConfiguración pi")
mostrar_jacobiana_resumida(np.array(Ja).astype(np.float64).round(decimals=3))

Ja=J.subs({t[0]:np.pi/2., t[1]:np.pi/2., t[2]:np.pi/2., t[3]:np.pi/2., t[4]:np.pi/2.})
print("\nConfiguración pi/2")
mostrar_jacobiana_resumida(np.array(Ja).astype(np.float64).round(decimals=3))

input("Presiona una tecla para continuar...")
""" 4. Configuraciones Singulares
En una configuración singular hay direcciones del movimiento que están restringidas. Cada columna de la matriz
Jacobiana puede interpretarse como la velocidad del elemento terminal en una situación en la que la velocidad
de la coordenada de la articulación correspondiente a esa columna, rota con una velocidad angular de 1 rad/s
y el resto de las articulaciones están fijas. Desde esta definición puede deducirse que en una configuración
singular 2 o más columnas de la matriz Jacobiana son linealmente dependientes, por lo que su determinante es
nulo. Por lo tanto, para encontrar las configuraciones singulares del robot basta con encontrar los valores de θi
que hacen nula la matriz Jacobiana. De nuevo podemos hacer uso del cálculo simbólico para encontrar estas
configuraciones singulares. La resolución del determinante completo, sin ninguna restricción, lleva demasiado
tiempo de cómputo. Podemos, sin embargo, imponer restricciones en varias configuraciones para hacer el cálculo
más sencillo. Veamos algunos ejemplos:
"""
sol1 = sp.solve(J.subs({t[2]:0, t[3]:0, t[4]:0}).det())
# Resultado: [{t1: -1.57079632679490}, {t1: 1.57079632679490}]

sol2= sp.solve(J.subs({t[1]:0, t[3]:0, t[4]:0}).det())
# Resultado: [{t2: -1.70541733137745},
            # {t2: -1.57079632679490},
            # {t2: 1.43617532221234},
            # {t2: 1.57079632679490}]

print("\nConfiguraciones singulares:", sol1,'\n', sol2)
"""
OJO!!! Estas configuraciones singulares, obtenidas matemáticamente, pueden no ser accesibles
por el robot. Antes de mover el robot a alguna de estas configuraciones, verifica que est ́e dentro
del rango alcanzable para las articulaciones correspondientes.
"""

""" 5. Elipsoides de Manipulabilidad y Fuerza
La matriz Jacobiana mapea las velocidades de las articulaciones en una configuración dada, a velocidades del
elemento terminal. Tambi ́en sirve para mapear los torques o pares de fuerza en las articulaciones, a la fuerza
neta que puede ejercer el elemento terminal en esa configuración.
    v = J(θ) * ̇θ    ;  F = [(J)**-T]*(θ)τ
En clase vimos como calcular elipses (2D) de manipulabilidad y fuerza para un robot plano 2R, partiendo de
una circunferencia de radio unidad en el espacio generado por las velocidades de las articulaciones (θ) o en
el espacio de los torques (τ). En el caso del robot Niryo One tendríamos un elipsoide en 6 dimensiones en el
espacio de velocidades o fuerzas a partir de hiperesferas en  ̇θ o τ. No es posible representar gr ́aficamente estas
figuras 6D de manera sencilla. Podemos sin embargo escoger sólo 2 ó 3 articulaciones del robot, manteniendo las
dem ́as fijas, y calcular el elipsoide de manipulabilidad en esas condiciones. Para utilizar 3 dimensiones, debes
partir de los puntos de la superficie de una esfera en el espacio de las velocidades de las articulaciones. recuerda
que la ecuación de una esfera de radio unidad en este escenario vendría dada por:
    ∑(θi)**2 = 1
"""
# Por lo que puedes generar puntos en su superficie en coordenadas esféricas como se muestra a continuación:
r = 1
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
for i in range (0,100,1):
    for j in range (0,100,1):
        x = r * np.cos(u[i]) * np.sin(v[j])
        y = r * np.sin(u[i]) * np.sin(v[j])
        z = r * np.cos(v[j])
# A partir de estos puntos podemos construir los vectores que contienen las 6 velocidades de las articulaciones (dθ/dt) para cada punto de la esfera:
vjoints = np.array([x,y,z,0,0,0])
"""
A continuacion, multiplicamos estos vectores por la matriz Jacobiana de la conformación que queremos estudiar.
Por ejemplo, la matriz Jacobiana de la posición cero del robot se puede calcular introduciendo en el código de
la sección 3 la siguiente línea:
"""
# Jacobiana de la configuración 0 del robot
Jp0=J.subs({t[0]:0, t[1]:0, t[2]:0, t[3]:0, t[4]:0})
# Que nos da como resultado: Matrix([[0, 0, 0, 1, 0, 1],[0, -1, -1, 0, -1, 0],[1, 0, 0, 0, 0, 0],[0, 0.183, 0.393, 0, 0.423, 0],[0, 0, 0, 0.423, 0, 0.4175],[0, 0, 0, 0, -0.2215, 0]])
print("\n", Jp0)

# Multiplicando esta matriz por los vectores  ̇θ (los puntos de la esfera), obtenemos el vector giro, que podemos almacenar en un array:
giro = []
giro.append(np.dot(Jp0,vjoints)) # Mod. Anteriormente np.dot(Jp0,v)
"""
    Para cada punto de la esfera en el espacio de velocidades de las articulaciones, se obtiene un punto del elipsoide
de manipulabilidad en el espacio de velocidades del elemento terminal. Tanto la esfera en el espacio de  ̇θ como
los vectores giro se pueden representar en 3D. Esto nos permite visualizar directamente el elipsoide.
Para el elipsoide de fuerza se sigue un proceso totalmente an ́alogo, aunque en este caso se utiliza la inversa de
la traspuesta de la matriz Jacobiana para transformar los torques de las articulaciones en la llave que ejerce el
elemento terminal.

    El caso bidimensional, en el que sólo consideramos 2 articulaciones, es mucho m ́as sencillo y r ́apido, ya que sólo
necesitamos los puntos de una circunferencia, que se transforman en una elipse a trav ́es de la matriz Jacobiana.
Para un Robot gen ́erico de lazo abierto y un espacio de tareas con coordenadas q ∈Rm donde m ≤n, el elipsoide
de manipulabilidad representa las velocidades del elemento terminal para velocidades de articulaciones dadas
por  ̇θ donde || ̇θ|| = 1 ⇒ una esfera unitaria en el espacio de las velocidades de las articulaciones ⇒  ̇θT  ̇θ = 1.
En este escenario, podemos definir una matriz A = JJT a partir de la cual es posible estimar el volumen del
elipsoide de manipulabilidad (ver demostración en los apuntes de clase):
    ∝ √det(J*(J)**T) (1)

    Análogamente podemos estimar el volumen del elipsoide de fuerza a partir de la inversa de la matriz A. Los
ejes principales del elipsoide de fuerza est ́an alineados con los del elipsoide de manipulabilidad (los autovectores
de la matriz A coinciden con los de la matriz A-1) pero los autovalores de A coinciden con los inversos de
los autovalores de A-1. Esto quiere decir que conforme el volumen del elipsoide de manipulabilidad aumenta,
disminuye el volumen del elipsoide de fuerza y viceversa, de manera que el producto de ambos vol ́umenes es
constante.
"""

# 5.1. Evaluación del volumen del elipsoide de manipulabilidad, utilizando el código de la sección 3
# Primero obtenemos la matriz Jacobiana. Tomando la configuración cero del robot:
Jp0=np.array(J.subs({t[0]:0, t[1]:0, t[2]:0, t[3]:0, t[4]:0}), dtype=np.float64)
# A partir de la Jacobiana podemos calcular los elipsoides de manipulabilidad y fuerza en 2 dimensiones, si
# restringimos las velocidades de las articulaciones a sólo 2 grados de libertad:

import numpy as np
import matplotlib.pyplot as plt
Jp0=np.array(J.subs({t[0]:0, t[1]:0, t[2]:0, t[3]:0, t[4]:0}), dtype=np.float64)

# Generamos las coordenadas de una circunferencia
u = np.linspace(0, np.pi/2, 100)
x=[];y=[]
for i in range(0,100,1):
    x.append(np.cos(u[i]))
    y.append(np.sin(u[i]))
x=np.array(x)
xx=np.r_[np.r_[x,-x], np.r_[-x,x]]
y=np.array(y)
yy=np.r_[np.r_[y,y], np.r_[-y,-y]]

# Calculamos los vectores giro, multiplicando por la Matriz Jacobiana
giro=[]
for i in range(0,400,1):
    vjoints=np.array([0,xx[i],yy[i],0,0,0])
    giro.append(np.dot(Jp0,vjoints))
giro=np.array(giro)

# Calculamos los vectores llave, multiplicando por la inversa de la traspuesta de J
llave=[]
Jp0_T=np.linalg.inv(np.transpose(Jp0))
for i in range(0,400,1):
    taujoints=np.array([0,xx[i],yy[i],0,0,0])
    llave.append(np.dot(Jp0_T,taujoints))
llave=np.array(llave)

# Extraemos las componentes no nulas del vector giro para representarlas
vx=giro[:,1]; vy=giro[:,3]
fx=llave[:,1]; fy=llave[:,3]

# Generamos gr ́afica con círculo y elipses
plt.scatter(xx,yy, label=r'${\dot\theta}$ ó $\tau$') #${\dot\theta}$ ó $\tau$'
plt.scatter(vx,vy, label="manipulabilidad")
plt.scatter(fx,fy, label="fuerza")
limitplot=8
plt.ylim(top = limitplot, bottom = -limitplot)
plt.xlim(left = limitplot, right = -limitplot)
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.tight_layout()
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()
"""
Como resultado de este código debes obtener la siguiente gráfica:
Puedes calcular el volumen de los elipsoides añadiendo las siguientes líneas al código que calcula las Matrices
Jacobianas (sección 3):
"""
Jpi = J.subs({t[0]:np.pi, t[1]:np.pi, t[2]:np.pi, t[3]:np.pi, t[4]:np.pi})

# Volumen del elipsoide de manipulabilidad
vol_EM = (Jpi*sp.Transpose(Jpi)).det()

# Volumen del elipsoide de fuerza
vol_EF = ((Jpi*sp.Transpose(Jpi)).inv()).det()

print(f"El vólumen del elipsoide de Manipulabilidad es {vol_EM} y el de Fuerza {vol_EF}")