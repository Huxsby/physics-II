""" Código de la práctica 2 punto:
3. Matriz Jacobiana
    La metodologia para calcular la matriz Jacobiana de cualquier robot de lazo abierto consiste simplemente en
    determinar los ejes helicoidales para cada una de las articulaciones del robot en la configuración considerada.
    Estos ejes son directamente las columnas de la matriz Jacobiana. El protocolo detallado pod ́eis verlo en este
    video y el c ́alculo de esta matriz para el Robot Niryo One est ́a descrito aqu ́ı.
    Los ejes de referencia y de rotación utizados en este  ́ultimo video no se corresponden con los utilzados en el
    software Niryo One Studio ni tampoco con el sentido de rotación positivo de algunas de las articulaciones. En
    el c ́odigo que copiamos a continuación se usan exactamente los mismos sistemas de referencia y ejes de rotación
    utilizados en el software del Robot. El c ́odigo est ́a desarrollado con c ́alculo simb ́olico (utilizando la libreria
    sympy) para tener un resultado gen ́erico, parametrizado en función de las coordenadas de las articulaciones.
"""
#!/usr/bin/env python
import numpy as np
import sympy as sp

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

# Calculamos las matrices de rotación a partir de los ejes w, utilizando la f ́ormula de Rodrigues
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
El resultado de este c ́odigo es una matriz Jacobiana parametrizada en función de los  ́angulos de las articulaciones.
A continuación se puede ver c ́omo queda la Matriz Jacobiana de manera general, en función de todas las
coordenadas de las articulaciones, con restricciones en algunas de ellas, y finalmente con valores constantes
para las coordenadas de todas las articulaciones:
"""
# Resultado general parametrizado:
print(J)

# Resultados con restricciones en algunos  ́angulos:
print(J.subs({t[1]:0, t[2]:0, t[3]:0, t[4]:0}))
print(J.subs({t[0]:0, t[2]:0, t[3]:0, t[4]:0}))

# Resultados para varias configuraciones específicas:
Ja=J.subs({t[0]:0, t[1]:0, t[2]:0, t[3]:0, t[4]:0})
print(np.array(Ja).astype(np.float64).round(decimals=3))
Ja=J.subs({t[0]:np.pi, t[1]:np.pi, t[2]:np.pi, t[3]:np.pi, t[4]:np.pi})
print(np.array(Ja).astype(np.float64).round(decimals=3))
Ja=J.subs({t[0]:np.pi/2., t[1]:np.pi/2., t[2]:np.pi/2., t[3]:np.pi/2., t[4]:np.pi/2.})
print(np.array(Ja).astype(np.float64).round(decimals=3))