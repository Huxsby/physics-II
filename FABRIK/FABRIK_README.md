# FABRIK 3D - Plan de Refactorizacion y Roadmap

## Objetivos

Refactorizar FABRIK_3D en un sistema modular y mantenible con componentes especializados.

## Arquitectura

```
FABRIK/
|-- fabrik_core/
|   |-- fabrik_serial_solver.py    [NUEVO] Solver FABRIK Alg 1+2+3, produccion
|   |-- quaternion_utils.py        [NUEVO] Utilidades cuaternion [w,x,y,z]
|   |-- math_utils.py              [X] Utilidades matematicas (existente)
|   `-- __init__.py                [X] Exportaciones del modulo
|-- tests/
|   `-- test_fabrik_niryo.py       [NUEVO] Bateria de pruebas contra Niryo One
|-- visualization/
|   |-- recorder.py                [X] Grabacion de animaciones
|   `-- __init__.py
|-- fabrik_paper_constrained_3d.py [LEGACY] Ver notas de bugs abajo
|-- FABRIK_2D/                     Implementaciones 2D (constrained/optimized)
|-- referencias/                   Implementaciones de referencia externas
`-- FABRIK_README.md               Este roadmap
```

## Estado actual de implementacion

### [COMPLETADO] fabrik_serial_solver.py + quaternion_utils.py

Implementacion de produccion del algoritmo FABRIK para cadenas seriales (Algorithm 1, 2, 3
segun Aristidou & Lasenby, 2011). Probado contra el robot Niryo One (6 DOF) con 5/5 targets
alcanzables convergiendo correctamente.

**Clases y funciones principales:**
- `JointType`: FREE, BALL, HINGE_GLOBAL, HINGE_LOCAL
- `JointDescriptor`: longitud, limites angulares, eje de bisagra, angulos de workspace
- `SolverResult`: posiciones finales, iteraciones, convergencia, error
- `FabrikSerialSolver`: solver principal
  - `.from_robot(robot)`: construccion desde objeto Robot del proyecto
  - `.solve(target)`: cinematica inversa con Algorithms 1+2+3
  - `.reset_to_initial()`: reinicia la configuracion

**Bugs corregidos respecto al codigo original (fabrik_paper_constrained_3d.py):**

1. Bug Algorithm 2 - cono hueco: `min_cone_angle = cone_angle * 0.1` creaba un cono hueco invalido
   (solo acepta angulos en [0.1*cone, cone] en vez de [0, cone]).
   FIX: restriccion de cono correcta, BALL limita angulo maximo desde la direccion entrante.

2. Bug Algorithm 2 - indice de descriptor incorrecto: se usaba `descriptors[joint_idx - 1]`
   para la restriccion en joint_idx, off-by-one que aplicaba la restriccion del joint BASE
   al joint HOMBRO y asi sucesivamente.
   FIX: se usa `descriptors[joint_idx]` para el joint correcto.

3. Bug Algorithm 2 - longitud de segmento incorrecta: `descriptors[min(idx_next, n-1)].length`
   usaba la longitud del segmento incorrecto al reposicionar el joint.
   FIX: `descriptors[min(idx_curr, idx_next)].length`.

4. Bug Algorithm 2 - limites CW/ACW incorrectos: los limites se calculaban simetricos respecto
   al centro del rango, no respecto a la configuracion cero del robot.
   FIX: cw_deg = degrees(-lo), acw_deg = degrees(hi).

5. Bug Algorithm 2 - HINGE_GLOBAL incorrecto para cadenas seriales: usar ejes de bisagra fijos
   en el frame global bloquea el movimiento en joints cuyo eje efectivo depende de la
   configuracion de joints previos (todos los joints del Niryo excepto la base).
   FIX: se usan restricciones BALL que limitan la desviacion angular en el frame local del joint,
   que es la interpretacion correcta para limites de articulacion en robots seriales.

6. Bug Algorithm 3 - O = target: la proyeccion de workspace usaba el propio target como origen
   (identidad), haciendo que la restriccion no tenga efecto.
   FIX: se usa la posicion de la primera articulacion como origen y la direccion al target
   como eje Z local.

7. Bug de convergencia - singularidad de cadena estirada: una cadena completamente estirada
   en la direccion del target produce oscilaciones sin convergencia (backward y forward se
   cancelan exactamente).
   FIX: configuracion inicial con pequena inclinacion (0.5 deg por segmento) para romper simetria.

**Notas de diseno:**
- Restricciones BALL son la aproximacion correcta para robots seriales en FABRIK.
  Para restricciones HINGE precisas se requiere HINGE_LOCAL con frame local dinamico
  (a implementar si se necesita precision de limite de angulo individual).
- Los ejes de articulacion del YAML estan en el frame LOCAL del joint.
  HINGE_GLOBAL con estos ejes solo es correcto para la articulacion BASE (eje=[0,0,1]).

### [PENDIENTE] Tareas futuras

- [ ] HINGE_LOCAL con frame local dinamico para restricciones precisas de joint angle.
      El eje del hombro/brazo/codo en frame global cambia con la configuracion del robot.
      Requiere propagar el frame local acumulado a traves de la cadena en cada iteracion.

- [ ] Algorithm 4: Conversion de posiciones FABRIK a angulos de articulacion.
      Actualmente el solver devuelve posiciones de joints, no angulos.
      Para control real del robot se necesita la cinematica inversa de angulos.

- [ ] Algorithm 5: FABRIK Multi-Target (cadenas con bifurcaciones, full body).
      Ver referencias/FABRIK_Full_Body-master para la implementacion de referencia.

- [ ] Algorithm 6: Control de orientacion del efector final.
      Restriccion de orientacion en el efector (no solo posicion).

- [ ] Visualizacion 3D del solver: mostrar la cadena cinematica durante la solucion.
      Aprovechar src/animation/class_robot_plotter.py del proyecto.

- [ ] Integracion con TAD Robot mejorado (Link -> Joint con tipo cinematico correcto).

- [ ] Benchmarks de rendimiento y comparacion con referencias externas.

## Bugs documentados en fabrik_paper_constrained_3d.py (LEGACY)

Este archivo tiene implementaciones incorrectas documentadas:

- Algorithm 2: restriccion de cono hueco invalida, eje de referencia hardcodeado a [0,0,1],
  misma funcion de restriccion en ambos passes (sin distincion forward/backward).

- Algorithm 3: O = target (proyeccion identidad), parametros de elipse arbitrarios
  (semi_minor = q*0.7 sin justificacion), punto mas cercano por proyeccion radial (incorrecto),
  no se aplica por joint. El header del fichero indica que Algorithm 3 no esta implementado
  pero el codigo pretende implementarlo.

Se mantiene para referencia historica. No usar para produccion.

## Dependencias instaladas en esta sesion

- scipy 1.17.1
- pyyaml 6.0.3
- imageio-ffmpeg 0.6.0
- sympy 1.14.0
- pandas 3.0.3
- numpy 1.26.4 (preexistente)
- matplotlib 3.10.9 (preexistente)

## Refactorizar (pendiente original)

- [ ] Estudiar e implementar rasgos de la estructura .yaml seguida por NVIDIA Isaac Lab.
- [ ] Modular clase FABRIK como modulo para probar algoritmos genericos.

### Core

- [ ] Mejorar TAD Robot: Link -> Joint con tipo cinematico correcto
- [ ] Integracion FABRIK solver con class_robot_plotter.py para visualizacion

### Fabrik-R

Necesidades Fabrik-R:

