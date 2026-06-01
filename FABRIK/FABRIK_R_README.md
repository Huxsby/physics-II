# FABRIK-R: Especificación Técnica y Guía de Desarrollo para Robots Manipuladores

Este documento centraliza el desglose matemático, algorítmico y operativo de la extensión **FABRIK-R** basada en las investigaciones de Santos et al. (2021, 2022). Está diseñado como una especificación de alta fidelidad para que IAs agénticas de capacidades avanzadas puedan codificar el solver sin desviaciones del modelo matemático original.

---

## 1. El Problema Fundamental de FABRIK Clásico en Robótica
El algoritmo FABRIK original (Aristidou & Lasenby, 2011) asume que las articulaciones son esféricas (Ball-and-Socket, 3-DOF), donde los cambios de posición se resuelven buscando puntos sobre líneas directas entre articulaciones. 

Cuando se intenta aplicar restricciones de 1-DOF (articulaciones de revolución o bisagras) como un post-procesamiento (Algoritmo 2 de Aristidou), el algoritmo sufre de los siguientes fallos críticos en entornos físicos:
1. **Inconsistencia Dimensional:** Al proyectar un punto propuesto $p^*_i$ de vuelta al plano de rotación permitido de la bisagra, la distancia física al eslabón anterior cambia. Al corregir la distancia, el punto vuelve a salirse del plano, generando oscilaciones infinitas o fallos de convergencia ("deadlocks").
2. **Dependencia del Objetivo:** La proyección clásica usa el vector hacia el *Target*. Si el Target está alineado con el eje de la articulación, el plano se vuelve indeterminado (singularidad geométrica), provocando saltos violentos o rotaciones erráticas incompatibles con motores reales.

---

## 2. Fundamentos Matemáticos de FABRIK-R (SANTOS21 / SANTOS22)
FABRIK-R elimina la dependencia del vector *Target* para definir los planos de movimiento de las articulaciones de 1-DOF. En su lugar, el plano de rotación de cada articulación se determina de forma endógena utilizando la topología geométrica de la propia cadena cinemática (los eslabones adyacentes y sus ejes intrínsecos de actuación).

### A. Definición del Plano de Rotación Local
Para una articulación $i$ con un único grado de libertad de rotación, su movimiento está restringido a un plano normal a su vector de eje de rotación $z_i$. 

1. El solver opera manteniendo y transformando los ejes de rotación locales a lo largo de las pasadas Forward y Backward.
2. En lugar de proyectar el punto modificado de manera ingenua, se calcula un vector de dirección corregido $d_i$ que se encuentra estrictamente en el espacio ortogonal a $e_i$ (eje de la bisagra).

### B. Modificación de la Pasada Backward (Hacia la Base)
En la pasada Backward, el efector final se coloca en el Target y los puntos se calculan secuencialmente hacia la base ($p_n 	o p_1$).
Para cada articulación intermedia $p_i$:
1. Se calcula la posición provisional del punto según FABRIK clásico:
   $$p^*_i = p_{i+1} + l_i rac{p_i - p_{i+1}}{\|p_i - p_{i+1}\|}$$
2. Se extrae el eje de rotación local actualizado de la articulación $z_i$.
3. Se proyecta el vector del eslabón corregido de modo que cumpla la condición de ortogonalidad con el eje de la bisagra:
   $$v_{proj} = (p^*_i - p_{i+1}) - \left[(p^*_i - p_{i+1}) \cdot z_iight] z_i$$
4. Se re-normaliza el vector resultante para asegurar de forma estricta la longitud del eslabón $l_i$:
   $$p^{final}_i = p_{i+1} + l_i rac{v_{proj}}{\|v_{proj}\|}$$

### C. Modificación de la Pasada Forward (Hacia el Efector Final)
En la pasada Forward, la base se fija de nuevo en su posición original ($p_1 = b_1$) y los puntos se calculan secuencialmente hacia el efector final ($p_1 	o p_n$).
Para cada articulación intermedia $p_i$:
1. Se calcula el punto provisional:
   $$p^*_i = p_{i-1} + l_{i-1} rac{p_i - p_{i-1}}{\|p_i - p_{i-1}\|}$$
2. Se aplica la restricción de plano basada en el eje local $z_{i-1}$:
   $$v_{proj} = (p^*_i - p_{i-1}) - \left[(p^*_i - p_{i-1}) \cdot z_{i-1}ight] z_{i-1}$$
3. Se establece la posición final del punto para esta pasada:
   $$p^{final}_i = p_{i-1} + l_{i-1} rac{v_{proj}}{\|v_{proj}\|}$$

---

## 3. Límites de Ángulo y Clamping en FABRIK-R
Una vez que el punto se encuentra en el plano de rotación correcto, se deben aplicar los límites físicos de la articulación $[	heta_{min}, 	heta_{max}]$. De acuerdo con **SANTOS22**:

1. Se define el vector de referencia cero ($lpha_i$) dentro del plano de rotación, que representa el ángulo $	heta = 0$.
2. Se calcula el ángulo actual del vector corregido con respecto a $lpha_i$ usando la función arcotangente de dos argumentos (`atan2`):
   $$	heta_{actual} = 	ext{atan2}\left((v_{proj} 	imes lpha_i) \cdot z_i, \, v_{proj} \cdot lpha_iight)$$
3. Si $	heta_{actual}$ viola los límites, se realiza un clamping directo:
   $$	heta_{clamped} = \max(	heta_{min}, \min(	heta_{max}, 	heta_{actual}))$$
4. El punto final ajustado se rota en el plano usando la fórmula de rotación de Rodrigues o cuaterniones utilizando el ángulo $	heta_{clamped}$ alrededor del eje $z_i$:
   $$v_{final} = \cos(	heta_{clamped})lpha_i + \sin(	heta_{clamped})(z_i 	imes lpha_i)$$
   $$p^{clamped}_i = p_{i-1} + l_{i-1} v_{final}$$

---

## 4. Trampas y Errores Críticos de Implementación (Evitar a toda costa)
* **La trampa de la Normalización de Vectores Nulos:** Si el vector del eslabón propuesto coincide exactamente con el eje de rotación $z_i$, el producto vectorial y la proyección darán un vector nulo `[0, 0, 0]`. El código DEBE detectar si $\|v_{proj}\| < 1e-6$ y, en ese caso, forzar el vector hacia la posición del paso de animación o iteración anterior para evitar divisiones por cero (`NaN`).
* **Inversión de Signo en Pasada Backward vs Forward:** Recuerda que en la pasada Backward te mueves desde el efector final a la base, por lo que el sentido de los vectores de los eslabones se invierte en comparación con la pasada Forward. Confundir las referencias espaciales de los eslabones $l_i$ y $l_{i-1}$ causará una divergencia explosiva de la cadena cinemática.
* **Ignorar el Giro del Eje del Frame Anterior (Twist):** En manipuladores de tipo revoluta secuencial (como el Niryo One o el Titan 2), el eje de rotación de la junta $i$ se ve afectado por las rotaciones acumuladas de todas las juntas anteriores ($1$ a $i-1$). Los ejes locales $z_i$ deben actualizarse utilizando cuaterniones de rotación basados en los ángulos calculados en la iteración previa.

---

## 5. Hoja de Ruta para el Desarrollo Agéntico
* [ ] **Fase 1:** Limpieza geométrica del espacio de trabajo moviendo scripts antiguos a `/legacy`.
* [ ] **Fase 2:** Creación de `fabrik_r_solver.py` definiendo la estructura de datos `FABRIKRChain` y `FABRIKRJoint` (con propiedades fijas de eje de rotación local $z_{local}$ y límites angulares).
* [ ] **Fase 3:** Escritura de funciones puras de proyección planar (`project_to_axis_plane`) y manejo de singularidades por vector nulo.
* [ ] **Fase 4:** Codificación del ciclo iterativo principal (Pasadas Backward modificada y Forward modificada con clamping de Rodrigues).
* [ ] **Fase 5:** Implementación del extractor de ángulos final $	heta$ compatible con la representación del robot físico.
* [ ] **Fase 6:** Ejecución de pruebas unitarias de convergencia espacial comparando la posición de los puntos de los eslabones contra un simulador de cinemática directa (Forward Kinematics).