# Examen 1 — Fundamentos de Aprendizaje Profundo (~semana 6)

**Formato:** en clase, individual, lápiz y papel. Sin computadora, sin código.
**Peso:** 25% de la calificación final.
**Material permitido:** una hoja de fórmulas (carta, por ambos lados) escrita a mano por ti.
**Cobertura:** slides *Repaso*, *NN*, *Entrenamiento*, *Entrenamiento2* y *Regularización*; Tareas de práctica 1–5.

Las preguntas **no** piden escribir código. Piden derivar, predecir, diagnosticar y justificar — exactamente el tipo de razonamiento que necesitas para dirigir (y verificar) a un agente de IA que sí escribe el código.

---

## 1. Repaso: aprendizaje supervisado y optimización (Tarea 1)

- Notación: $x$, $y$, $\phi$, $f(x,\phi)$; conjunto de entrenamiento y función de pérdida.
- Derivar a mano $\partial L/\partial \phi_0$, $\partial L/\partial \phi_1$ y la regla de actualización de descenso en gradiente para regresión lineal univariada.
- ¿Por qué abandonamos la solución cerrada (ecuaciones normales) en redes profundas? (al menos dos razones).
- Descenso en gradiente vs. SGD con minibatch: pasos del algoritmo, efecto del tamaño de batch en estabilidad y velocidad de convergencia.
- Aritmética de épocas: dadas $n$ observaciones, tamaño de batch $b$ y $k$ iteraciones, ¿cuántas épocas se completaron?
- Learning rate: efecto de un lr demasiado grande / demasiado pequeño; qué hace un *scheduler* tipo StepLR.
- Generalización: error de entrenamiento vs. riesgo poblacional; sobreajuste y subajuste; papel del split train/val/test.

## 2. Redes neuronales superficiales y profundas (Tarea 2)

- La red ReLU como función lineal por pedazos: regiones lineales, "joints", cómo cada unidad oculta aporta un hiperplano.
- Trazar **a mano** el forward pass de una red pequeña (1-3-1) con parámetros dados (el "Paso 1–4" de clase).
- ¿Qué mapeo resulta si la activación es lineal/identidad? ¿Por qué la no linealidad es indispensable?
- Propiedades de ReLU: demostrar la homogeneidad no negativa $\text{ReLU}(\alpha z) = \alpha\,\text{ReLU}(z)$ para $\alpha \geq 0$.
- Invarianzas de reescalamiento de parámetros: por qué distintos $\phi$ producen la misma función y qué implica para la identificabilidad.
- Conteo de parámetros: fórmula general para $D_i$ entradas, $D_o$ salidas, $D$ unidades ocultas; demostración para $K$ capas ocultas.
- Teorema de aproximación universal (enunciado e intuición, no demostración).
- Superficial vs. profunda: regiones lineales por parámetro, composición, ventajas y costos.

## 3. Funciones de pérdida desde supuestos distribucionales (Tarea 3)

- La "receta" de 4 pasos: elegir distribución → parametrizarla con la red → máxima verosimilitud → menos log-verosimilitud.
- Derivar: pérdida cuadrática desde ruido Gaussiano; entropía cruzada binaria desde Bernoulli + sigmoide; entropía cruzada multiclase desde categórica + softmax.
- Diseñar una pérdida para un problema nuevo (ej.: conteos Poisson de peatones; el examen puede plantear otro contexto de política pública).
- Regresión heteroscedástica: qué cambia en el modelo y en la pérdida.
- Salidas múltiples con unidades distintas (estatura en m, peso en kg): el problema y dos soluciones; cómo priorizar la precisión de una salida sobre otra.
- Convexidad: ¿es convexa la pérdida logística en $\phi$? ¿Cómo lo argumentarías/probarías?

## 4. Optimización (Tareas 3–4)

- Escribir las reglas de actualización de: GD, SGD, SGD con momentum, Nesterov, RMSprop y Adam; explicar qué problema resuelve cada término.
- No convexidad: mínimos locales, puntos silla, valles planos; ¿puede el GD determinista con lr fijo escapar de un mínimo local? ¿y el SGD?
- El confundido optimizador × learning rate: por qué comparar SGD y Adam con el *mismo* lr es una comparación injusta.

## 5. Backpropagation e inicialización (Tarea 4)

- La recurrencia del backward pass (regla de la cadena sobre el grafo de cómputo); calcular a mano los gradientes de una composición escalar pequeña.
- Derivada de la sigmoide: qué pasa con el gradiente para entradas muy positivas/negativas → **gradientes que se desvanecen**.
- ¿Se pueden entrenar por gradiente activaciones tipo Heaviside o rect? ¿Por qué no?
- Inicialización en cero: **ruptura de simetría** — qué sale mal y por qué.
- Propagación de varianza: por qué importa $\sigma^2$ inicial; inicialización de He ($2/D_h$) y Xavier; a qué activación corresponde cada una.
- ¿Puede la pérdida de entrenamiento de entropía cruzada multiclase llegar exactamente a 0? Justifica.

## 6. Regularización (Tarea 5)

- Regularización explícita: cómo cambia L2 el gradiente algebraicamente; equivalencia con *weight decay* en la actualización de SGD; interpretación bayesiana (prior).
- Regularización implícita: sesgos de GD/SGD; cómo la modulan el tamaño de batch y el lr.
- Heurísticas y cuándo preferir cada una: dropout (incluida la regla de escalamiento en inferencia), BatchNorm (qué hace realmente; qué pasa con batch de tamaño 1), early stopping (parecidos y diferencias con L2), ensambles, aumento de datos (por qué las transformaciones deben preservar la etiqueta), transfer learning.
- L1 y esparsidad: por qué L1 produce pesos exactamente cero y L2 no; poda (*pruning*) e implicaciones.
- **Doble descenso**: dibujar/etiquetar la curva, umbral de interpolación, implicaciones para elegir capacidad.
- Régimen de pocos datos: por qué la regularización importa más cuando $n \ll$ número de parámetros.

## 7. Diagnóstico (transversal — aparece en todo el examen)

Dado un artefacto de un entrenamiento real (curvas de pérdida train/val, normas de gradiente por época, histogramas de pesos, tabla de resultados), deberás:
- Diagnosticar: ¿sobreajuste, subajuste, problema de optimización, lr mal elegido, saturación?
- Proponer dos intervenciones concretas y predecir su efecto.
- Detectar resultados sospechosos (ej.: val acc > train acc, pérdida que sube, norma de gradiente que explota).

---

## Tipos de pregunta

| Tipo | Ejemplo |
|---|---|
| Derivación | "Deriva la BCE desde el supuesto Bernoulli." |
| Traza a mano | "Con estos 10 parámetros, evalúa la red en $x=2$ y dibuja la función." |
| Predicción | "Antes de ver los resultados: ¿qué esperas que pase si inicializas todo en 0? ¿Por qué?" |
| Diagnóstico | "Estas son las curvas de pérdida de un modelo real. ¿Qué está mal y qué harías?" |
| Diseño | "Una secretaría quiere predecir conteos de solicitudes diarias por oficina. Propón modelo, pérdida y protocolo de validación." |
| Demostración corta | "Prueba que la red de $K$ capas y $D$ neuronas tiene $3D + 1 + (K-1)D(D+1)$ parámetros." |

## Cómo estudiar

1. Resuelve las **Tareas 1–5** (teoría *y* ejercicios) — los enunciados y soluciones están publicados. Las preguntas del examen siguen su estilo directamente.
2. Para los ejercicios de código de las tareas: puedes pedirle a un LLM que los ejecute, pero practica **predecir el resultado antes** y explicar el porqué después. Eso es lo que evalúa el examen.
3. Repasa las derivaciones de las slides de *Entrenamiento* y *Entrenamiento2* hasta poder reproducirlas sin ver.
