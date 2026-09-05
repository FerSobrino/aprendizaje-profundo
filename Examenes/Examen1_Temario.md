# Examen 1 — Fundamentos de Aprendizaje Profundo

**Cuándo:** en clase, semana 6.
**Formato:** individual, lápiz y papel, **40 minutos**. Sin computadora, sin celular, sin código.
**Peso:** 25% de la calificación final.
**Material permitido:** una hoja de fórmulas (tamaño carta, por ambos lados) escrita **a mano por ti**. No se permiten fotocopias ni impresiones.
**Cobertura:** slides *Repaso*, *NN*, *Entrenamiento*, *Entrenamiento 2* y *Regularización*, más las Tareas de práctica 1–5. **No** incluye CNNs ni nada posterior.

El examen no pide escribir código ni recordar sintaxis de PyTorch. Pide entender **cómo funcionan las cosas y por qué**: derivar, trazar a mano, predecir qué va a pasar, diagnosticar qué salió mal y justificar. Es exactamente lo que necesitas para dirigir y verificar a un agente de IA que sí escribe el código.

---

## Estructura del examen

| Parte | Qué es | Puntos | Tiempo sugerido |
|---|---|---|---|
| **A. Conceptos rápidos** | 8 afirmaciones: verdadero/falso **con justificación** de 1–2 líneas. Sin justificación no hay puntos. | 24 | ~14 min |
| **B. Desarrollo** | 3 preguntas cortas: una traza a mano de una red pequeña, una derivación de función de pérdida, un cálculo de backprop. | 48 | ~18 min |
| **C. Diagnóstico** | Un escenario de política pública con resultados de entrenamiento (tabla o curvas). Diagnosticar, proponer intervenciones, detectar resultados sospechosos. | 28 | ~8 min |

Las respuestas se califican por el **razonamiento**, no solo por el número final. Una respuesta breve y correcta vale más que una larga y vaga.

---

## Lo que debes poder hacer

### 1. Repaso: aprendizaje supervisado y optimización (slides *Repaso*, Tarea 1)

- Escribir el problema de entrenamiento: modelo $f(x,\phi)$, pérdida por observación $\ell$, pérdida total $L(\phi)$, $\hat\phi=\arg\min_\phi L(\phi)$.
- Derivar a mano las parciales y la actualización de gradient descent para regresión lineal univariada.
- Explicar por qué en deep learning no usamos la solución cerrada ni búsqueda exhaustiva (dar dos razones).
- GD vs. SGD con mini-batch: qué cambia en la actualización, qué compromiso resuelve el tamaño de batch (costo, ruido, hardware), qué es una época.
- **Aritmética de épocas:** dadas $n$ observaciones, batch $b$ e iteraciones $k$, calcular cuántas épocas se completaron.
- Explicar de dónde viene la pérdida cuadrática: máxima verosimilitud con ruido gaussiano de varianza constante. Saber qué término desaparece y por qué.
- Riesgo empírico vs. riesgo de población; para qué sirve cada conjunto (train / validation / test) y qué sale mal si tomas decisiones con el test.
- Regularización $L_2$ en regresión: cómo cambia el gradiente y por qué la actualización resultante se llama *weight decay*.

### 2. Redes superficiales y profundas (slides *NN*, Tarea 2)

- La red ReLU como función **lineal por partes**: cada unidad oculta aporta una articulación; $D$ unidades dan hasta $D+1$ regiones en 1D.
- **Trazar a mano** el forward pass de una red 1-3-1 con parámetros dados: evaluar en varios puntos, ubicar las articulaciones, calcular la pendiente en cada región.
- Qué pasa si la activación es la identidad o lineal (la red colapsa a una función lineal).
- Homogeneidad no negativa de ReLU y la invarianza de reescalamiento de parámetros: distintos $\phi$ pueden dar la misma función.
- **Contar parámetros:** red superficial con $D_i$ entradas, $D$ unidades y $D_o$ salidas; red profunda con $K$ capas de $D$ unidades ($3D+1+(K-1)D(D+1)$ para 1 entrada y 1 salida).
- Teorema de aproximación universal: enunciado e intuición. Por qué **no** implica que la profundidad sea inútil: aproximación universal $\neq$ eficiencia representacional ($D+1$ regiones vs. hasta $(D+1)^K$).
- Hiperparámetros ($K$, $D_k$) vs. parámetros: quién decide cada uno y cuándo.
- Las tres cajas de la comparación superficial vs. profunda: más profundidad $\not\Rightarrow$ entrenamiento más fácil; buena optimización $\neq$ buena generalización.

### 3. Funciones de pérdida desde distribuciones (slides *Entrenamiento*, Tarea 3)

- La **receta**: elegir distribución para el output, predecir sus parámetros con la red, máxima verosimilitud, minimizar la negative log-likelihood.
- Reproducir las tres derivaciones: Normal $\Rightarrow$ MSE; Bernoulli + sigmoide $\Rightarrow$ binary cross-entropy; Categórica + softmax $\Rightarrow$ cross-entropy multiclase.
- **Diseñar una pérdida para un problema nuevo** (por ejemplo conteos): qué distribución, cómo garantizar que el parámetro predicho esté en el rango válido (sigmoide, softmax, exponencial), escribir y simplificar la NLL.
- Regresión heteroscedástica: qué cambia en el modelo y qué cuidado hay que tener con $\sigma^2(x)>0$.
- Outputs múltiples: el supuesto de independencia condicional y cuándo falla.
- Cross-entropy y KL: por qué minimizar $D_{KL}(q\|p_\phi)$ equivale a minimizar $H(q,p_\phi)$ y por qué con etiquetas one-hot coincide con la NLL.
- Optimizadores: escribir las actualizaciones de GD, SGD, momentum, Nesterov y Adam; explicar qué problema resuelve cada ingrediente (promedio de gradientes, mirar adelante, normalización por coordenada, corrección de sesgo).
- No convexidad: mínimos locales, puntos silla, regiones planas. Qué puede hacer el ruido de SGD que GD determinista no.

### 4. Backpropagation e inicialización (slides *Entrenamiento 2*, Tarea 4)

- Forward pass guarda valores intermedios; backward pass aplica la regla de la cadena de atrás hacia adelante y reutiliza derivadas. Por qué requiere memoria.
- **Calcular a mano** los gradientes de una composición pequeña (dos capas escalares con ReLU) respecto a todos los parámetros.
- La recurrencia vectorial: $\delta_k=\partial\ell/\partial f_k$, $\delta_{k-1}=(\Omega_k^T\delta_k)\odot a'(f_{k-1})$, $\partial\ell/\partial\beta_k=\delta_k$, $\partial\ell/\partial\Omega_k=\delta_k h_k^T$.
- Qué pasa con el gradiente cuando una ReLU está inactiva, o cuando una sigmoide está saturada.
- **Inicialización en cero:** ruptura de simetría, por qué las unidades de una capa quedan idénticas.
- Propagación de varianza: por qué $\sigma_\Omega^2$ importa, de dónde sale He ($2/D_h$) y a qué activación corresponde; Xavier ($2/(D_{in}+D_{out})$) y a qué activación corresponde. Vanishing y exploding gradients.
- Ruido, sesgo y varianza: qué es cada uno, qué reduce la varianza y qué reduce el sesgo, y el trade-off clásico.
- **Double descent:** dibujar la curva, ubicar el umbral de interpolación, nombrar las tres zonas, y explicar por qué contradice la historia clásica. El papel del sesgo inductivo.
- Cómo se eligen hiperparámetros en la práctica y por qué el test set se usa una sola vez.

### 5. Regularización (slides *Regularización*, Tarea 5)

- Regularización explícita: $\lambda g(\phi)$, e interpretación como prior (estimación MAP, $\lambda g(\phi)\approx-\log P(\phi)$).
- $L_2$ / weight decay: qué hace a los pesos, por qué no se aplica a los biases, y por qué **no** garantiza mejor generalización.
- Regularización **implícita**: en qué sentido GD y SGD prefieren ciertas soluciones (la pérdida modificada con $\|\nabla L\|^2$ y la varianza entre mini-batches); cómo influyen learning rate y batch size.
- Heurísticas: para cada una, qué hace, cuándo ayuda y cuándo no:
  - **BatchNorm:** la operación, entrenamiento vs. inferencia, qué pasa con batches muy pequeños y qué alternativas hay.
  - **Early stopping:** qué monitorea, sus hiperparámetros, en qué se parece y en qué difiere de $L_2$.
  - **Dropout:** la regla de inverted dropout y por qué **no** se reescala en inferencia; la interpretación como ensamble de subredes; MC dropout.
  - **Ensambles**, **ruido** en inputs/pesos/etiquetas (cuándo daña), **data augmentation** (las transformaciones deben preservar la etiqueta), **transfer learning** y **multi-task** (transferencia negativa), **self-supervised**.
- Régimen de pocos datos: qué priorizar (augmentation, transfer learning, modelos preentrenados) y por qué agregar capacidad sin datos ni sesgo inductivo puede empeorar.

### 6. Diagnóstico (transversal, Parte C)

Dado un resumen de entrenamiento (pérdida train/val por época, accuracy, norma de gradientes, tabla de experimentos), debes poder:

- Clasificar el problema: subajuste (sesgo), sobreajuste (varianza), problema de optimización (learning rate, inicialización, gradientes que explotan o desaparecen).
- Proponer **dos intervenciones concretas** y predecir su efecto sobre train y val.
- Detectar resultados sospechosos: val acc mucho mayor que train acc, pérdida que sube o da NaN, test usado para elegir hiperparámetros, fuga de información entre train y test.
- Evaluar críticamente una propuesta ("un agente sugiere duplicar las capas"): ¿ataca el problema correcto?

---

## Ejemplos del tipo de pregunta

Estas **no** son las preguntas del examen, pero son del mismo estilo y dificultad.

| Tipo | Ejemplo |
|---|---|
| Verdadero/falso | "Con inverted dropout hay que multiplicar las activaciones por $(1-p)$ en inferencia." Justifica. |
| Traza a mano | "Con estos 10 parámetros, evalúa la red 1-3-1 en $x=0,2,4$, ubica las articulaciones y da la pendiente de cada región." |
| Derivación | "Una dependencia quiere predecir cuántos días tarda un trámite. Elige distribución, escribe la NLL y simplifícala." |
| Backprop | "Para $\ell=(\beta_1+\omega_1\,\text{ReLU}(\beta_0+\omega_0 x)-y)^2$ con estos valores, calcula las cuatro parciales." |
| Predicción | "¿Qué esperas que pase si inicializas todo en cero? ¿Y si $\sigma_\Omega^2\ll 2/D_h$ en una red de 50 capas?" |
| Diagnóstico | "Train loss 0.02, val loss 2.9 y subiendo desde la época 8. ¿Qué pasa y qué harías?" |
| Crítica | "Un agente propone hacer la red más grande para bajar el error de validación de un modelo que subajusta. ¿Tiene sentido?" |

---

## Cómo estudiar

1. **Resuelve las Tareas 1–5** completas. Los enunciados y las soluciones de la parte práctica están en el repo. Las preguntas del examen siguen ese estilo.
2. Para cada notebook de clase (Optimizadores, Diagnósticos, Regularización): antes de correr una celda, **predice** qué va a salir y por qué. Después compara. Eso es lo que evalúa la Parte C.
3. Reproduce **sin ver las slides** las derivaciones de *Entrenamiento* (MSE, BCE, cross-entropy) y de *Entrenamiento 2* (backward pass escalar, varianza de He).
4. Practica trazar a mano redes 1-3-1 con parámetros inventados hasta que las articulaciones y pendientes te salgan sin pensar.
5. Arma tu hoja de fórmulas **escribiéndola tú**: el acto de resumir es el estudio. Incluye las actualizaciones de los optimizadores, las tres pérdidas, la recurrencia de backprop y He/Xavier.
6. Puedes usar un LLM para estudiar: pídele que te ponga preguntas de este temario y **que no te dé la respuesta hasta que tú des la tuya**. El examen es a mano y sin él.
