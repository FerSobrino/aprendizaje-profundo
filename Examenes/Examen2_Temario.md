# Examen 2 — Arquitecturas: CNNs, RNNs, Transformers y LLMs

**Cuándo:** en clase, aprox. semana 11.
**Formato:** individual, lápiz y papel, **40 minutos**. Sin computadora, sin celular, sin código.
**Peso:** 30% de la calificación final.
**Material permitido:** ninguno propio. El examen incluye al final un **formulario de arquitecturas** con las ecuaciones de referencia (tamaño de salida y parámetros de una convolución, receptive field, actualización de RNN, gates de LSTM y GRU, atención de producto punto escalado, multi-head, codificación posicional, perplejidad). Tener la fórmula no es la habilidad evaluada; saber **qué hace, cuándo aplica y qué pasa si la cambias**, sí.
**Cobertura:** slides *CNN*, *RNN* y *Transformers* (incluida la parte de LLMs: pre-entrenamiento, SFT/RLHF/RLVR, LoRA, RAG, evaluación), más las Tareas de práctica 6–8. **No** incluye los decks de *Agentes* ni de *Ética*. Los fundamentos del Examen 1 se asumen conocidos y pueden aparecer dentro de una pregunta, no como pregunta principal.

Igual que en el Examen 1: nada de escribir código. Se evalúa que entiendas **cómo y por qué funcionan** las arquitecturas, que puedas hacer los cálculos pequeños que revelan si entiendes (tamaños, parámetros, pesos de atención), y que puedas razonar sobre cuál usar, qué puede fallar y cómo verificar lo que un agente de IA construya por ti.

---

## Lo que debes poder hacer

### 1. Redes convolucionales (slides *CNN*, Tarea 6)

- Explicar por qué una red totalmente conectada falla en imágenes (el conteo de parámetros de $224\times224\times3$ como argumento) y qué dos supuestos explota una CNN: **localidad** y **compartición de parámetros**.
- **Invarianza vs. equivarianza** a traslaciones: definiciones, cuál aporta la convolución y cuál el pooling o la agregación global; para qué tarea quieres cada una (clasificación vs. segmentación); dar un ejemplo donde la invarianza es una mala idea.
- De la capa totalmente conectada a la convolución: reindexar alrededor de cada pixel, compartir pesos, restringir a una vecindad. Convolución vs. correlación cruzada y por qué la diferencia no importa cuando los pesos se aprenden.
- **Aritmética de convoluciones:** tamaño de salida con padding y stride; conteo de parámetros $c_o c_i k_h k_w + c_o$; qué pasa con parámetros y cómputo si duplicas $c_i$ y $c_o$; costo en función del tamaño espacial de la salida.
- **Receptive field:** calcularlo tras apilar capas, y cómo lo agrandan la profundidad y el stride.
- Convoluciones $1\times1$: qué mezclan y qué no, y para qué se usan.
- Pooling: qué aporta, qué no (no crea invariancia total), por qué no tiene parámetros.
- Upsampling: repetir, max-unpooling, bilineal, transposed convolutions; cuándo hace falta (segmentación).
- AlexNet y ResNet: qué combinación de ingredientes hizo funcionar a AlexNet; qué problema resuelven las conexiones residuales $h_{k+1}=h_k+F(h_k)$ y por qué reaparecen en Transformers.
- Detección y segmentación a nivel conceptual: qué predice YOLO por celda, qué es IoU, por qué las redes de segmentación son encoder-decoder con skip connections.
- **Caso Jean et al. (2016):** transfer learning en dos pasos con luminosidad nocturna como proxy; qué supuestos hace y qué puede salir mal (generalización entre regiones, quién queda invisible).

### 2. Redes recurrentes y modelos de lenguaje (slides *RNN*, Tarea 7)

- Modelos autoregresivos: ventana fija vs. variable latente $h_t=g(h_{t-1},x_t)$; qué gana y qué pierde cada una.
- Estacionariedad (estricta vs. débil) y por qué importa para partir una secuencia en train y validación.
- Modelos de lenguaje: la regla de la cadena $P(x_1,\ldots,x_T)=\prod_t P(x_t|x_{<t})$, Markov de orden $k$, n-gramas, por qué el conteo falla (Zipf, ceros) y qué hace el smoothing.
- **Perplejidad:** definición desde la cross-entropy, interpretación como número efectivo de opciones, por qué vale $|V|$ para el modelo uniforme y por qué **no es comparable entre tokenizaciones distintas**.
- La RNN simple $H_t=\phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h)$: parámetros compartidos en el tiempo, por qué su número no crece con $T$; hidden layer vs. hidden state.
- **BPTT:** de dónde salen los productos a través del tiempo y por qué producen vanishing/exploding gradients ($0.9^{100}$ vs. $1.1^{100}$); BPTT truncado y qué sacrifica.
- **Gradient clipping:** qué hace exactamente (reescala, conserva la dirección), qué problema resuelve y cuál **no**.
- **LSTM:** el papel de cada gate; por qué $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ con $f_t\approx1$, $i_t\approx0$ preserva información; por qué eso crea un camino donde el gradiente no se multiplica repetidamente por $W_{hh}$.
- **GRU:** reset y update; en qué se simplifica respecto a LSTM y por qué no hay una regla universal de cuál es mejor.
- RNNs bidireccionales: cuándo sí (toda la secuencia disponible) y cuándo no (predicción en tiempo real, generación).
- Por qué los Transformers desplazaron a las RNNs y dónde las RNNs siguen teniendo sentido (latencia, memoria constante, recursos limitados).

### 3. Atención y Transformers (slides *Transformers*, Tarea 8)

- El cuello de botella del vector de contexto en encoder-decoder recurrente y por qué motivó la atención.
- **Q/K/V:** la analogía de la biblioteca; atención como promedio ponderado $\sum_i\alpha(q,k_i)v_i$ con pesos softmax; casos extremos (un solo peso en 1, pesos uniformes).
- **Calcular a mano** una atención pequeña: scores, softmax, salida. Producto punto escalado: por qué dividir entre $\sqrt{d_k}$ y qué se rompe sin eso. Atención aditiva como alternativa y cuándo tendría ventaja.
- **Multi-head:** qué gana sobre una sola cabeza; cómo diseñarías el experimento para medir la importancia de una cabeza.
- Self-attention es **invariante al orden** de los tokens: por qué, y por qué eso obliga a la codificación posicional. Sinusoidal vs. aprendida: generalización a secuencias largas, parámetros.
- **Máscaras:** qué hace la máscara aditiva $M_{ij}\in\{0,-\infty\}$ antes del softmax; padding mask; causal mask y qué degenera si entrenas un decoder sin ella.
- El bloque Transformer: multi-head self-attention, feed-forward por posición, residual + LayerNorm; encoder vs. decoder (masked self-attention y cross-attention); poder narrar el flujo completo.
- Tokenización por subpalabras (BPE): qué problema resuelve (OOV vs. longitud de secuencia) y por qué cambia la perplejidad.
- **Objetivos de pre-entrenamiento:** MLM (BERT) vs. LM causal (GPT) vs. denoising seq2seq (T5); qué arquitectura conviene a qué tarea y dónde va la cabeza de clasificación.
- Métricas: accuracy/F1, BLEU, ROUGE, perplejidad, y sus límites: fluidez $\neq$ veracidad $\neq$ utilidad.

### 4. LLMs: de modelo base a asistente (parte final del deck *Transformers*)

- Por qué un LM base no es un asistente; qué agrega el **SFT** (mismo objetivo, otros datos).
- **RLHF** a nivel conceptual (comparaciones → modelo de recompensa → RL) y qué simplifica **DPO**; **RLVR** y por qué el progreso fue más rápido donde hay verificador mecánico (matemáticas, código).
- Las tres vías de cómputo: pre-entrenamiento, post-entrenamiento, test-time compute; qué son los reasoning budgets y su costo.
- **MoE:** router y expertos; por qué los parámetros totales ya no predicen el costo por token.
- **LoRA:** la corrección de rango bajo $\Delta W=AB$ y por qué abarata el fine-tuning.
- **RAG:** qué problema resuelve (conocimiento congelado, trazabilidad) y por qué es el patrón dominante en gobierno.
- **El árbol de decisión prompting vs. RAG vs. fine-tuning vs. agente:** dado un caso, elegir y justificar.
- Evaluación de LLMs: por qué BLEU/ROUGE/perplejidad no bastan; contaminación y saturación de benchmarks; LLM-as-judge y sus sesgos; alucinación y calibración; los cinco mínimos de una evaluación de despliegue público.

### 5. Diseño aplicado (transversal, Parte C)

Al menos una pregunta larga planteará un problema real de política pública y pedirá diseñar la solución **en papel**: qué familia de arquitectura, qué estrategia de pre-entrenamiento o transfer learning, qué protocolo de validación (¡split por unidad y por tiempo, no por imagen o por renglón!), qué métricas (incluyendo por subgrupo), y qué modos de falla previsibles. También pedirá **criticar una propuesta** de un equipo o de un agente de IA: señalar los errores estructurales y proponer la corrección. Contextos tipo:

- Mapeo de pobreza con imágenes satelitales y pocas etiquetas (cf. Jean et al. 2016).
- Clasificación de textos legislativos o administrativos (cf. Chalkidis et al. 2019).
- Predicción de demanda de un servicio público a partir de series temporales.
- Elegir entre prompting, RAG y fine-tuning para una tarea de una dependencia.

---

## Ejemplos del tipo de pregunta

Estas **no** son las preguntas del examen, pero son del mismo estilo y dificultad.

| Tipo | Ejemplo |
|---|---|
| Verdadero/falso | "Gradient clipping resuelve tanto los gradientes que explotan como los que se desvanecen." Justifica. |
| Cálculo | "Entrada $64\times64\times3$, 16 filtros $5\times5$, padding 2, stride 2: tamaño de salida, número de parámetros y receptive field tras una segunda capa $3\times3$." |
| Cálculo | "Con esta query y estas tres keys/values, calcula los pesos de atención y la salida. ¿Qué cambia si permutas las keys?" |
| Mecanismo | "¿Qué valores de los gates hacen que una LSTM conserve información durante muchos pasos, y por qué eso ayuda al gradiente?" |
| Ablación mental | "¿Qué pasa con el error de entrenamiento y con la generación si entrenas un decoder sin máscara causal?" |
| Trade-off | "Dos modelos de lenguaje reportan perplejidad 60 y 95 con tokenizadores distintos. ¿Cuál es mejor? ¿Qué tendrías que comparar?" |
| Diseño aplicado | "600 localidades con encuesta, 50,000 imágenes satelitales sin etiqueta: propón arquitectura, pre-entrenamiento, validación y riesgos." |
| Crítica | "Un agente propone entrenar una CNN desde cero con 600 imágenes y split aleatorio por imagen. Señala dos problemas y corrígelos." |

---

## Cómo estudiar

1. **Resuelve las Tareas 6–8** completas. Los enunciados y las soluciones de la parte práctica están en el repo. Las preguntas del examen siguen ese estilo.
2. Para los notebooks de CNN, RNN y Transformers: antes de correr cada celda, **predice** qué va a salir y por qué. En el de Transformers, las ablaciones (quitar la máscara causal, quitar la codificación posicional) son material de examen directo.
3. Practica la aritmética de convoluciones y de atención con números inventados hasta que salga sin pensar. Son cinco minutos del examen y separan a quien entiende de quien memorizó.
4. Reproduce sin ver las slides: la derivación de la convolución desde la capa completamente conectada, la actualización de la LSTM y por qué preserva el gradiente, y el flujo completo de un bloque Transformer.
5. Para la Parte C, lee al menos el abstract y las figuras de Jean et al. (2016) y Chalkidis et al. (2019), y ten claro el árbol de decisión prompting / RAG / fine-tuning / agente.
6. Puedes usar un LLM para estudiar: pídele que te ponga preguntas de este temario y que **no te dé la respuesta hasta que tú des la tuya**. El examen es a mano y sin él.
