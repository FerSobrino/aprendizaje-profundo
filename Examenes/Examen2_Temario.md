# Examen 2 — Arquitecturas: CNNs, RNNs y Transformers (~semana 11)

**Formato:** en clase, individual, lápiz y papel. Sin computadora, sin código.
**Peso:** 25% de la calificación final.
**Material permitido:** una hoja de fórmulas (carta, por ambos lados) escrita a mano por ti.
**Cobertura:** slides *CNN*, *RNN* y *Transformers*; Tareas de práctica 6–8. Los fundamentos del Examen 1 se asumen conocidos (pueden aparecer como parte de una pregunta, no como pregunta principal).

Igual que en el Examen 1: nada de escribir código. Se evalúa que entiendas **cómo y por qué funcionan** las arquitecturas, y que puedas razonar sobre cuál usar, qué puede fallar y cómo verificar lo que un agente de IA construya por ti.

---

## 1. Redes convolucionales (Tarea 6)

- Motivación: por qué una red totalmente conectada falla en imágenes de 224×224×3 (conteo de parámetros como argumento).
- **Invarianza vs. equivarianza** a traslaciones: definiciones formales, cuál aporta la convolución y cuál el pooling; dar un ejemplo donde la invarianza traslacional es una **mala** idea.
- Derivar la convolución desde la capa general imponiendo invarianza traslacional + localidad; convolución vs. correlación cruzada (y qué implementan realmente los frameworks); demostrar que la convolución es conmutativa.
- Aritmética de convoluciones: fórmulas de tamaño de salida con padding y stride; conteo de parámetros de una capa con $c_i$ canales de entrada, $c_o$ de salida y kernel $k\times k$.
- Costo computacional: ¿por qué factor crece el cómputo si duplicas $c_i$ y $c_o$ a la vez? ¿beneficios computacionales y estadísticos de stride > 1?
- **Campos receptivos**: calcular el campo receptivo tras apilar capas; por qué la profundidad los agranda.
- Convoluciones 1×1: qué hacen y para qué sirven.
- Pooling (max/avg): qué aporta; por qué un "softmax pooling" podría ser mala idea.
- Convoluciones sobre texto: ¿tiene sentido? ¿qué problemas específicos del lenguaje aparecen?
- Sesgo inductivo: pesos compartidos + localidad como regularización estructural; por qué la CNN gana a la MLP con *menos* parámetros (eficiencia muestral, "accuracy por parámetro").
- Transfer learning y fine-tuning en visión: cuándo congelar capas, cuándo reentrenar todo.
- Caso de estudio AlexNet: qué combinación de ingredientes lo hizo funcionar (datos, GPU, ReLU, dropout, aumento de datos).

## 2. Redes recurrentes y modelado de secuencias (Tarea 7)

- Modelos autoregresivos: ventana fija vs. variable latente; ¿cuándo se necesita un modelo autoregresivo latente?
- Estacionariedad (estricta vs. débil) y por qué importa para entrenar con secuencias.
- Pipeline de texto: tokenización, vocabulario, `<unk>`, `min_freq` como trade-off tamaño-de-vocabulario/OOV; ley de Zipf; n-gramas y suavizamiento de Laplace (y sus fallas).
- **Perplejidad**: definición, interpretación ("número efectivo de opciones"), por qué no es comparable entre tokenizaciones distintas.
- La recurrencia $H_t = \phi(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$: parámetros compartidos en el tiempo; dimensión de la salida por paso para predicción a nivel de caracteres.
- Minibatches con secuencias: qué se rompe si cada ejemplo es una oración completa y cómo se arregla (padding, truncamiento, empaquetado).
- **BPTT**: de dónde salen los gradientes que explotan/se desvanecen en secuencias largas; gradient clipping (argumento de Lipschitz) y alternativas.
- **LSTM**: las cuatro ecuaciones de compuertas; por qué se aplica tanh de nuevo al calcular $h_t$ si $\tilde{c}_t$ ya está en $[-1,1]$; el camino directo del estado de celda como solución al desvanecimiento.
- **GRU**: compuertas de reset y update; qué pasa si implementas solo una de las dos; cuándo preferir GRU sobre LSTM.
- RNNs bidireccionales: cuándo sí y cuándo no (¿por qué no para generación?).
- Juicio de despliegue: con datos limitados y recursos moderados, ¿qué arquitectura eliges para producción y por qué? (calidad vs. costo por época).

## 3. Atención y Transformers (Tarea 8)

- El cuello de botella del vector de contexto en encoder-decoder recurrente: por qué motivó la atención.
- **Q/K/V**: la analogía de la biblioteca; atención como agrupamiento ponderado $\sum_i \alpha(q,k_i)\,v_i$.
- Atención de producto punto escalado: ¿por qué dividir entre $\sqrt{d_k}$ y qué se rompe sin eso? Atención aditiva (Bahdanau) como alternativa: ¿cuándo tendría ventaja?
- **Multi-head**: qué gana sobre una sola cabeza; cómo diseñarías un experimento para medir la importancia de cada cabeza (poda de cabezas).
- Auto-atención es invariante al orden → **codificación posicional**: sinusoidal vs. aprendida (generalización a secuencias largas, parámetros, interpretabilidad); problemas al apilar muchas capas.
- **Máscaras**: de padding, causal y de segmento; qué hace exactamente la máscara causal y qué degenera en la generación si la quitas (ablación de la Tarea 8).
- El bloque Transformer completo: residuales + LayerNorm, MLP, embeddings; poder narrar el flujo encoder→decoder.
- Escalamiento: costo en cómputo y memoria de la atención en la longitud de secuencia y en $d$; por qué las secuencias muy largas son un reto; ancho de banda de memoria.
- **Objetivos de preentrenamiento**: MLM+NSP (BERT) vs. LM causal (GPT) vs. denoising seq2seq (T5/BART); dar un caso donde MLM crea una ambigüedad que un LM causal no; limitaciones de los modelos decoder-only para tareas seq2seq con entrada completa disponible.
- Fine-tuning: dónde agregar capas para clasificación de texto y por qué; panorama de estrategias eficientes (adapters, LoRA, prompt tuning) a nivel conceptual.
- LLMs y prompting: zero-shot, few-shot, chain-of-thought; qué es cada uno y cuándo ayuda.
- Métricas de NLP: exactitud/F1, BLEU, ROUGE, perplejidad — y sus límites para evaluar LLMs modernos.

## 4. De LLM base a asistente (parte final del deck de Transformers)

- Por qué un LM base no es un asistente; qué agrega el instruction tuning (SFT).
- **RLHF** a nivel conceptual: comparaciones humanas → modelo de recompensa → optimización; qué simplifica DPO.
- **LoRA**: la idea de la corrección de rango bajo y por qué hace barato el fine-tuning.
- **RAG**: qué problema resuelve (conocimiento congelado, trazabilidad de fuentes), cómo funciona a grandes rasgos, y por qué es el patrón dominante en gobierno.
- **Agentes**: el ciclo razonar→actuar→observar; qué cambia en las habilidades que importan.
- El árbol de decisión prompting vs. RAG vs. fine-tuning vs. agente: dado un caso, elegir y justificar.
- Evaluación de LLMs: por qué BLEU/ROUGE/perplejidad no bastan; contaminación de benchmarks; LLM-as-judge y sus sesgos; alucinación y calibración; los 5 mínimos de una evaluación de despliegue público.

## 5. Aplicación a políticas públicas (transversal)

Al menos una pregunta larga planteará un problema real de política pública y pedirá diseñar la solución **en papel**: elección de arquitectura, estrategia de preentrenamiento/transfer learning, protocolo de validación, métricas (incluyendo por subgrupo), y modos de falla previsibles. Contextos tipo:
- Mapeo de pobreza con imágenes satelitales y pocas etiquetas (cf. Jean et al. 2016).
- Clasificación multi-etiqueta de textos legislativos (cf. Chalkidis et al. 2019).
- Predicción de demanda de un servicio público a partir de series temporales.
- Auditoría de un sistema de riesgo tipo COMPAS: qué mirarías y qué métricas pedirías.

---

## Tipos de pregunta

| Tipo | Ejemplo |
|---|---|
| Cálculo | "Kernel 5×5, padding 2, stride 2, entrada 64×64: ¿tamaño de salida? ¿campo receptivo tras dos capas?" |
| Mecanismo | "Escribe las ecuaciones de las compuertas del LSTM y explica cuál resuelve el desvanecimiento del gradiente y por qué." |
| Ablación mental | "¿Qué pasa con la generación de un GPT si quitas la máscara causal? ¿Por qué?" |
| Trade-off | "¿Cuándo preferirías atención aditiva sobre producto punto escalado?" |
| Diseño aplicado | "500 municipios etiquetados, imágenes satelitales de todo el país: propón arquitectura, preentrenamiento, validación y riesgos." |
| Verificación | "Un agente de IA te entregó este resumen de arquitectura para tu problema. Señala dos decisiones cuestionables y qué preguntarías." |

## Cómo estudiar

1. Resuelve la teoría de las **Tareas 6–8** (enunciados y soluciones publicados) — el examen sigue su estilo directamente.
2. Corre (o pide a un LLM que corra) las ablaciones del notebook de Transformers §10 y las comparaciones RNN/LSTM/GRU de la Tarea 7; lo evaluable es que puedas **predecir y explicar** los resultados, no producirlos.
3. Repasa las preguntas de defensa de las presentaciones (`PreguntasPresentaciones`): son el nivel de profundidad al que apunta este examen.
4. Lee al menos el abstract y las figuras de los papers de política pública del README; las preguntas aplicadas usan esos contextos.
