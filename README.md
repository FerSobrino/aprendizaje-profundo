# Aprendizaje profundo y sus usos en las políticas públicas

**Escuela de Gobierno y Transformación Pública, Tec de Monterrey**

Profesora: Fernanda Sobrino (fersobrinono@tec.mx)

Entregas en Canvas

## Filosofía del curso

Los agentes de IA ya construyen y entrenan redes neuronales mejor y
más rápido que la mayoría de los humanos. Eso no hace que este curso
sobre: lo hace más importante. El agente hace *exactamente* lo que le
pides — si no sabes pedir una división train/val/test, pesos por clase
o métricas por subgrupo, no las vas a obtener; y si no sabes leer una
curva de pérdida, no vas a notar cuando el resultado esté mal. Este
curso enseña las dos cosas que la IA no pone: **entender cómo funciona
el aprendizaje profundo** y **verificar con escepticismo**.

Por eso:

1. **El curso es de teoría, no de código.** Derivamos las pérdidas
   desde máxima verosimilitud, hacemos backprop a mano y entendemos
   cada arquitectura desde sus supuestos. Los exámenes son a lápiz y
   papel. El código lo escribe el agente; el criterio lo pones tú.
2. **No prohibimos la IA.** Las tareas de práctica son libres (y no
   calificadas); en el proyecto final, dirigir a un agente **es** la
   habilidad evaluada, con transcripción obligatoria.
3. **Cubrimos hasta la frontera.** El curso llega hasta LLMs, RLVR y
   modelos de razonamiento, MoE, RAG y una sesión completa sobre
   dirigir, supervisar y auditar agentes — con sus modos de falla
   documentados.

El estándar del curso: *no "¿qué respondió el agente?", sino "¿cómo sé
si está bien?"*

## Evaluación

| Componente | Peso | Detalle |
|---|---|---|
| Exámenes en clase (2) | 50% | Escritos, sin computadora (25% c/u) |
| Presentación de paper (parejas) | 35% | IA permitida para entender, con transcripción |
| Proyecto de colaboración con IA | 15% | Dirigir a un agente; se califica especificación, verificación e interpretación |

- **Exámenes**: en clase, individuales, lápiz y papel. No piden
  escribir código: piden derivar, predecir, diagnosticar y justificar.
  Temarios en [Examenes/](Examenes/). Examen 1 (~semana 6):
  fundamentos. Examen 2 (~semana 11): arquitecturas, LLMs y agentes.
- **Presentación**: papers asignados aleatoriamente, en parejas.
  Formato y rúbrica en
  [Evaluacion/GuiaPresentaciones.Rmd](Evaluacion/GuiaPresentaciones.Rmd).
- **Proyecto**: diriges a un agente de IA para ejecutar un experimento
  de deep learning. Se califica tu especificación, tu verificación del
  trabajo del agente y tu interpretación — no el código. Guía en
  [Evaluacion/ProyectoColaboracionIA.md](Evaluacion/ProyectoColaboracionIA.md).
- **Tareas de práctica (no calificadas)**: las 8 tareas en
  [tareas/](tareas/) son la preparación directa para los exámenes, con
  soluciones de la parte práctica en [soluciones/](soluciones/).

## Temario semanal

| Semana | Temas | Slides | Notebook | Práctica | Exámenes |
|---|---|---|---|---|---|
| 1 | Introducción y política de IA · Repaso de ML | [Intro](slides/Intro.pdf) · [Repaso](slides/Repaso.pdf) | [Tensores](notebooks/Intro_Tensores.ipynb) · [PyTorch](notebooks/Intro_Pytorch.ipynb) | T1 | |
| 2 | Redes neuronales superficiales y profundas | [NN](slides/NN.pdf) | [MNIST](notebooks/Clase_RN_MNIST.ipynb) | T2 | |
| 3 | Entrenamiento I: pérdidas y optimización | [Entrenamiento](slides/Entrenamiento.pdf) | [Optimizadores](notebooks/Optimizadores.ipynb) | T3 | |
| 4 | Entrenamiento II: backprop, inicialización, performance | [Entrenamiento 2](slides/Entrenamiento2.pdf) | [Diagnósticos](notebooks/Entrenamiento2.ipynb) | T4 | |
| 5 | Regularización | [Regularización](slides/Regularizacion.pdf) | [Regularización](notebooks/Regularizacion.ipynb) | T5 | |
| 6 | CNNs I | [CNN](slides/CNN.pdf) | [CNN](notebooks/NotebookCNN.ipynb) | | **Examen 1** |
| 7 | CNNs II y aplicaciones (satélite y pobreza) | [CNN](slides/CNN.pdf) | [CNN](notebooks/NotebookCNN.ipynb) | T6 | |
| 8 | RNNs y modelos de secuencias | [RNN](slides/RNN.pdf) | [RNN](notebooks/NotebookRNN.ipynb) | T7 | |
| 9 | Atención y Transformers | [Transformers](slides/Transformers.pdf) | [Mini-GPT](notebooks/Transformers.ipynb) | T8 | |
| 10 | LLMs: post-entrenamiento, RAG · Agentes | [Transformers](slides/Transformers.pdf) · [Agentes](slides/Agentes.pdf) | | | |
| 11 | ¿Por qué funciona DL? · Ética y gobernanza | [Ética](slides/EticayPreguntas.pdf) | | | **Examen 2** |
| 12 | Presentaciones · entrega del proyecto | | | | |

## Política de IA (resumen)

La política completa está en [PoliticaIA.md](PoliticaIA.md). En corto:

- En **tareas de práctica y proyecto**: cualquier LLM o agente está
  permitido. En el proyecto (y en la preparación de la presentación)
  entregas la transcripción completa; sin transcripción, no se
  califica.
- En **exámenes**: sin IA — a lápiz y papel.
- Eres responsable de todo lo que entregas: "lo escribió el agente" no
  es defensa de un resultado incorrecto.

## Materiales

Todo el material vive en este repositorio (público); los avisos y las
entregas, en Canvas. Recomendado: copiar la carpeta completa una sola
vez — guía paso a paso (Colab o instalación local) en
[tech-help/configuracion_entorno.md](tech-help/configuracion_entorno.md).

- `slides/` — presentaciones (.Rmd → PDF)
- `notebooks/` — notebooks de clase (PyTorch) para seguir en vivo
- `tareas/` — las 8 tareas de práctica (no calificadas)
- `soluciones/` — soluciones de la parte práctica de las tareas
- `Examenes/` — temarios de los dos exámenes
- `Evaluacion/` — guía de presentaciones y del proyecto de colaboración con IA
- `images/` — figuras usadas por las slides

## Bibliografía

- Prince — *Understanding Deep Learning* (texto base; gratis en udlbook.github.io)
- Xiao & Zhu — *Foundations of Large Language Models* (arXiv 2501.09223; complemento LLM gratuito)
- Goodfellow, Bengio & Courville — *Deep Learning*
- Zhang et al. — *Dive into Deep Learning (d2l)*
- Chollet — *Deep Learning with Python*

## Papers a presentar

1. AlexNet — Krizhevsky, Sutskever & Hinton (2012)
2. ResNet — He et al. (2015)
3. Batch Normalization — Ioffe & Szegedy (2015)
4. Attention is All You Need (Transformers) — Vaswani et al. (2017)
5. BERT — Devlin et al. (2018)
6. Variational Autoencoders (VAE) — Kingma & Welling (2013)
7. GANs — Goodfellow et al. (2014)
8. Double Descent — Belkin et al. (2019) / Nakkiran et al. (2019)

**Adicionales (era LLM, para grupos grandes o como alternativa):**

9. Scaling Laws / Chinchilla — Hoffmann et al. (2022)
10. InstructGPT (RLHF) — Ouyang et al. (2022)
11. LoRA — Hu et al. (2021)
12. RAG (Retrieval-Augmented Generation) — Lewis et al. (2020)
13. DeepSeek-R1 (RL con recompensas verificables) — DeepSeek-AI (2025)

## Lecturas de DL en políticas públicas (opcionales)

**Medición de pobreza con imágenes satelitales**
- Jean et al. (2016) — *Combining satellite imagery and machine learning to predict poverty* (Science)
- Yeh et al. (2020) — *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa* (Nature Communications)
- Huang, Hsiang & Gonzalez-Navarro (2021) — *Using Satellite Imagery and Deep Learning to Evaluate the Impact of Anti-Poverty Programs* (NBER)
- Rolf et al. (2021) — *A generalizable and accessible approach to machine learning with global satellite imagery* (MOSAIKS, Nature Communications)

**Análisis de textos legislativos**
- Korn & Newman (2020) — *A Deep Learning Model to Predict Congressional Roll Call Votes From Legislative Texts*
- Chalkidis et al. (2019) — *Large-Scale Multi-Label Text Classification on EU Legislation*
- Wei et al. (2019) — *Empirical Study of Deep Learning for Text Classification in Legal Document Review*

**Agentes: dirigir, supervisar y auditar**
- Microsoft (2026) — *Taxonomy of Failure Modes in Agentic AI Systems v2.0* (la lectura central del deck de agentes)
- Abdurahman et al. (2025) — *A Primer for Evaluating Large Language Models in Social-Science Research* (AMPPS)
- Amnesty International (2021) — *Xenophobic Machines* (el caso holandés de las guarderías)

**LLMs como herramienta de investigación y medición**
- Gilardi, Alizadeh & Kubli (2023) — *ChatGPT outperforms crowd workers for text-annotation tasks* (PNAS)
- Ziems et al. (2024) — *Can Large Language Models Transform Computational Social Science?* (Computational Linguistics)
- Dell (2025) — *Deep Learning for Economists* (Journal of Economic Literature)

---
