# Aprendizaje profundo y sus usos en las políticas públicas

---

## Objetivos del curso
- Entender las arquitecturas centrales del aprendizaje profundo (CNNs, RNNs, Transformers, etc.).  
- Implementar modelos de aprendizaje profundo de extremo a extremo usando **PyTorch**.  
- Evaluar de manera crítica modelos de aprendizaje profundo (desempeño, equidad, impacto social).  
- Aplicar aprendizaje profundo a problemas reales (análisis de texto, imágenes, series temporales, etc.).  
- Diseñar sistemas robustos, transparentes y explicables — más allá de “cajas negras”.

---

## Prerrequisitos (qué deberías saber antes)
- **Programación en Python:** estructuras, funciones, clases y OOP básica.  
- **ML básico:** funciones de pérdida, optimización (GD/SGD), evaluación y métricas.  
- **Flujos de trabajo de ciencia de datos:** limpieza, split train/val/test, versionado de código y datos.

---

## Roadmap (resumen semanal)
- **Semanas 1–4 — Fundamentos**
  - Repaso ML básico  
  - Redes neuronales superficiales y profundas  
  - Entrenamiento (optimización, inicialización, performance)  
  - Regularización
- **Semanas 5–7 — CNNs y RNNs** — *Examen 1 (~semana 6)*  
- **Semanas 8–10 — Transformers**  
- **Semana 11 — Por qué funciona DL / Fairness / Ética / preguntas abiertas** — *Examen 2*  
- **Semana 12 — Presentaciones y entrega del proyecto de colaboración con IA**

---

## Evaluación
- **2 exámenes en clase (teoría, sin código)** — **50%** (25% cada uno).  
  - **Examen 1 (~semana 6):** fundamentos — repaso de ML, redes neuronales, funciones de pérdida, optimización, backpropagation, inicialización y regularización.  
  - **Examen 2 (~semana 11):** arquitecturas — CNNs, RNNs, atención y Transformers, con preguntas de aplicación a políticas públicas.  
  - Temarios detallados en `Examenes/`. Las preguntas siguen el estilo de las tareas de práctica: derivaciones a mano, diagnóstico de fallas, diseño de pérdidas y juicios de trade-off — **no se escribe código en los exámenes**.
- **Presentación de paper (en parejas)** — **35%**.  
  - Papers asignados aleatoriamente. Formato y rúbrica en guía aparte (`Tareas/GuiaPresentaciones`).
- **Proyecto de colaboración con IA** — **15%**.  
  - Diriges a un agente de IA (Claude, ChatGPT, etc.) para ejecutar un experimento de deep learning. Se califica la calidad de tu especificación, tu verificación del trabajo del agente y tu interpretación de resultados — no el código. Guía en `Tareas/ProyectoColaboracionIA.md`.

### Política de uso de IA
- **Tareas de práctica (no calificadas):** las 8 tareas del curso están disponibles en `Tareas/` como material de práctica y preparación para los exámenes, con soluciones de la parte práctica en `Soluciones/`. Puedes usar LLMs libremente en ellas — pero recuerda que los exámenes son en clase, con lápiz y papel, y evalúan que *tú* entiendas.
- **Presentaciones y proyecto:** el uso de LLMs está permitido y (en el proyecto) es el objeto mismo de la evaluación, **pero es obligatorio entregar la conversación/registro completo (chat)** junto con los entregables.
  - **Para las presentaciones** se permite usar LLMs **para entender** o preparar ideas, **pero** es **obligatorio** entregar la conversación/registro (chat) junto con los entregables.

---

## Papers a presentar (lista)
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

---

## Libros y lecturas recomendadas
- Prince — *Unifying Deep Learning* (texto base del curso)  
- Goodfellow, Bengio & Courville — *Deep Learning*  
- Zhang et al. — *Dive into Deep Learning (d2l)*  
- Chollet — *Deep Learning with Python*

---

## Papers de DL en políticas públicas (opcional / lectura recomendada)
**Medición de pobreza con imágenes satelitales**
- Jean et al. (2016) — *Combining satellite imagery and machine learning to predict poverty* (Science)  
- Yeh et al. (2020) — *Using publicly available satellite imagery and deep learning to understand economic well-being in Africa* (Nature Communications)  
- Huang, Hsiang & Gonzalez-Navarro (2021) — *Using Satellite Imagery and Deep Learning to Evaluate the Impact of Anti-Poverty Programs* (NBER)  
- Rolf et al. (2021) — *A generalizable and accessible approach to machine learning with global satellite imagery* (MOSAIKS, Nature Communications)  
- Babenko et al. (2017) — *Poverty Mapping Using Convolutional Neural Networks* (arXiv)

**Análisis de textos legislativos**
- Korn & Newman (2020) — *A Deep Learning Model to Predict Congressional Roll Call Votes From Legislative Texts*  
- Chalkidis et al. (2019) — *Large-Scale Multi-Label Text Classification on EU Legislation*  
- Wei et al. (2019) — *Empirical Study of Deep Learning for Text Classification in Legal Document Review*

**LLMs como herramienta de investigación y medición (era LLM)**
- Gilardi, Alizadeh & Kubli (2023) — *ChatGPT outperforms crowd workers for text-annotation tasks* (PNAS)  
- Ziems et al. (2024) — *Can Large Language Models Transform Computational Social Science?* (Computational Linguistics)  
- Dell (2025) — *Deep Learning for Economists* (Journal of Economic Literature)  
- Ouyang et al. (2022) — *Training language models to follow instructions with human feedback* (InstructGPT/RLHF)  
- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*

---

## Notas prácticas (reproducibilidad y ejecución)
- **Notebooks:** todos los notebooks de clase estarán en **PyTorch** y contendrán instrucciones reproducibles: semilla fija, detección de device (CPU/GPU/MPS), y opciones para checkpoints.  
- **Si no tienes GPU:** diseña experimentos pequeños (submuestras, redes reducidas) que permitan ilustrar los fenómenos estudiados.  

