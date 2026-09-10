---
title: "Examen de práctica 1: Fundamentos"
subtitle: "Aprendizaje Profundo y sus usos en Políticas Públicas · 2026"
lang: es
geometry: margin=2.3cm
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb}
  - \usepackage{booktabs}
  - \setlength{\parskip}{4pt}
---

Este examen de práctica tiene el mismo formato que el examen real: tendrá 3 secciones y, al final, un formulario con las distribuciones de probabilidad, para que no tengas que memorizarlas. Las respuestas están al final del documento. Recuerda que tendrás 40 minutos para el examen real.

---

# Parte A — Conceptos rápidos

Escribe V o F y justifica en una o dos líneas.

A1. Gradient descent (no estocástico) con learning rate fijo siempre llega al mínimo global de la pérdida de una red neuronal.

A2. En regresión lineal, inicializar todos los parámetros en cero funciona sin problema.

A3. La binary cross-entropy es lo que se obtiene al asumir una distribución Bernoulli para $y$ y conectar la red con una sigmoide.

A4. Aumentar el número de observaciones de entrenamiento reduce el sesgo del modelo.

A5. En SGD, la regularización $L_2$ equivale a multiplicar los pesos por $(1-\eta\lambda)$ en cada actualización, y por eso se llama weight decay.

A6. En inferencia, BatchNorm normaliza usando la media y la varianza del mini-batch que se está evaluando.

A7. Una red ReLU superficial con un input y $D$ unidades ocultas produce a lo más $D+1$ regiones lineales.

A8. La inicialización He, $\sigma^2_\Omega=2/D_h$, está diseñada para redes con activación tanh.

A9. Corremos SGD con mini-batches de tamaño 50 sobre 1,000 observaciones durante 600 iteraciones. Eso equivale a 30 épocas.

A10. En el fenómeno de double descent, después del umbral de interpolación el error de test solo puede subir.

---

# Parte B — Desarrollo

## B1. Trazar una red a mano

Red superficial con activación ReLU, un input y dos unidades ocultas:

$$y=\phi_0+\phi_1h_1+\phi_2h_2,\qquad h_1=\text{ReLU}(x),\qquad h_2=\text{ReLU}(x-2),$$
$$\phi_0=1,\quad\phi_1=2,\quad\phi_2=-3.$$

a) Calcula $h_1,h_2$ y $y$ para $x=-1$, $x=1$ y $x=3$.

b) ¿Dónde están las articulaciones? ¿Cuántas regiones lineales hay y cuál es la pendiente en cada una?

c) Si la activación fuera la identidad, ¿qué función obtendrías?

d) Si multiplicas los parámetros de $h_2$ por 4 (es decir, $h_2=\text{ReLU}(4x-8)$) y divides $\phi_2$ entre 4, ¿cambia $y(x)$? ¿Por qué?

## B2. Diseñar una función de pérdida

Un centro de atención telefónica quiere predecir el **tiempo de espera** $y>0$ (en minutos) de cada llamada a partir de variables $x$ como hora, día y número de operadores. Usará una red $f(x,\phi)\in\mathbb R$.

a) Consulta el formulario. ¿Qué distribución propones para $y$ y por qué no es adecuada la Normal?

b) La distribución que elegiste tiene un parámetro con una restricción. ¿Cómo conectas $f(x_i,\phi)$ con ese parámetro?

c) Escribe la negative log-likelihood y simplifícala en términos de $f(x_i,\phi)$ y $y_i$.

d) Ahora el centro quiere predecir si la llamada **se abandona** antes de ser atendida (sí/no). ¿Qué cambia en cada paso de la receta?

## B3. Backpropagation e inicialización

$$f_0=\beta_0+\omega_0x,\qquad h_1=\text{ReLU}(f_0),\qquad f_1=\beta_1+\omega_1h_1,\qquad \ell=(f_1-y)^2.$$

Valores: $x=1$, $y=2$, $\beta_0=0.5$, $\omega_0=1$, $\beta_1=-1$, $\omega_1=3$.

a) Forward pass: $f_0,h_1,f_1,\ell$.

b) Backward pass: las cuatro parciales $\partial\ell/\partial\omega_1$, $\partial\ell/\partial\beta_1$, $\partial\ell/\partial\omega_0$, $\partial\ell/\partial\beta_0$. Escribe la cadena que usas.

c) Si $\omega_0$ fuera $-1$ (todo lo demás igual), ¿qué valdrían $\partial\ell/\partial\omega_0$ y $\partial\ell/\partial\beta_0$? ¿Cómo se llama esto?

d) ¿Qué pasa si inicializamos todos los pesos y biases de una red profunda exactamente en cero? ¿Por qué en regresión lineal eso no es problema?

---

# Parte C — Diagnóstico

Una dependencia entrena redes para clasificar **solicitudes ciudadanas** en 4 categorías a partir de 20 variables administrativas. Tiene 8,000 solicitudes etiquetadas, split 70/15/15. Después de 40 épocas (cross-entropy; el azar da 25% de accuracy):

| Exp | Modelo | LR | Train loss (ép. 40) | Val loss (ép. 40) | Val acc | Observaciones |
|--|----------------|-----|------|------|-----|--------------------------------|
| 1 | 1 capa, 8 unidades | 0.001 | 0.90 | 0.92 | 60% | Ambas pérdidas bajaron y se estancaron desde la época 15. |
| 2 | 4 capas, 256 unidades | 0.001 | 0.05 | 1.40 | 68% | Val loss mínima (0.75, acc 74%) en la época 10; después sube. |
| 3 | 4 capas, 256 unidades | 0.050 | 1.38 | 1.39 | 26% | La pérdida oscila fuertemente desde el inicio y no baja. |

a) Diagnostica cada experimento: ¿sesgo, varianza u optimización? ¿Qué evidencia lo sostiene?

b) Para el experimento 2, propón dos intervenciones y predice el efecto en train y val.

c) El equipo dice: "probamos 15 combinaciones de hiperparámetros y reportamos la mejor accuracy en test: 74%". ¿Qué está mal y qué debieron hacer?

d) Alguien propone "para el experimento 1, agreguen dropout 0.5". ¿Tiene sentido? Justifica.

---

# Formulario: distribuciones de probabilidad

| Distribución | Soporte de $y$ | Parámetros | $P(y)$ o $p(y)$ | $E[y]$ |
|-----------|---------|-------------------|--------------------------------|-------------|
| Bernoulli | $\{0,1\}$ | $\lambda\in[0,1]$ | $\lambda^{y}(1-\lambda)^{1-y}$ | $\lambda$ |
| Binomial | $\{0,\dots,m\}$ | $m$ fijo, $\lambda\in[0,1]$ | $\binom{m}{y}\lambda^{y}(1-\lambda)^{m-y}$ | $m\lambda$ |
| Categórica | $\{1,\dots,K\}$ | $\lambda_k\ge0$, $\sum_k\lambda_k=1$ | $\lambda_y$ | — |
| Geométrica | $\{0,1,2,\dots\}$ | $\lambda\in(0,1]$ | $(1-\lambda)^{y}\lambda$ | $(1-\lambda)/\lambda$ |
| Poisson | $\{0,1,2,\dots\}$ | $\lambda>0$ | $\dfrac{\lambda^{y}e^{-\lambda}}{y!}$ | $\lambda$ |
| Uniforme | $[a,b]$ | $a<b$ | $\dfrac{1}{b-a}$ | $(a+b)/2$ |
| Normal | $\mathbb{R}$ | $\mu\in\mathbb{R}$, $\sigma^2>0$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\dfrac{(y-\mu)^2}{2\sigma^2}\right)$ | $\mu$ |
| Laplace | $\mathbb{R}$ | $\mu\in\mathbb{R}$, $b>0$ | $\dfrac{1}{2b}\exp\!\left(-\dfrac{|y-\mu|}{b}\right)$ | $\mu$ |
| Exponencial | $[0,\infty)$ | $\lambda>0$ | $\lambda e^{-\lambda y}$ | $1/\lambda$ |
| Gamma | $(0,\infty)$ | $\alpha>0$, $\beta>0$ | $\dfrac{\beta^{\alpha}}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y}$ | $\alpha/\beta$ |
| Beta | $[0,1]$ | $\alpha>0$, $\beta>0$ | $\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}y^{\alpha-1}(1-y)^{\beta-1}$ | $\dfrac{\alpha}{\alpha+\beta}$ |

\newpage

# Respuestas orientativas

No son las únicas redacciones válidas. Lo que se califica es el razonamiento.

## Parte A

A1. **F.** La pérdida de una red no es convexa: hay mínimos locales, puntos silla y regiones planas; GD con lr fijo puede quedarse en cualquiera. La garantía de mínimo global es solo para funciones convexas.

A2. **V.** En regresión lineal no hay unidades ocultas que romper: el gradiente en cero no es nulo ni simétrico, y GD avanza. El problema de la inicialización en cero aparece con capas ocultas (ver B3d).

A3. **V.** $P(y|\lambda)=\lambda^y(1-\lambda)^{1-y}$ con $\lambda=\sigma(f(x,\phi))$; la NLL es $-\sum_i[y_i\log\lambda_i+(1-y_i)\log(1-\lambda_i)]$, que es la BCE.

A4. **F.** Más datos reduce la **varianza** (menos dependencia de la muestra particular). El sesgo depende de la familia de modelos: si la familia no puede representar la relación, más datos no lo arregla.

A5. **V.** $\nabla_w\frac{\lambda}{2}\|w\|^2=\lambda w$; la actualización queda $w\leftarrow(1-\eta\lambda)w-\eta\nabla_w L$: el peso se encoge un poco en cada paso.

A6. **F.** En inferencia usa estadísticas acumuladas (promedios móviles) durante el entrenamiento; si usara las del batch, la predicción de un ejemplo dependería de con quién le tocó ir en el batch.

A7. **V.** Cada unidad aporta una articulación donde su preactivación cruza cero; $D$ articulaciones dividen la recta en a lo más $D+1$ segmentos.

A8. **F.** He es para ReLU: como ReLU apaga la mitad de las unidades, $E[h^2]\approx\frac12\sigma_f^2$ y se necesita $\sigma^2_\Omega=2/D_h$ para conservar la varianza. Para tanh/sigmoide se usa Xavier, $2/(D_{in}+D_{out})$.

A9. **V.** Una época son $1000/50=20$ iteraciones; $600/20=30$ épocas.

A10. **F.** Precisamente lo sorprendente del double descent es que, en el régimen sobreparametrizado, el error de test puede volver a **bajar** aunque el de entrenamiento ya sea cero.

## Parte B

**B1.** a) $x=-1$: $h_1=0,h_2=0,y=1$. $x=1$: $h_1=1,h_2=0,y=3$. $x=3$: $h_1=3,h_2=1,y=1+6-3=4$.
b) Articulaciones en $x=0$ y $x=2$: tres regiones. Pendientes: $x<0$: ninguna unidad activa, pendiente $0$; $0<x<2$: solo $h_1$, pendiente $\phi_1\cdot1=2$; $x>2$: ambas, $2-3=-1$. Comprobación: $y(0)=1$, $y(2)=5$, $y(3)=4$.
c) Con $a(z)=z$, $y=1+2x-3(x-2)=7-x$: una recta, una sola región. Sin no linealidad la red colapsa a regresión lineal.
d) No cambia: $\frac{\phi_2}{4}\text{ReLU}(4(x-2))=\phi_2\text{ReLU}(x-2)$ por homogeneidad no negativa de ReLU. Distintos parámetros, misma función.

**B2.** a) Exponencial (o Gamma): soporte $y>0$; la Normal asigna probabilidad a tiempos negativos y supone simetría. b) Exponencial con tasa $\lambda>0$: $\lambda_i=\exp(f(x_i,\phi))$ (o softplus). c) $L(\phi)=-\sum_i\log(\lambda_ie^{-\lambda_iy_i})=\sum_i[\lambda_iy_i-\log\lambda_i]=\sum_i\left[e^{f(x_i,\phi)}y_i-f(x_i,\phi)\right]$. No hay constantes que eliminar en este caso. d) Bernoulli, sigmoide $\lambda_i=\sigma(f(x_i,\phi))$, binary cross-entropy.

**B3.** a) $f_0=1.5$, $h_1=1.5$, $f_1=-1+4.5=3.5$, $\ell=(3.5-2)^2=2.25$.
b) $\partial\ell/\partial f_1=2(3.5-2)=3$. $\partial\ell/\partial\omega_1=3\cdot h_1=4.5$; $\partial\ell/\partial\beta_1=3$; $\partial\ell/\partial h_1=3\cdot\omega_1=9$; $\partial\ell/\partial f_0=9\cdot\mathbf 1[f_0>0]=9$; $\partial\ell/\partial\omega_0=9\cdot x=9$; $\partial\ell/\partial\beta_0=9$.
c) Con $\omega_0=-1$, $f_0=-0.5<0$: $h_1=0$ y $\mathbf 1[f_0>0]=0$, así que ambas parciales son $0$. La unidad está "muerta" para ese ejemplo y no aprende.
d) Simetría: todas las unidades de una capa reciben el mismo input, producen lo mismo y reciben el mismo gradiente; permanecen idénticas y la red equivale a una unidad por capa. Con ReLU además los gradientes ocultos son cero. En regresión lineal no hay unidades que distinguir.

## Parte C

a) Exp 1: **sesgo** (subajuste): train y val altas y casi iguales, sin brecha; el modelo es demasiado pequeño. Exp 2: **varianza** (sobreajuste): train casi cero, val sube desde la época 10, brecha enorme. Exp 3: **optimización**: lr 50× mayor, la pérdida oscila y nunca baja del nivel del azar ($\log4\approx1.39$); es la misma arquitectura que 2, así que el problema es el paso, no el modelo.

b) Early stopping en la época 10 (val baja a ~0.75, acc ~74%; train queda más alto); weight decay o dropout (train sube, val baja); reducir capacidad a un punto intermedio; más datos si se pueden conseguir. Cualquier dos con predicción coherente.

c) Usar el test para elegir entre 15 configuraciones lo convierte en un conjunto de validación: el 74% es optimista. Elegir con validación y tocar el test **una sola vez** con la configuración final.

d) No: el experimento 1 subajusta (sesgo alto). Dropout reduce varianza y aumenta sesgo, así que empeoraría train y probablemente val. Lo que necesita es más capacidad (más unidades o capas), no más regularización.
