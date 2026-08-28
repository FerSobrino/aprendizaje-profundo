# Cómo correr los notebooks del curso

Tienes dos opciones. Si no quieres instalar nada (o tu computadora no
tiene GPU), usa **Google Colab** — es la opción recomendada: te da GPU
gratis, que para deep learning importa. Si prefieres trabajar sin
internet y tener control total, instala Python **localmente**.

## Dónde están los materiales

- **Canvas**: ahí se publican los avisos y ahí entregas la presentación
  y el proyecto.
- **El repositorio de GitHub del curso**: contiene TODO el material
  (slides, notebooks, tareas y soluciones), es público y no necesitas
  cuenta para verlo o descargarlo:
  [github.com/FerSobrino/aprendizaje-profundo](https://github.com/FerSobrino/aprendizaje-profundo)

(De todos modos te conviene [crear tu cuenta de GitHub](https://github.com/signup)
— gratis, la usarás en otras clases y te va a servir toda la carrera.)

## Descarga los materiales una sola vez (recomendado)

En lugar de descargar archivo por archivo, copia la carpeta completa
del repositorio una vez — con eso ya tienes todos los notebooks con
las rutas relativas funcionando. Tres maneras, de más fácil a más
'difícil':

- **Zip (sin cuenta):** en la página del repo, botón verde *Code →
  Download ZIP*, y descomprime.
- **GitHub Desktop** (recomendada si git te da miedo): instala
  [desktop.github.com](https://desktop.github.com), *Clone repository* →
  URL `https://github.com/FerSobrino/aprendizaje-profundo`. Actualizar
  después es un botón (*Fetch/Pull*).
- **Terminal:**
  ```bash
  git clone https://github.com/FerSobrino/aprendizaje-profundo.git
  ```

Para actualizar cuando salga material nuevo: `git pull` (o vuelve a
bajar el zip y reemplaza).

## Opción 1 — Google Colab (recomendada)

1. Sube la carpeta completa del repo a tu **Google Drive** (arrástrala
   a drive.google.com).
2. Abre el notebook desde Drive (doble clic → *Abrir con → Google
   Colaboratory*; la primera vez: *Conectar más aplicaciones* →
   Colaboratory).
3. **Activa la GPU**: menú *Entorno de ejecución → Cambiar tipo de
   entorno de ejecución → T4 GPU*. Sin esto, entrenar en Colab es
   lento.
4. Monta tu Drive en la primera celda para que las rutas a datos
   funcionen:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. PyTorch y torchvision ya vienen instalados en Colab — no necesitas
   instalar nada.

## Opción 2 — Instalación local

1. Instala [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Crea el entorno del curso:
   ```bash
   conda create -n dl python=3.12
   conda activate dl
   pip install torch torchvision jupyter matplotlib pandas numpy
   ```
3. Lanza Jupyter desde la carpeta del repo:
   ```bash
   jupyter notebook
   ```
4. Sobre la GPU local:
   - **Mac con Apple Silicon**: PyTorch usa la GPU automáticamente vía
     MPS; los notebooks del curso ya detectan el dispositivo
     (`find_device()`).
   - **Windows/Linux con NVIDIA**: instala la variante CUDA de PyTorch
     (instrucciones en [pytorch.org](https://pytorch.org/get-started/locally/)).
   - **Sin GPU**: todo corre en CPU pero más lento — usa los tamaños
     `small_data` que los notebooks incluyen, o cámbiate a Colab.

## Verificación rápida

Corre esto en un notebook; si imprime un dispositivo y un tensor,
estás listo:

```python
import torch
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(device, torch.rand(2, 2, device=device))
```
