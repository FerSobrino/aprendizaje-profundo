# Proyecto de colaboración con IA — Dirigir a un agente para hacer deep learning

**Peso:** 15% de la calificación final.
**Formato:** individual. Puedes discutir con compañeros, pero la conversación con el agente y los entregables son tuyos.
**Entrega:** semana 12, junto con las presentaciones.

## Por qué este proyecto

En la práctica ya casi nadie escribe un loop de entrenamiento desde cero: se lo pides a un agente de IA. Pero el agente hace *exactamente lo que le pides* — si no sabes pedir una división train/val/test, pesos por clase o métricas por subgrupo, no las vas a obtener; y si no sabes leer una curva de pérdida, no vas a notar cuando el resultado esté mal. Este proyecto evalúa la habilidad que sí importa ahora: **especificar bien, verificar con escepticismo e interpretar con criterio**. El agente pone el código; tú pones el entendimiento.

## Qué vas a hacer

Elige **una** de las siguientes opciones y dirígela de principio a fin usando un agente de IA (Claude, ChatGPT, Gemini, Claude Code, etc. — el que quieras):

1. **Regularización (base: Tarea 5).** El grid de optimizadores × weight decay en MNIST, más el régimen de pocos datos (N=2000) con aumento de datos. Reporta curvas, tabla final con val_acc, $\|\theta\|_2$ y ECE.
2. **CNN (base: Tarea 6).** El grid 3×3 de dropout × weight decay en small_data, más la comparación de aumentos de datos con la cuadrícula visual que verifica que las transformaciones preservan la etiqueta.
3. **Secuencias (base: Tarea 7).** RNN vs. LSTM vs. GRU × 2 optimizadores sobre un corpus, con curvas de perplejidad y tiempos por época, cerrando con una recomendación de despliegue.
4. **Opción aplicada (recomendada, un poco más ambiciosa).** Un problema de política pública con datos reales y abiertos: clasificación de texto legislativo/gubernamental, predicción de demanda de un servicio público, clasificación de imágenes satelitales con transfer learning, etc. Acuérdalo conmigo antes de empezar. Debe incluir un protocolo de validación serio y métricas por subgrupo cuando aplique.

En cualquier opción el estándar experimental es el del curso: semilla fija, mismo split entre configuraciones, un factor variado a la vez, curvas + tabla resumen + interpretación escrita.

## Entregables (5)

1. **Especificación previa** (1 página, escrita **antes** de abrir el chat). Qué vas a pedir y por qué: dataset y splits, arquitectura, configuraciones a comparar, hiperparámetros fijos, métricas y qué esperas observar teóricamente. Esta especificación se congela: entregas la versión original, con correcciones posteriores marcadas como tales.
2. **Transcripción completa** de la(s) conversación(es) con el agente, sin editar. Exporta el chat o copia todo; si usaste varias sesiones, inclúyelas todas.
3. **Reporte de verificación** (1–2 páginas). La parte más importante: ¿qué revisaste del trabajo del agente y cómo? Como mínimo: (a) verifica que el protocolo pedido se cumplió (semillas, splits, factor único); (b) revisa el código en los puntos críticos (¿la pérdida es la correcta? ¿`model.eval()` y `no_grad` en evaluación? ¿se normalizó con estadísticas solo de train?); (c) contrasta al menos un resultado contra tu predicción teórica. **Documenta al menos dos errores, decisiones cuestionables o cosas que tuviste que corregir del agente** — en nuestra experiencia siempre las hay; si de verdad no encontraste ninguna, explica qué revisaste para descartarlas.
4. **Interpretación de resultados** (1–2 páginas). Las preguntas de "explica por qué" de siempre: qué configuración ganó y por qué tiene sentido (o no) a la luz de la teoría del curso, limitaciones, y qué recomendarías a alguien que fuera a usar esto.
5. **Reflexión breve** (media página). ¿Qué tuviste que saber tú para que esto saliera bien? ¿Dónde el agente fue mejor que tú y dónde tú fuiste indispensable?

## Rúbrica (100 pts)

| Componente | Pts | Qué se evalúa |
|---|---|---|
| Especificación | 30 | Completa y teóricamente fundamentada *antes* de empezar: splits, semillas, control de factores, métricas correctas para el problema. Una especificación a la que el agente no le pueda meter un gol. |
| Verificación | 30 | Escepticismo con evidencia: revisiones concretas al código y al protocolo, errores del agente detectados y documentados, contraste resultado-vs-predicción. |
| Interpretación | 25 | Conexión con la teoría del curso, honestidad sobre limitaciones, recomendación defendible. |
| Reflexión y forma | 15 | Reflexión genuina; entrega completa (los 5 entregables), a tiempo, con la transcripción íntegra. |

**Lo que NO se califica:** la elegancia del código (lo escribió el agente), ni qué tan buenos salieron los números. Un experimento con resultados mediocres, bien especificado, bien verificado y bien interpretado, obtiene mejor nota que uno con números espectaculares que no puedes explicar.

**Descalificación automática:** entregar sin la transcripción completa, o una transcripción que no corresponde a los resultados reportados.

## Consejos

- Pídele al agente el plan antes que el código y critícalo contra tu especificación.
- Pide instrumentación desde el inicio (curvas train/val, normas de gradiente, tablas) — es más barato que reconstruirla después.
- Cuando algo se vea demasiado bien (val acc > train acc, pérdida en cero), sospecha primero de fuga de datos o de un bug de evaluación, no de tu suerte.
- Los notebooks del curso (`notebooks/`) son tu referencia de qué debería aparecer en el código; úsalos para auditar lo que el agente produzca.
