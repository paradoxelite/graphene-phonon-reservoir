# Model Card — Graphene Phonon Reservoir reducido

## Identidad

- Tipo: simulador reducido determinista con heterogeneidad sembrada.
- Versión candidata: 1.0.0.
- Evidencia: exclusivamente simulación.
- Resultado principal: negativo bajo el protocolo canónico.

## Uso previsto

- Reproducir un ejemplo de auditoría y falsificación temprana en computación física.
- Estudiar una familia concreta de osciladores acoplados con electrostática dependiente del hueco y lectura TMM.
- Comparar de forma pareada una dinámica mecánica no lineal con su ablación lineal.
- Enseñar cómo congelar semillas, separar cronológicamente y reportar resultados que no apoyan la hipótesis inicial.

## Uso no previsto

No usar para diseñar una oblea, fijar voltajes seguros, predecir *pull-in*, estimar presión/Q, dimensionar un fotodetector, calcular potencia, prometer exactitud de aplicación, evaluar int8 ni sustituir FEM, caracterización óptica o mediciones.

## Entradas y salidas

- Entrada: una secuencia escalar adimensional acotada en `[-1,1]`.
- Estado: desplazamientos y velocidades de 16 modos efectivos.
- Salida expuesta al readout: cambio de reflectancia TMM por modo.
- Contacto: excepción explícita, no una salida válida.

## Datos

Sólo se usan señales sintéticas:

- NARMA-10, con semillas 5000–5011;
- paridad-3, con semillas 6000–6011.

No hay datos humanos, industriales, RF, eléctricos ni experimentales.

## Evaluación congelada

Doce semillas de modelo 4000–4011, emparejadas entre condiciones. El escalado y el readout se ajustan únicamente sobre el prefijo cronológico de entrenamiento. La línea de retardo digital es una referencia explícita, no un reservorio físico equiparado.

Resultados medios:

- NARMA-10 NRMSE: no lineal 1.033; lineal 1.000; retardo 0.502.
- Paridad-3: no lineal 0.497; lineal 0.501; retardo 0.502.

No se observó ventaja computacional. El intervalo bootstrap de la diferencia NARMA no lineal menos lineal es enteramente positivo, es decir, peor para la condición no lineal en este protocolo.

## Controles técnicos

- TMM acotado, periódico en la capa de aire y consistente con una recursión de Fresnel independiente para la convención pasiva `n+iκ`.
- Fuerza electrostática monótona al cerrar hueco.
- Contacto fail-closed.
- Disipación sin excitación.
- Refinamiento temporal.
- Repetibilidad por semilla.
- Heterogeneidad idéntica entre ablaciones; cada extracción uniforme de frecuencias queda serializada por trial.
- Diez subsemillas bootstrap efectivas registradas por nombre de resumen.
- Rechazo de trials duplicados o con semillas solapadas entre roles.
- Publicación recuperable de cinco payloads bajo lock de OS: journal durable previo a backups, restauración idempotente al reiniciar, rechazo de reparse points, rollback probado y manifiesto SHA-256 estricto instalado al final. Toda lectura de confianza captura las identidades físicas de todos los componentes antes de `open`, liga el descriptor a ese snapshot, aplica límites explícitos y revalida al cierre; crecimiento y sustitución pre-open/en lectura tienen falsificadores dedicados, incluso con bytes idénticos.
- Tipos públicos fail-closed: enteros no booleanos, flags `bool` exactos, escalares físicos reales finitos, índices ópticos numéricos finitos/no nulos y rechazo de señales, gaps o desplazamientos booleanos/textuales como entrada numérica.
- Ausencia de *leakage* del conjunto de prueba.
- Regeneración de resultados y figuras.
- Margen de dominio: hueco mínimo observado `0.838 g`, límite fail-closed `0.05 g`.

## Riesgos y límites

- El modelo reducido puede omitir modos, geometría, tensión no uniforme, electrostática distribuida, calentamiento y ruido.
- Los parámetros ilustrativos no provienen de un dispositivo único caracterizado.
- La lectura simula canales simultáneos; no incluye óptica ni adquisición realizables.
- El tamaño muestral (12 pares) describe el protocolo, no el universo de diseños.
- No hubo preregistro externo.

## Decisión de uso

El resultado soporta **detener** cualquier afirmación de ventaja o viabilidad y conservar el proyecto como prueba de concepto numérica negativa. Nuevas variantes deben formularse como estudios nuevos, con semillas y criterios definidos antes de observar sus resultados.
