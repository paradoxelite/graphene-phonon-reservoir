# Graphene Phonon Reservoir

## Resultado negativo reproducible en simulación reducida

Este repositorio evalúa un modelo reducido de modos mecánicos acoplados inspirado en resonadores de grafeno. **No es un dispositivo construido, una validación experimental ni un gemelo digital de una geometría fabricada.**

## Identidad de versión

Esta publicación conserva dos dominios de versión intencionalmente independientes:

- versión del software: `1.0.1`; identifica el repositorio, `VERSION`, `CITATION.cff`, esta documentación y la corrección del contrato de CI multiplataforma;
- versión del informe científico congelado: `1.0.0`; identifica `paper/main.tex`, `paper/main.pdf` y la generación científica publicada originalmente con `v1.0.0`.

El informe científico se conserva byte a byte junto con los demás payloads congelados en la versión de software 1.0.1: no se regenera ni se reinterpreta para corregir CI. Por ello el rótulo «Versión 1.0.0» de la primera página es la identidad histórica del informe, no la versión del software que lo distribuye. Al citar, use software `v1.0.1` para el repositorio y versión `1.0.0` para el informe incluido.

La comprobación canónica usa 12 pares de semillas y no encontró ventaja computacional del modelo no lineal bajo el protocolo congelado:

| Tarea | Modelo no lineal | Ablación mecánica lineal | Línea de retardo digital |
|---|---:|---:|---:|
| NARMA-10, NRMSE ↓ | 1.033 | 1.000 | 0.502 |
| Paridad-3, exactitud ↑ | 0.497 | 0.501 | 0.502 |

En NARMA-10, la diferencia pareada no lineal menos lineal fue **+0.033** (IC bootstrap 95 % de la media: **[+0.018, +0.048]**): en esta configuración, la no linealidad empeoró el resultado. En paridad-3, todos los métodos quedaron cerca del azar y el intervalo de la diferencia no lineal menos lineal cruzó cero. Estos datos no justifican afirmar ventaja de reservorio, superioridad del grafeno ni viabilidad de hardware.

## Qué sí contiene

- Ecuación reducida en SI para un anillo de modos mecánicos acoplados.
- Fuerza electrostática dependiente del hueco y del voltaje total.
- Dominio de contacto que falla de forma explícita; no se recorta la trayectoria.
- Lectura óptica por matriz de transferencia con **un solo hueco nominal autorizado**, convención pasiva `n+iκ` y prueba cruzada por recursión de Fresnel.
- Ablación mecánica lineal pareada con las mismas frecuencias, máscara, amortiguamiento, acoplamiento, electrostática y lectura.
- Línea de retardo digital como referencia separada.
- Separación cronológica; escalado y readout ajustados sólo con el prefijo de entrenamiento.
- 12 pares de semillas explícitos, frecuencias realizadas serializadas por trial, 10 000 remuestras bootstrap y las diez subsemillas efectivas registradas.
- Pruebas de límites ópticos, contacto, disipación, refinamiento temporal, semillas y ausencia de *leakage*. En el entorno Windows canónico, JSON y PNG se reproducen byte a byte; en Linux se exige estructura/texto exactos, equivalencia numérica estrecha y píxeles RGBA exactos.
- Margen de dominio registrado: el hueco mínimo entre 48 trayectorias fue **0.838 g**, lejos del límite de contacto **0.05 g**.

## Qué no demuestra

- No hay fabricación, medición, HIL, MCU, cuantización int8 ni electrónica de lectura.
- No se estima rendimiento energético, RF, arco eléctrico, dígitos, capacidad universal ni costo por oblea.
- El parámetro `Q=20` es un amortiguamiento efectivo; no se traduce a gas o presión.
- El modelo es reducido y no resuelve una membrana 2D ni una curva de *pull-in* validada.
- Las constantes ópticas y mecánicas son un punto de simulación ilustrativo, no una ficha de proceso.
- El protocolo no fue preregistrado externamente.

## Reproducción

Entorno canónico: Python 3.11.9. Desde una exportación limpia, cree un entorno
**nuevo y externo**: ni el venv ni sus cachés pueden vivir dentro del árbol
publicable.

### Windows PowerShell

```powershell
$ErrorActionPreference = 'Stop'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
Get-ChildItem Env:PIP_* | ForEach-Object {
    Remove-Item "Env:$($_.Name)"
}
$env:PIP_CONFIG_FILE = 'NUL'
$env:PYTHONNOUSERSITE = '1'
$BasePython = $env:PYTHON3119
if (-not $BasePython) { $BasePython = (Get-Command python).Source }
$DetectedPython = & $BasePython -c 'import platform; print(platform.python_version())'
if ($LASTEXITCODE -ne 0 -or $DetectedPython -ne '3.11.9') {
    throw 'Se requiere exactamente Python 3.11.9'
}
$RunRoot = Join-Path $env:TEMP ("graphene-repro-" + [guid]::NewGuid())
$Venv = Join-Path $RunRoot 'venv'
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
& $BasePython -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $RunRoot 'pycache'
$env:PYTHONHASHSEED = '0'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = Join-Path $RunRoot 'mpl'
& $VenvPython -m pip install --use-feature=truststore --require-hashes -r requirements-dev-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Falló la instalación hashada' }
& $VenvPython -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw 'Falló la suite' }
& $VenvPython reproduce.py --compile-report
if ($LASTEXITCODE -ne 0) { throw 'Falló el productor canónico' }
```

### Linux/macOS

```bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
while IFS='=' read -r name _; do
  case "$name" in
    PIP_*) unset "$name" ;;
  esac
done < <(env)
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
BASE_PYTHON="${PYTHON3119:-python3.11}"
test "$("$BASE_PYTHON" -c 'import platform; print(platform.python_version())')" = '3.11.9'
RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/graphene-repro.XXXXXX")"
VENV="$RUN_ROOT/venv"
"$BASE_PYTHON" -m venv "$VENV"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_ROOT/mpl"
"$VENV/bin/python" -m pip install --use-feature=truststore --require-hashes -r requirements-dev-lock.txt
"$VENV/bin/python" -m pytest -q -p no:cacheprovider
"$VENV/bin/python" reproduce.py --compile-report
```

Ambas recetas eliminan primero todas las variables heredadas `PIP_*` y anulan los archivos de configuración efectivos de pip (`NUL` o `/dev/null`). Así, `--use-feature=truststore` conserva verificación TLS sin heredar `trusted-host`; `--require-hashes` verifica además cada distribución instalada.

Los artefactos congelados de esta versión usan Windows x86-64 como plataforma canónica de bytes. Con las mismas versiones bloqueadas, Ubuntu 24.04 conserva todas las claves y textos; 51 floats derivados de BLAS difieren como máximo `1.65e-12` en valor absoluto y `1.38e-11` en valor relativo. Los PNG decodifican a dimensiones y píxeles RGBA idénticos aunque su codificación binaria difiera. La suite falla si se exceden `rtol=2e-11` o `atol=2e-12`, si cambia cualquier campo no flotante o cualquier píxel, y conserva la comparación byte a byte completa en Windows. Esto es portabilidad numérica/visual medida, no identidad binaria entre sistemas operativos.

`reproduce.py` es la entrada única para regenerar:

- `results.json`;
- `figures/paired_narma.png`;
- `figures/paired_parity.png`;
- `paper/results_macros.tex`;
- opcionalmente `paper/main.pdf`;
- `artifact_manifest.json`, publicado al final con tamaño y SHA-256 de cada payload.

Los productores trabajan primero fuera del árbol publicado. Bajo un lock de OS se escribe y sincroniza un journal durable **antes del primer respaldo**, se toman copias inmutables y se instalan los cinco payloads con `os.replace`; `artifact_manifest.json` se mueve al final como marcador de confianza. Si el proceso termina abruptamente, la siguiente invocación detecta el journal bajo el mismo lock y restaura de forma idempotente la generación anterior antes de generar nada nuevo. La publicación rechaza symlinks, junctions y otros reparse points en cada componente del root, destinos, lock, backups y recovery. Las pruebas matan procesos reales después de cada backup y de cada movimiento público, fuerzan fallos moved-then-raised, rollback y unlock, y exigen cleanup explícito.

Las lecturas que deciden confianza no usan `Path.read_bytes()`: capturan primero la identidad física `lstat` de todos los componentes, abren una vez un descriptor, exigen que éste y la ruta sigan ligados al snapshot pre-open, calculan SHA-256 por bloques de 64 KiB y vuelven a comparar componentes, tipo, tamaño y timestamps antes de aceptar. Los límites físicos son 4 MiB para resultados, 8 MiB por figura, 256 KiB para macros, 32 MiB para PDF, 16 KiB para el manifiesto y 64 KiB para el journal. Crecimiento, sustitución antes de `open`, sustitución durante la lectura o exceso de límite fallan cerrado; las regresiones incluyen reemplazos con bytes y tamaño idénticos para demostrar que el hash no sustituye la identidad física.

Las APIs numéricas preservan procedencia mediante una sola instantánea por entrada: validan sus escalares originales y convierten esa misma instantánea, sin volver a leer proveedores array-like stateful. Enteros no aceptan booleanos ni fracciones, flags exigen `bool` exacto y todo escalar físico real debe ser finito. Señales, gaps y desplazamientos booleanos o textuales no se interpretan como números; las señales complejas también se rechazan en vez de descartar su parte imaginaria. `delay_embed` conserva la señal real vacía como una matriz de forma `(0, order)`. Tanto las matrices `X` como los targets `y` de ambos ajustes auditados deben ser reales numéricos y rechazan booleanos —incluidos los mezclados con enteros o reales—, texto y complejos. Los índices ópticos deben ser escalares numéricos finitos y no nulos. Se aceptan escalares integrales, reales y complejos de NumPy sólo donde la API conserva explícitamente esas categorías. La serialización de resultados usa JSON estricto y rechaza `NaN`/`Infinity`; el renderer de macros exige enteros exactos y reales finitos antes de emitir LaTeX.

La suite completa vuelve a ejecutar las 12 parejas. Compara exactamente la generación canónica Windows y aplica en otros sistemas el contrato portable estricto documentado arriba; no confunde diferencias de BLAS o codificación PNG con cambios del resultado científico.

## Archivos principales

- `physical_model.py`: física reducida, electrostática, contacto y TMM.
- `study.py`: protocolo, readouts, estadísticas y figuras.
- `tasks.py`: señales sintéticas NARMA-10 y paridad.
- `MATEMATICAS.md`: ecuaciones, unidades y dominio de validez.
- `MODEL_CARD.md`: uso permitido y límites.
- `AUDIT_RESOLUTION.md`: resolución de los 17 bloqueos originales y de revisiones adversariales posteriores.
- `SOURCE_VERIFICATION.md`: DOI, títulos y alcance de las fuentes comprobadas.
- `paper/main.pdf`: informe técnico derivado del JSON canónico.

## Referencias de contexto

- Tanaka et al., *Recent advances in physical reservoir computing: A review*, 2019. https://doi.org/10.1016/j.neunet.2019.03.005
- Bunch et al., *Electromechanical Resonators from Graphene Sheets*, 2007. https://doi.org/10.1126/science.1136836
- Eichler et al., *Nonlinear damping in mechanical resonators made from carbon nanotubes and graphene*, 2011. https://doi.org/10.1038/nnano.2011.71
- Dion et al., *Reservoir computing with a single delay-coupled non-linear mechanical oscillator*, 2018. https://doi.org/10.1063/1.5038038
- Aguila et al., *Fabry–Perot interferometric calibration of van der Waals material-based nanomechanical resonators*, 2022. https://doi.org/10.1039/D1NA00794G

## Licencia

No se concede una licencia abierta global. Consulte al autor antes de reutilizar código o documentación. Las referencias bibliográficas no implican incorporación de código de terceros.
