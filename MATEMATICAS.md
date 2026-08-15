# Matemáticas y dominio de validez

## 1. Estado reducido

El modelo publicado no discretiza una membrana 2D. Representa `n` modos/resonadores efectivos en un anillo periódico. Para cada modo `i`:

\[
m\ddot z_i = -m\omega_i^2 z_i - \alpha_i z_i^3
+ k_c(z_{i-1}+z_{i+1}-2z_i)
- m\gamma_i\dot z_i - \eta_i z_i^2\dot z_i + F_i(z_i,u).
\]

Las unidades son SI: `z` en m, `m` en kg, `ω` en rad/s, `k_c` en N/m y `F` en N. Los índices son periódicos. La energía mecánica usada en la prueba de disipación contiene energía cinética, potencial lineal, Duffing y de acoplamiento.

Parámetros canónicos:

| Parámetro | Valor | Interpretación |
|---|---:|---|
| `n` | 16 | modos efectivos del benchmark |
| `f_i` | 10–40 MHz | muestreo uniforme independiente por semilla; sin perturbación adicional |
| `m` | 2×10⁻¹⁸ kg | masa efectiva ilustrativa |
| `Q` | 20 | amortiguamiento efectivo, no presión |
| `g` | 250 nm | hueco de aire nominal único |
| `d_ox` | 285 nm | espesor óptico/electrostático de SiO₂ |
| `V_dc`, `V_ac` | 1.5 V, 0.15 V | polarización y entrada |
| `Δt` | 0.25 ns | paso interno |
| pasos/símbolo | 8 | periodo de símbolo 2 ns |

La no linealidad se parametriza por una amplitud de referencia `a_D=5 nm`:

\[
\alpha_i = \frac{m\omega_i^2}{a_D^2},\qquad
\eta_i = 0.2\frac{m\gamma_i}{a_D^2},\qquad
\gamma_i=\omega_i/Q.
\]

La ablación lineal fija `α_i=η_i=0` y conserva el resto. Por tanto, la comparación aísla las no linealidades mecánicas dentro de este modelo; **no** aísla la no linealidad óptica común a ambos.

## 2. Excitación electrostática dependiente del hueco

La máscara de entrada `w_i` se congela por semilla:

\[
V_i(u)=V_{dc}+V_{ac}w_i u.
\]

La separación eléctrica efectiva del apilamiento aire/óxido es

\[
d_{eff,i}=g-z_i+\frac{d_{ox}}{\epsilon_{ox}},
\]

y la aproximación de placa paralela usada es

\[
F_i=\frac{\epsilon_0 A}{2}\frac{V_i^2}{d_{eff,i}^2}.
\]

Esta fuerza aumenta al cerrarse el hueco. No se usa una fuerza constante ni sólo el término cruzado `2V_dc V_ac`.

### Dominio de contacto

El continuo deja de ser válido cuando `g-z_i≤0.05g`. El integrador lanza `ContactError`; no recorta `z`, no continúa a través del sustrato y no llama a ese evento un umbral de *pull-in*. El estudio canónico registra cero contactos y un hueco mínimo de `0.838g`, por lo que ninguna de sus 48 trayectorias se acercó al umbral numérico.

## 3. Lectura óptica

Cada modo comparte el mismo punto de trabajo `g`. La observación es

\[
x_i = R(g-z_i)-R(g),
\]

donde `R` se calcula por matriz de transferencia a incidencia normal para aire/grafeno/aire/SiO₂/Si. Se adopta dependencia temporal `exp(-iωt)` e índices pasivos `n=n_r+iκ`, `κ≥0`; la propagación `exp(ikz)` atenúa. Para la dirección de transferencia usada por el código, cada capa es

\[
M_j=\begin{bmatrix}
\cos\delta_j & -i\sin\delta_j/n_j\\
-i n_j\sin\delta_j & \cos\delta_j
\end{bmatrix},\qquad
\delta_j=2\pi n_j d_j/\lambda.
\]

Una prueba independiente por recursión de coeficientes de Fresnel exige igualdad numérica con esta matriz para capas absorbentes; así detecta tanto una inversión de signo como una convención no documentada. El código también verifica `0≤R≤1` en el intervalo operativo y la periodicidad de media longitud de onda de la capa de aire. Los índices complejos son constantes ilustrativas a `λ=632.8 nm`; no constituyen una calibración de una oblea concreta. No se crean huecos aleatorios independientes para obtener diversidad artificial.

## 4. Integración y convergencia

Se usa Euler semiimplícito:

\[
v_{k+1}=v_k+a(z_k,v_k,u)\Delta t,\qquad
z_{k+1}=z_k+v_{k+1}\Delta t.
\]

Una prueba compara `Δt=0.5 ns` con `0.25 ns` manteniendo 2 ns por símbolo; el RMS entre lecturas debe ser menor que 8 % del RMS fino. Otra prueba exige pérdida neta de energía con voltaje cero y amortiguamiento positivo. Estas pruebas verifican consistencia numérica local, no convergencia espacial ni fidelidad experimental.

El contrato numérico falla cerrado antes de integrar: dimensiones, pasos, semillas y conteos deben ser enteros no booleanos; los flags requieren `bool` exacto; y los parámetros físicos deben ser escalares reales finitos. Arreglos booleanos o textuales no se convierten implícitamente en señales, gaps o desplazamientos; los índices ópticos deben ser escalares numéricos finitos y no nulos. Los escalares NumPy son admisibles cuando conservan las categorías integral, real o compleja requeridas.

## 5. Protocolo estadístico

Hay 12 pares explícitos:

- semillas del modelo: 4000–4011;
- NARMA-10: 5000–5011;
- paridad-3: 6000–6011.

Las tareas usan 1200 símbolos y descartan 200. El 65 % inicial del resto entrena y el 35 % final prueba. `StandardScaler`, `Ridge(α=0.001)` y `RidgeClassifier(α=1)` se ajustan sólo con entrenamiento.

Para NARMA:

\[
\operatorname{NRMSE}=\sqrt{\frac{\mathbb E[(y-\hat y)^2]}{\operatorname{Var}(y)}}.
\]

Los intervalos son percentiles 2.5/97.5 de 10 000 remuestras de la media sobre pares de semillas. La semilla base es 20260814 y cada uno de los diez resúmenes usa una subsemilla efectiva distinta, 20260814–20260823, serializada por nombre en `results.json`. Son intervalos Monte Carlo descriptivos del protocolo; no convierten 12 simulaciones en validación física. El protocolo no fue preregistrado externamente.

## 6. Afirmaciones prohibidas por el alcance

Las ecuaciones y pruebas anteriores no autorizan afirmar:

- dispositivo fabricado o medido;
- eficiencia energética o costo por inferencia;
- presión, gas o `Q` experimental;
- *pull-in* validado;
- superioridad general de ondas, grafeno o reservorios;
- una cota universal de capacidad;
- robustez int8, MCU, HIL, RF, arco eléctrico o reconocimiento de dígitos.
