# Problema 1: decisión y entropía

## a) Fracción de veces que paga el protagonista

Las probabilidades conocidas son:

$$
P(\text{María})=\frac{1}{2}, \quad P(\text{Pablo})=\frac{1}{4}, \quad P(\text{Juan})=0.
$$

Además, el enunciado dice que Sara y Carlos pagan indistintamente entre ambos la octava parte de las veces. Si suponemos que ambos contribuyen por igual, entonces:

$$
P(\text{Sara})=P(\text{Carlos})=\frac{1}{16}.
$$

La suma de las probabilidades conocidas es:

$$
\frac{1}{2}+\frac{1}{4}+\frac{1}{16}+\frac{1}{16}+0=\frac{7}{8}.
$$

Como las probabilidades de todos los posibles pagadores deben sumar $1$, la probabilidad restante corresponde al protagonista:

$$
P(\text{pague el protagonista})=1-\frac{7}{8}=\frac{1}{8}.
$$

Por lo tanto, el protagonista paga la ronda $\frac{1}{8}$ de las veces.

---

## b) Entropía de la distribución

La distribución de probabilidad queda:

| Persona | Probabilidad |
|:---:|:---:|
| María | $\frac{1}{2}$ |
| Pablo | $\frac{1}{4}$ |
| Sara | $\frac{1}{16}$ |
| Carlos | $\frac{1}{16}$ |
| Juan | $0$ |
| Protagonista | $\frac{1}{8}$ |

La entropía es una medida de incertidumbre. En este contexto, mide cuánta incertidumbre tenemos sobre quién pagará la ronda antes de observar el resultado.

Si una sola persona pagara siempre, no habría incertidumbre y la entropía sería $0$. En cambio, si varias personas pudieran pagar con probabilidades parecidas, sería más difícil anticipar el resultado y la entropía sería mayor.

La entropía también puede interpretarse como la cantidad promedio de información necesaria para identificar el resultado. Como usamos logaritmo en base $2$, la unidad es el bit. Por eso, en este problema, la entropía representa el número promedio de preguntas binarias necesarias para descubrir quién paga.

En clase hablamos de la distribucion del clima, si nunca nieva en la ciudad de cordoba, entonces la frase "mañana no nieva" no aporta nada de informacion, caso contrario, si la mitad de alumnos son de licenciatura en matematica aplicada, y la otra mitad no, la frase "el alumno con mejor notas fue de matematica aplicada", me permite descartar a la mitad del curso.

Esta idea de que "lo improbable aporta mas informacion", se ve reflejada en la formula:

$$
H(X)=-\sum_i p_i\log_2(p_i) = E[\log_2(X)].
$$

El término de Juan no aporta porque tiene probabilidad cero. Por convención, $0\log_2(0)=0$.

Entonces, para este caso que veniamos viendo:

$$
 H(X)=-\frac{1}{2}\log_2\left(\frac{1}{2}\right)-\frac{1}{4}\log_2\left(\frac{1}{4}\right)-2\frac{1}{16}\log_2\left(\frac{1}{16}\right)-\frac{1}{8}\log_2\left(\frac{1}{8}\right).\\
\iff H(X)=\frac{1}{2}+\frac{1}{2}+\frac{1}{2}+\frac{3}{8}=1.875.
$$

$$
H(X)=1.875 \text{ bits}.
$$

Esto significa que, en promedio, se necesitan aproximadamente $1.875$ preguntas binarias para determinar quién paga la ronda.

---

## c) Árbol de decisión usando entropía

La variable objetivo es `Bebida`. Las variables explicativas son `Sexo`, `Estudiante` y `Baile`.

Los datos disponibles son:

| Bebida | Sexo | Estudiante | Baile |
|:---:|:---:|:---:|:---:|
| cerveza | M | T | T |
| cerveza | M | F | T |
| vodka | M | F | F |
| vodka | M | F | F |
| vodka | F | T | T |
| vodka | F | F | F |
| vodka | F | T | T |
| vodka | F | T | T |

Un árbol de decisión construye reglas sucesivas para dividir el conjunto de datos. En cada nodo se elige una variable, y cada valor posible de esa variable genera una rama.

La pregunta central es qué variable conviene elegir primero. Para decidirlo, usamos la ganancia de información.

La ganancia de información mide cuánto disminuye la incertidumbre después de dividir el conjunto usando una variable. Si una variable separa bien las clases, la entropía posterior baja mucho y la ganancia de información es alta. Si una variable no ayuda a distinguir las clases, la entropía casi no cambia y la ganancia es baja.

La fórmula es:

$$
IG(S,A)=H(S)-H(S|A).
$$

Acá, $H(S)$ es la entropía del conjunto antes de dividir, y $H(S|A)$ es la entropía promedio después de dividir usando la variable $A$.

### Entropía inicial

En total hay $8$ ejemplos: $2$ de `cerveza` y $6$ de `vodka`.

$$
P(\text{cerveza})=\frac{2}{8}=\frac{1}{4}, \quad P(\text{vodka})=\frac{6}{8}=\frac{3}{4}.
$$

La entropía inicial es:

$$
H(S)=-\frac{1}{4}\log_2\left(\frac{1}{4}\right)-\frac{3}{4}\log_2\left(\frac{3}{4}\right)\approx 0.811.
$$

Esta es la incertidumbre inicial antes de elegir una variable para dividir.

### Ganancia de información para `Sexo`

Si dividimos usando `Sexo`, obtenemos dos subconjuntos.

Para `Sexo = M` hay $4$ ejemplos: $2$ cerveza y $2$ vodka. Como las clases están perfectamente balanceadas, la incertidumbre es máxima:

$$
H(S_M)=1.
$$

Para `Sexo = F` hay $4$ ejemplos y todos son `vodka`. Entonces no hay incertidumbre:

$$
H(S_F)=0.
$$

La entropía ponderada posterior es:

$$
H(S|\text{Sexo})=\frac{4}{8}\cdot 1+\frac{4}{8}\cdot 0=0.5.
$$

Por lo tanto:

$$
IG(S,\text{Sexo})=0.811-0.5=0.311.
$$

### Ganancia de información para `Baile`

Para `Baile = T` hay $5$ ejemplos: $2$ cerveza y $3$ vodka.

$$
H(S_T)=-\frac{2}{5}\log_2\left(\frac{2}{5}\right)-\frac{3}{5}\log_2\left(\frac{3}{5}\right)\approx 0.971.
$$

Para `Baile = F` hay $3$ ejemplos y todos son `vodka`, por lo que:

$$
H(S_F)=0.
$$

La entropía ponderada posterior es:

$$
H(S|\text{Baile})=\frac{5}{8}\cdot 0.971+\frac{3}{8}\cdot 0\approx 0.607.
$$

Entonces:

$$
IG(S,\text{Baile})=0.811-0.607\approx 0.204.
$$

### Ganancia de información para `Estudiante`

Para `Estudiante = T` hay $4$ ejemplos: $1$ cerveza y $3$ vodka.

$$
H(S_T)=-\frac{1}{4}\log_2\left(\frac{1}{4}\right)-\frac{3}{4}\log_2\left(\frac{3}{4}\right)\approx 0.811.
$$

Para `Estudiante = F` hay $4$ ejemplos: $1$ cerveza y $3$ vodka.

$$
H(S_F)\approx 0.811.
$$

La entropía ponderada posterior es:

$$
H(S|\text{Estudiante})=\frac{4}{8}\cdot 0.811+\frac{4}{8}\cdot 0.811=0.811.
$$

Entonces:

$$
IG(S,\text{Estudiante})=0.811-0.811=0.
$$

Resumiendo:

| Variable | Ganancia de información |
|:---:|:---:|
| `Sexo` | $0.311$ |
| `Baile` | $0.204$ |
| `Estudiante` | $0$ |

La mayor ganancia de información corresponde a `Sexo`, por lo que se elige `Sexo` como nodo raíz.

La rama `Sexo = F` queda resuelta, porque todos los ejemplos son `vodka`.

$$
\text{Sexo = F} \Rightarrow \text{vodka}.
$$

Ahora falta continuar la rama `Sexo = M`.

### Rama `Sexo = M`

En esta rama quedan los siguientes ejemplos:

| Bebida | Sexo | Estudiante | Baile |
|:---:|:---:|:---:|:---:|
| cerveza | M | T | T |
| cerveza | M | F | T |
| vodka | M | F | F |
| vodka | M | F | F |

Hay $2$ ejemplos de `cerveza` y $2$ ejemplos de `vodka`, por lo que:

$$
H(S')=1.
$$

Calculamos la ganancia de información con las variables restantes.

Para `Baile`, si `Baile = T`, ambos ejemplos son `cerveza`; si `Baile = F`, ambos ejemplos son `vodka`. La separación es perfecta:

$$
IG(S',\text{Baile})=1.
$$

Para `Estudiante`, la división no separa completamente las clases. Si `Estudiante = T`, hay un único ejemplo y es `cerveza`; si `Estudiante = F`, hay $3$ ejemplos: $1$ cerveza y $2$ vodka.

$$
H(S_F)=-\frac{1}{3}\log_2\left(\frac{1}{3}\right)-\frac{2}{3}\log_2\left(\frac{2}{3}\right)\approx 0.918.
$$

La entropía ponderada es:

$$
H(S'|\text{Estudiante})=\frac{1}{4}\cdot 0+\frac{3}{4}\cdot 0.918\approx 0.689.
$$

Entonces:

$$
IG(S',\text{Estudiante})=1-0.689\approx 0.311.
$$

Comparando ambas variables:

| Variable | Ganancia de información |
|:---:|:---:|
| `Baile` | $1$ |
| `Estudiante` | $0.311$ |

Se elige `Baile` porque produce la mayor ganancia de información.

El árbol final es:

    Sexo
    ├── F → vodka
    └── M
        └── Baile
            ├── T → cerveza
            └── F → vodka

Este árbol clasifica correctamente los $8$ ejemplos del conjunto de entrenamiento. Por lo tanto:

$$
\text{accuracy}=\frac{8}{8}=1.
$$

---

## d) Poda del árbol

Una poda posible consiste en eliminar el nodo `Baile` dentro de la rama `Sexo = M`.

El árbol podado queda:

    Sexo
    ├── F → vodka
    └── M → clase mayoritaria

En la rama `Sexo = F` se clasifican correctamente los $4$ ejemplos.

En la rama `Sexo = M` hay empate: $2$ ejemplos de `cerveza` y $2$ ejemplos de `vodka`. Ante un empate, debe definirse una regla de desempate. Si elegimos una de las dos clases como predicción constante, se aciertan $2$ de los $4$ casos de esa rama.

Entonces, el árbol podado acierta:

$$
4+2=6
$$

ejemplos sobre un total de $8$. La nueva accuracy es:

$$
\text{accuracy}=\frac{6}{8}=0.75.
$$

La poda simplifica el árbol y reduce su complejidad. En este ejemplo, la accuracy de entrenamiento baja de $1$ a $0.75$.

Esto no significa necesariamente que la poda sea mala. En problemas reales, podar puede ayudar a evitar sobreajuste y mejorar la capacidad de generalización.


## Algunas aclaraciones

- En este ejercicio usamos **entropía** para medir la impureza de cada nodo y **ganancia de información** para decidir qué variable conviene usar en cada partición. La idea general es elegir la división que más reduzca la incertidumbre sobre la clase.

- La entropía no es la única medida posible. En árboles de clasificación también es muy común usar el **índice de Gini**, que mide qué tan mezcladas están las clases dentro de un nodo. En `scikit-learn`, el criterio por defecto de `DecisionTreeClassifier` es `gini`, aunque también puede elegirse `entropy`(en mi experiencia nunca tuve resultados muy distintos al usar uno por sobre otro).

- En este ejemplo todas las variables eran binarias, por lo que cada división era directa: por ejemplo, `Sexo = M` o `Sexo = F`. Si las variables fueran numéricas, no bastaría con elegir una columna; habría que elegir también un **umbral de corte**, por ejemplo `edad <= 30` o `ingreso <= 1000`.

- Los árboles de decisión se construyen de manera **greedy**: en cada nodo eligen la mejor división local según el criterio elegido. Esto no garantiza necesariamente el árbol globalmente óptimo, pero permite construir modelos de forma eficiente, (construir el arbol optimo es una tarea NP-Complete) .

- Si a un arbol, lo dejo crecer sin restricciones, normalmente puede ajustarse demasiado al conjunto de entrenamiento. Esto significa que estos modelos tienden a hacer **overfitting** si no se eligen bien los hiperparámetros(mirar sklearn porque son varios).

- Algunos hiperparámetros importantes para controlar la complejidad del árbol son `max_depth`, `min_samples_split`, `min_samples_leaf` y `ccp_alpha`. Estos parámetros limitan cuánto puede crecer el árbol o permiten podarlo.

- En este ejercicio usamos árboles para **clasificación**, donde cada hoja predice una clase. Sin embargo, también existen árboles de **regresión**, donde cada hoja predice un valor numérico, usualmente el promedio de los valores de entrenamiento que caen en esa hoja.

- Una ventaja importante de los árboles es que son modelos muy **interpretables**: sus decisiones pueden leerse como una secuencia de reglas. Sin embargo, esta interpretabilidad puede perderse parcialmente si el árbol crece demasiado.