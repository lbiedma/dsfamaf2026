### a

La regla de decisión establece:
* $\omega_1$ si $x > \theta$.
* $\omega_2$ si $x \le \theta$.

Cometer un error en este problema puede ocurrir de dos maneras:
1.  **Falso Negativo**: La clase verdadera es $\omega_1$, pero la regla decide $\omega_2$. Esto ocurre cuando $x \le \theta$.
2.  **Falso Positivo**: La clase verdadera es $\omega_2$, pero la regla decide $\omega_1$. Esto ocurre cuando $x > \theta$.

Ambas son mutuamente excluyentes, por lo tanto la probabilidad total de error es la suma de las probabilidades:
$$P(error) = P(x \le \theta, \omega_1) + P(x > \theta, \omega_2)$$

Utilizando la regla de la probabilidad conjunta $P(A, B) = P(A|B)P(B)$, podemos reescribir esto en términos de las densidades de probabilidad condicionales y las probabilidades a priori:
$$P(error) = \int_{-\infty}^{\theta} p(x, \omega_1)dx + \int_{\theta}^{\infty} p(x, \omega_2)dx$$
$$P(error) = \int_{-\infty}^{\theta} p(x|\omega_1)P(\omega_1)dx + \int_{\theta}^{\infty} p(x|\omega_2)P(\omega_2)dx$$

Como las probabilidades a priori $P(\omega_1)$ y $P(\omega_2)$ son constantes con respecto a $x$, pueden salir de las integrales y tenemos:
$$P(error) = P(\omega_1) \int_{-\infty}^{\theta} p(x|\omega_1)dx + P(\omega_2) \int_{\theta}^{\infty} p(x|\omega_2)dx$$

---

### b

Para encontrar el valor de $\theta$ que minimiza la función $P(error)$, debemos derivar la expresión del inciso (a) con respecto a $\theta$ e igualar a cero (buscar puntos críticos).

$$\frac{d P(error)}{d\theta} = \frac{d}{d\theta} \left[ P(\omega_1) \int_{-\infty}^{\theta} p(x|\omega_1)dx + P(\omega_2) \int_{\theta}^{\infty} p(x|\omega_2)dx \right] = 0$$

Aplicando el Teorema Fundamental del Cálculo para derivar integrales con límites variables:
* La derivada de $\int_{-\infty}^{\theta} f(x)dx$ respecto a $\theta$ es $\lim_{y \rightarrow -\infty} f(\theta) - f(y)$.
* La derivada de $\int_{\theta}^{\infty} g(x)dx$ respecto a $\theta$ es $\lim_{y \rightarrow \infty} g(y) - g(\theta)$.
* Como $p$ es una densidad de probabilidad, sus límites en $\infty$ y $-\infty$ se hacen cero.

Aplicando esto obtenemos:
$$P(\omega_1) p(\theta|\omega_1) - P(\omega_2) p(\theta|\omega_2) = 0$$

Despejando, llegamos a la condición necesaria:
$$p(\theta|\omega_1) P(\omega_1) = p(\theta|\omega_2) P(\omega_2)$$

Esta ecuación nos dice que el umbral de decisión óptimo se encuentra en el punto de intersección de las distribuciones de probabilidad marginales $p(x, \omega_1)$ y $p(x, \omega_2)$.

---

### c

Define la ecuación un valor de $\theta$ único?

**No, no necesariamente.** La ecuaciónsólo encuentra los puntos donde las curvas de las funciones de densidad ponderadas por sus probabilidades a priori se cruzan.

Dependiendo de la forma de las distribuciones $p(x|\omega_i)$:
* Si las distribuciones son multimodales o tienen varianzas significativamente distintas, las curvas pueden cruzarse en **múltiples puntos**, resultando en varios valores posibles que satisfacen la ecuación (algunos podrían ser mínimos locales, otros máximos locales del error).
* Si las distribuciones fueran exactamente iguales, habría **infinitos** valores.
Solamente en casos específicos (por ejemplo, distribuciones normales con la misma varianza) se garantiza un único punto de intersección.

---

### d

Si $P(X|\omega_i) \sim N(\mu_i, \sigma_i)$, sus funciones de densidad de probabilidad son:
$$p(x|\omega_i) = \frac{1}{\sqrt{2\pi}\sigma_i} \exp\left( -\frac{(x-\mu_i)^2}{2\sigma_i^2} \right)$$

Sustituimos esto en la condición del inciso (b):
$$P(\omega_1) \frac{1}{\sqrt{2\pi}\sigma_1} \exp\left( -\frac{(\theta-\mu_1)^2}{2\sigma_1^2} \right) = P(\omega_2) \frac{1}{\sqrt{2\pi}\sigma_2} \exp\left( -\frac{(\theta-\mu_2)^2}{2\sigma_2^2} \right)$$

Para resolver para $\theta$, aplicamos logaritmo natural ($\ln$) a ambos lados:
$$\ln P(\omega_1) - \ln(\sqrt{2\pi}\sigma_1) - \frac{(\theta-\mu_1)^2}{2\sigma_1^2} = \ln P(\omega_2) - \ln(\sqrt{2\pi}\sigma_2) - \frac{(\theta-\mu_2)^2}{2\sigma_2^2}$$

Reordenando los términos para agrupar las potencias de $\theta$, obtenemos una ecuación cuadrática general de la forma $A\theta^2 + B\theta + C = 0$:

$$\left( \frac{1}{2\sigma_2^2} - \frac{1}{2\sigma_1^2} \right) \theta^2 + \left( \frac{\mu_1}{\sigma_1^2} - \frac{\mu_2}{\sigma_2^2} \right) \theta + \left( \frac{\mu_2^2}{2\sigma_2^2} - \frac{\mu_1^2}{2\sigma_1^2} - \ln\left( \frac{P(\omega_1)\sigma_2}{P(\omega_2)\sigma_1} \right) \right) = 0$$

**Análisis de este resultado:**
1.  **Caso de varianzas distintas ($\sigma_1 \neq \sigma_2$):** El coeficiente cuadrático es distinto de cero. Esto significa que podemos tener hasta **dos** raíces para $\theta$ (resolviendo con la fórmula resolvente de Bhaskara), confirmando lo dicho en el inciso (c).
2.  **Caso de varianzas iguales ($\sigma_1 = \sigma_2 = \sigma$):** El término cuadrático se anula ($A=0$) y la ecuación se vuelve **lineal**, dando como resultado un único valor óptimo para $\theta$:
    $$\theta = \frac{\mu_1 + \mu_2}{2} - \frac{\sigma^2}{\mu_1 - \mu_2} \ln\left( \frac{P(\omega_1)}{P(\omega_2)} \right)$$
    *(En este caso simplificado, si las clases son equiprobables $P(\omega_1)=P(\omega_2)$, el término del logaritmo es cero y el umbral $\theta$ se ubica exactamente en el punto medio entre las medias $\mu_1$ y $\mu_2$).*