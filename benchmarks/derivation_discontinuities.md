# Derivation of discontinuities for a computer program using Monte-Carlo integration

When using Monte-Carlo integration to price options, discontinuities in the payoff function can lead to inaccuracies. To handle these discontinuities, we can use the Dirac delta function to represent the points of discontinuity. Let's assume we have a scripting language that can recognize and manipulate the Dirac, Heavyside, Ramp and higher-order Ramp functions, and also distributions of random number generators.

In the most general setup, in finance, let's assume we have a random variable $x$ with probability density function (PDF) $f(x)$, and a payoff function with potential discontinuitites. At This point it is important to distinguish between the Heavyside step function $H(x)$, which is discontinuous at $x=0$, and the Dirac delta function $\delta(x)$, which is zero everywhere except at
$x=0$ where it is infinite, and integrates to 1 over the entire real line. The relationship between these two functions is given by:

$$\frac{d}{dx}H(x) = \delta(x)$$

In general, Monte carlo simulations handle Heavyside step functions well, but miss any contribution by the Dirac function at the discontinuity. To account for this, we can express the integral of a function $f(x)$ multiplied by a Heavyside function $H(g(x))$ as follows:

$$P =\int_{-\infty}^{\infty} f(x) H(g(x)) \, dx$$

## The General Transformation Property

When the argument of the Dirac delta is a more general function $g(x)$, we have:

$$\boxed{\delta(g(x)) = \sum_{i: g(x_i) = 0} \frac{\delta(x - x_i)}{|g'(x_i)|}}$$

where the sum is over all zeros $x_i$ of $g(x)$, and $g'(x_i)$ is the derivative of $g$ at $x_i$.

<details>
  <summary><b>Proof</b></summary>

This follows from the change of variables formula. If we make a substitution $u = g(x)$, then $du = g'(x) dx$.

**Case 1: $g'(x) > 0$ (increasing function)**

When $g$ is increasing, $du = g'(x) dx$ with $g'(x) > 0$:
$$\int f(x) \delta(g(x)) \, dx = \int f(x(u)) \delta(u) \frac{du}{g'(x(u))} = \int f(x(u)) \delta(u) \frac{1}{g'(x(u))} \, du = \frac{f(x_0)}{g'(x_0)}$$

**Case 2: $g'(x) < 0$ (decreasing function)**

When $g$ is decreasing, $du = g'(x) dx$ with $g'(x) < 0$, which means $dx = \frac{du}{g'(x)}$ where $g'(x) < 0$.

The limits of integration **flip** when $g$ is decreasing. If $x: a \to b$, then $u = g(x): g(a) \to g(b)$ goes backwards when $g$ is decreasing. To maintain the integral over $u$ from $-\infty$ to $+\infty$, we must introduce a negative sign:

$$\int_a^b f(x) \delta(g(x)) \, dx = \int_{g(a)}^{g(b)} f(x(u)) \delta(u) \frac{1}{g'(x(u))} \, du = -\int_{g(b)}^{g(a)} f(x(u)) \delta(u) \frac{1}{g'(x(u))} \, du$$

But since $g'(x) < 0$, we have $\frac{1}{g'(x(u))} < 0$, so:
$$= -\int f(x(u)) \delta(u) \frac{1}{g'(x(u))} \, du = -\frac{f(x_0)}{g'(x_0)} = -\frac{f(x_0)}{-|g'(x_0)|} = \frac{f(x_0)}{|g'(x_0)|}$$

</details>

## AD for discontinuities

Let's introduce an arbitrary variable $S$ for which we whish to compute the derivative of $P$. We can now rewrite $P$ as $P(S)$, assuming depedendence on $S$ through $f(x, S)$ and/or $g(x, S)$:

$$P(S) = \int_{-\infty}^{\infty} f(x, S) H(g(x, S)) \, dx$$

We now derive under the integral sign:

$$\frac{dP}{dS} = \int_{-\infty}^{\infty} \left( \frac{\partial f(x, S)}{\partial S} H(g(x, S)) + f(x, S) \delta(g(x, S)) \frac{\partial g(x, S)}{\partial S} \right) dx$$

Separating the two terms, we have:

$$\frac{dP}{dS} = \int_{-\infty}^{\infty} \frac{\partial f(x, S)}{\partial S} H(g(x, S)) \, dx + \int_{-\infty}^{\infty} f(x, S) \delta(g(x, S)) \frac{\partial g(x, S)}{\partial S} \, dx$$

The first term can be computed using standard automatic differentiation (AD) techniques within a Monte-Carlo simulation, while the contribution of the second term is completely missed by such a simulation.

Its calculation requires the transformation property of the Dirac delta function. Applying the transformation property, we get:

$$\int_{-\infty}^{\infty} f(x, S) \frac{\partial g(x, S)}{\partial S} \delta(g(x, S))  \, dx = \int_{-\infty}^{\infty} f(x, S) \frac{\partial g(x, S)}{\partial S} \sum_{i: g(x_i, S) = 0} \frac{\delta(x - x_i)}{|\frac{\partial g}{\partial x}(x_i, S)|}  \, dx$$

Adn applying the sifting property we finally have:

$$\int_{-\infty}^{\infty} f(x, S) \frac{\partial g(x, S)}{\partial S} \delta(g(x, S)) \, dx = \sum_{i: g(x_i, S) = 0} \frac{f(x_i, S)}{|\frac{\partial g}{\partial x}(x_i, S)|} \frac{\partial g}{\partial S}(x_i, S)$$

For this quantity to be able to be calculated automatically, we would have to:

- represent and evaluate the function $f$. As a reminder, $f$ depends only on the distribution of $x$, so this could be done adding special operators to a class of RNG distributions.
- represent and evaluate the function $g$ and its derivatives. As a reminder, $g$ depends only on the payoff, and can be easily extracted form a scripting language. The derivatives of $g$ can be easily calculated with standard AD techniques.
- find the roots $x_i$ of $g(x, S) = 0$. This can be done using numerical root-finding methods, but could also be alleviated whenever these functions are simple enough to have analytical roots.
