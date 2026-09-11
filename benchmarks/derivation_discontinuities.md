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

### Multidimensional counterpart

For a scalar constraint $g:\mathbb{R}^N \to \mathbb{R}$, the corresponding identity is

$$
\int_{\mathbb{R}^N} \varphi(\mathbf{u})\,\delta\!\big(g(\mathbf{u})\big)\,d\mathbf{u}
=
\int_{g=0} \frac{\varphi(\mathbf{u})}{\lVert\nabla g(\mathbf{u})\rVert}\,d\sigma(\mathbf{u}),
$$

where $d\sigma$ is the induced surface measure on the level set $g(\mathbf{u})=0$.
So in multiple dimensions, the denominator is the gradient norm
$\lVert\nabla g\rVert$, not a single derivative component.

<details>
   <summary><b>Proof sketch (multidimensional case)</b></summary>

??

</details>

If one chooses a coordinate (for example $u_N$) as solve variable and writes
$\mathbf{u}=(\mathbf{v},u_N)$ with $\mathbf{v}\in\mathbb{R}^{N-1}$, this is
equivalently

$$
\int_{\mathbb{R}^N} \varphi(\mathbf{u})\,\delta\!\big(g(\mathbf{u})\big)\,d\mathbf{u}
=
\int_{\mathbb{R}^{N-1}}
\sum_{u_N^*: g(\mathbf{v},u_N^*)=0}
\frac{\varphi(\mathbf{v},u_N^*)}{\left|\partial g/\partial u_N\right|(\mathbf{v},u_N^*)}
\,d\mathbf{v},
$$

provided the solved-direction derivative does not vanish at the roots.

## AD for discontinuities

Let's introduce an arbitrary variable $S$ for which we wish to compute the derivative of $P$.
Here we keep the density independent of $S$ and place the parameter dependence in
the switching function $g(x, S)$:

$$P(S) = \int_{-\infty}^{\infty} f(x) H(g(x, S)) \, dx$$

We now derive under the integral sign:

$$\frac{dP}{dS} = \int_{-\infty}^{\infty} f(x) \delta(g(x, S)) \frac{\partial g(x, S)}{\partial S} \, dx$$

This is exactly the discontinuity contribution that standard pathwise AD misses.

Its calculation requires the transformation property of the Dirac delta function. Applying the transformation property, we get:

$$\int_{-\infty}^{\infty} f(x) \frac{\partial g(x, S)}{\partial S} \delta(g(x, S))  \, dx = \int_{-\infty}^{\infty} f(x) \frac{\partial g(x, S)}{\partial S} \sum_{i: g(x_i, S) = 0} \frac{\delta(x - x_i)}{|\frac{\partial g}{\partial x}(x_i, S)|}  \, dx$$

And applying the sifting property we finally have:

$$\int_{-\infty}^{\infty} f(x) \frac{\partial g(x, S)}{\partial S} \delta(g(x, S)) \, dx = \sum_{i: g(x_i, S) = 0} \frac{f(x_i)}{|\frac{\partial g}{\partial x}(x_i, S)|} \frac{\partial g}{\partial S}(x_i, S)$$

For this quantity to be able to be calculated automatically, we would have to:

- represent and evaluate the function $f$. As a reminder, $f$ depends only on the distribution of $x$, so this could be done adding special operators to a class of RNG distributions.
- represent and evaluate the function $g$ and its derivatives. As a reminder, $g$ depends only on the payoff, and can be easily extracted form a scripting language. The derivatives of $g$ can be easily calculated with standard AD techniques.
- find the roots $x_i$ of $g(x, S) = 0$. This can be done using numerical root-finding methods, but could also be alleviated whenever these functions are simple enough to have analytical roots.

## Treatment of discontinuities for multidimensional MC simulations

We start from an $N$-dimensional Monte-Carlo representation.
For a path driven by $N$ random draws, the payoff depends on the full vector
$\mathbf{u} = (u_1,\dots,u_N) \in [0,1]^N$, not on a single scalar variable.

Let the discounted payoff be written as:

$$
P(S) = \int_{[0,1]^N} f(\mathbf{u})\,H\big(g(\mathbf{u}, S)\big)\,d\mathbf{u}.
$$

For a digital call, one can take:

$$
g(\mathbf{u}, S) = S_T(\mathbf{u}, S) - K,
$$

so the switching surface is the hypersurface $g=0$ in $N$ dimensions.

Differentiating with respect to $S$:

For fixed $\mathbf{u}$ with $g(\mathbf{u},S) \neq 0$, the payoff indicator
$H(g(\mathbf{u},S))$ is locally constant in $S$ (equal to 0 on one side of the
switching surface and 1 on the other side). Therefore its ordinary pathwise
derivative is zero away from the surface:

$$
\frac{\partial}{\partial S}H(g(\mathbf{u},S)) = 0 \qquad \text{for } g(\mathbf{u},S)\neq 0.
$$

Hence the regular pathwise contribution vanishes, and only the singular
contribution concentrated on $g=0$ remains.

$$
\frac{dP}{dS} =
\int_{[0,1]^N} f\,\delta(g)\,\frac{\partial g}{\partial S}\,d\mathbf{u}.
$$

As in the 1D case, the second term is the discontinuity contribution.

### Reduce by solving one coordinate

Choose one coordinate as solve variable, for example $u_N$, and keep
$\mathbf{v}=(u_1,\dots,u_{N-1})$ as Monte-Carlo variables.
For fixed $\mathbf{v}$, define:

$$
h_{\mathbf{v}}(u_N) := g(\mathbf{v}, u_N, S).
$$

The Dirac constraint enforces the switching condition, so the quantity to solve
is explicitly

$$
h_{\mathbf{v}}(u_N) = 0
\quad\Longleftrightarrow\quad
g(\mathbf{v},u_N,S)=0.
$$

In this benchmark setting, for fixed $\mathbf{v}$ and admissible parameters,
$h_{\mathbf{v}}$ is monotone in $u_N$ and therefore bijective onto its image;
hence the root map is invertible and the solved value $u_N^*$ is unique whenever
the strike level is reachable.

Then

$$
\int_{[0,1]^N} f\,\delta(g)\,\frac{\partial g}{\partial S}\,d\mathbf{u}
=
\int_{[0,1]^{N-1}} \sum_{u_N^*: h_{\mathbf{v}}(u_N^*)=0}
\frac{f(\mathbf{v},u_N^*)\,\frac{\partial g}{\partial S}(\mathbf{v},u_N^*,S)}
{\left|\frac{\partial g}{\partial u_N}(\mathbf{v},u_N^*,S)\right|}
\,d\mathbf{v}.
$$

So we still target the same original $N$-dimensional integral; the
Dirac constraint analytically removes exactly one integration variable.

### Interpretation of the reduction

The reduction can be read as follows:

1. Keep $u_1,\dots,u_{N-1}$ as random integration coordinates.
2. Solve for $u_N^*$ on the switching surface, that is
   $g(\mathbf{v},u_N^*,S)=0$.
3. Integrate the weighted contribution
   $\frac{\partial g/\partial S}{\left|\partial g/\partial u_N\right|}$
   over the remaining $(N-1)$ coordinates.

If one parameterizes the last coordinate by a normal variable through
$u = \Phi(z)$, then an additional factor involving the normal PDF appears in
$\partial g/\partial u_N$ by chain rule. This is a Jacobian effect and does
not change the underlying multidimensional target integral.

## Black-Scholes specialization of $g$ and $\partial g/\partial S$

In the Black-Scholes digital-call setting, the switching function is

$$
g(\mathbf{u}, S) = S_T(\mathbf{u}, S) - K,
$$

with terminal spot written in multiplicative form as

$$
S_T(\mathbf{u}, S)
= S\prod_{j=1}^{N}
\exp\!\left(\left(r-\tfrac12\sigma^2\right)\Delta t_j
+ \sigma\sqrt{\Delta t_j}\,z_j\right).
$$

Define

$$
A(\mathbf{u}) := \prod_{j=1}^{N}
\exp\!\left(\left(r-\tfrac12\sigma^2\right)\Delta t_j
+ \sigma\sqrt{\Delta t_j}\,z_j\right),
$$

so that $S_T(\mathbf{u}, S)=S\,A(\mathbf{u})$. Therefore

$$
\frac{\partial g}{\partial S}(\mathbf{u}, S)
= \frac{\partial S_T}{\partial S}(\mathbf{u}, S)
= A(\mathbf{u})
= \frac{S_T(\mathbf{u}, S)}{S}.
$$

On the switching surface, where $g=0$ and hence $S_T=K$, this simplifies to

$$
\left.\frac{\partial g}{\partial S}\right|_{g=0} = \frac{K}{S}.
$$

For the denominator in the reduced formula, we also need
$\partial g/\partial u_N$ explicitly. With

$$
z_N = \Phi^{-1}(u_N),
$$

and (for fixed $\mathbf{v}$) last-step dependence

$$
S_T = S_{\mathrm{before}}(\mathbf{v},S)
\exp\!\left(\left(r-\tfrac12\sigma^2\right)\Delta t_N
+ \sigma\sqrt{\Delta t_N}\,z_N\right),
$$

we have

$$
\frac{\partial g}{\partial u_N}
= \frac{\partial S_T}{\partial u_N}
= \frac{\partial S_T}{\partial z_N}\frac{\partial z_N}{\partial u_N}
= S_T\,\sigma\sqrt{\Delta t_N}\,\frac{1}{\phi(z_N)}.
$$

At the solved point on the switching surface ($S_T=K$, $z_N=z_N^*$):

$$
\left.\frac{\partial g}{\partial u_N}\right|_{g=0}
= \frac{K\,\sigma\sqrt{\Delta t_N}}{\phi(z_N^*)}.
$$

This gives a clear theoretical flow:

1. Start from the full multidimensional discontinuity integral.
2. Use Dirac reduction along one chosen coordinate.
3. Monte-Carlo average over the remaining coordinates.
4. Keep the same mathematical object as the original $N$-dimensional model.

### Connection to the implemented estimator

The reduced identity is applied pathwise in the following form: for each fixed
$\mathbf{v}=(u_1,\dots,u_{N-1})$, solve $u_N^*$ from
$g(\mathbf{v},u_N,S)=0$, then accumulate

$$
w(\mathbf{v})
=
\frac{\frac{\partial g}{\partial S}(\mathbf{v},u_N^*,S)}
{\left|\frac{\partial g}{\partial u_N}(\mathbf{v},u_N^*,S)\right|}.
$$

In the Black-Scholes digital-call case,

$$
\left.\frac{\partial g}{\partial S}\right|_{g=0}=\frac{K}{S},
\qquad
\left.\frac{\partial g}{\partial u_N}\right|_{g=0}
=\frac{K\,\sigma\sqrt{\Delta t_N}}{\phi(z_N^*)}.
$$

Therefore

$$
w(\mathbf{v})
=
\frac{K/S}{K\,\sigma\sqrt{\Delta t_N}/\phi(z_N^*)}
=
\frac{\phi(z_N^*)}{S\,\sigma\sqrt{\Delta t_N}}.
$$

This is exactly the quantity represented in the implementation by the product
of a numerator term for $\partial g/\partial S$ and the inverse absolute
denominator term for $\partial g/\partial u_N$ after solving the last draw.
