## Part 1

We want to prove

$$
\boxed{
\int_{\mathbb R^n} f(\mathbf x)\,\delta(g(\mathbf x))\,d\mathbf x
=
\int_{g^{-1}(0)}
\frac{f(\mathbf x)}{\|\nabla g(\mathbf x)\|}\,dS
}
$$

where

$$
g:\mathbb R^n\to\mathbb R.
$$

I’ll use bold letters for vectors.

⸻

1. Pick a point on the zero level surface

Let

$$
\mathbf x_0\in\mathbb R^n
$$

be a point satisfying

$$
g(\mathbf x_0)=0.
$$

Write the vector explicitly as

$$
\mathbf x_0
=
(x_{0,1},x_{0,2},\ldots,x_{0,n}).
$$

Thus

$$
g(\mathbf x_0)
=
g(x_{0,1},x_{0,2},\ldots,x_{0,n})
=
0.
$$

The collection of all such points is

$$
S=g^{-1}(0)
=
\{\mathbf x\in\mathbb R^n:g(\mathbf x)=0\}.
$$

This is our (n-1)-dimensional surface.

Assume

$$
\nabla g(\mathbf x)\neq0
$$

everywhere on S.

⸻

2. Choose one coordinate in which g changes

Since

$$
\nabla g(\mathbf x_0)
=
\left(
\frac{\partial g}{\partial x_1}(\mathbf x_0),
\ldots,
\frac{\partial g}{\partial x_n}(\mathbf x_0)
\right)
\neq0,
$$

at least one partial derivative is nonzero.

For simplicity, suppose

$$
\frac{\partial g}{\partial x_n}(\mathbf x_0)\neq0.
$$

Now we temporarily write a generic vector

$$
\mathbf x=(x_1,\ldots,x_n)\in\mathbb R^n.
$$

Here the x_i’s are simply the scalar coordinates of the vector $\mathbf x$.

The implicit-function theorem says that, near $\mathbf x_0$, the equation

$$
g(x_1,\ldots,x_n)=0
$$

can be solved for x_n:

$$
x_n=h(x_1,\ldots,x_{n-1}).
$$

So locally our surface S is

$$
\mathbf X(u_1,\ldots,u_{n-1})
=
\left(
u_1,\ldots,u_{n-1},
h(u_1,\ldots,u_{n-1})
\right).
$$

I’m deliberately using u_i here so we don’t confuse the coordinates of a generic point with the coordinates of $\mathbf x_0$.

⸻

3. Now make g one of our coordinates

Instead of describing a point by

$$
(x_1,\ldots,x_n),
$$

define new coordinates

$$
(u_1,\ldots,u_{n-1},t)
$$

by

$$
u_i=x_i,\qquad i=1,\ldots,n-1,
$$

and

$$
t=g(x_1,\ldots,x_n).
$$

In other words,

$$
\boxed{
t=g(\mathbf x).
}
$$

The inverse transformation can be written

$$
\mathbf x
=
\mathbf X(u_1,\ldots,u_{n-1},t).
$$

The important point is:

$$
t=0
\quad\Longleftrightarrow\quad
g(\mathbf x)=0.
$$

So the zero level surface is precisely the slice t=0 in these new coordinates.

⸻

4. Apply the ordinary change-of-variables theorem

The Jacobian of the transformation

$$
\mathbf x\mapsto
(u_1,\ldots,u_{n-1},t)
$$

is

$$
\frac{\partial(u_1,\ldots,u_{n-1},t)}
{\partial(x_1,\ldots,x_n)}
=
\frac{\partial g}{\partial x_n}.
$$

Therefore

$$
d\mathbf x
=
\frac{1}
{\left|\frac{\partial g}{\partial x_n}\right|}
\,du_1\cdots du_{n-1}\,dt.
$$

Consequently,

$$
\int f(\mathbf x)\delta(g(\mathbf x))\,d\mathbf x
$$

becomes locally

$$
\int
f(\mathbf X(u,t))
\delta(t)
\frac{1}{|g_{x_n}(\mathbf X(u,t))|}
\,du\,dt.
$$

And now we use ordinary one-dimensional sifting:

$$
\int\delta(t)F(t)\,dt=F(0).
$$

Hence

$$
=
\int
\frac{
f(\mathbf X(u,0))
}{
|g_{x_n}(\mathbf X(u,0))|
}
\,du.
$$

## Part 2

Let’s continue exactly from there.

We have

$$
I=
\int
f\!\left(\Phi^{-1}(u,t)\right)
\delta(t)
\left|\det D\Phi^{-1}(u,t)\right|
\,du\,dt.
$$

Here $u=(u_1,\ldots,u_{n-1})$.

1. Do the t-integration

Define

$$
F(u,t)
=
f\!\left(\Phi^{-1}(u,t)\right)
\left|\det D\Phi^{-1}(u,t)\right|.
$$

Then

$$
I=\int_{\mathbb R^{n-1}}\int_{\mathbb R}
F(u,t)\delta(t)\,dt\,du.
$$

Now the ordinary one-dimensional sifting property gives

$$
\int_{\mathbb R}F(u,t)\delta(t)\,dt=F(u,0).
$$

Therefore

$$
I=
\int_{\mathbb R^{n-1}}
f\!\left(\Phi^{-1}(u,0)\right)
\left|\det D\Phi^{-1}(u,0)\right|
\,du.
$$

This is already a completely legitimate result. Now we have to show that it is the desired surface integral.

⸻

2. What is \Phi^{-1}(u,0)?

Recall

$$
\Phi^{-1}(u,t)
=
\left(
u_1,\ldots,u_{n-1},
h(u,t)
\right),
$$

where

$$
g(u_1,\ldots,u_{n-1},h(u,t))=t.
$$

Therefore at t=0,

$$
\Phi^{-1}(u,0)
=
\left(
u_1,\ldots,u_{n-1},
h(u,0)
\right).
$$

But

$$
g(u_1,\ldots,u_{n-1},h(u,0))=0.
$$

So define the parametrisation

$$
\boxed{
X(u)=\Phi^{-1}(u,0).
}
$$

Then X(u) parametrises the piece of the surface

$$
g(\mathbf x)=0.
$$

Thus

$$
I=
\int
f(X(u))
\left|\det D\Phi^{-1}(u,0)\right|
\,du.
$$

⸻

3. Calculate the Jacobian

This is where the gradient enters.

Since

$$
\Phi(\mathbf x)
=
(x_1,\ldots,x_{n-1},g(\mathbf x)),
$$

its Jacobian matrix is

$$
D\Phi=
\begin{pmatrix}
1&0&\cdots&0&0\\
0&1&\cdots&0&0\\
&&\ddots&&\\
0&0&\cdots&1&0\\
g_{x_1}&g_{x_2}&\cdots&g_{x_{n-1}}&g_{x_n}
\end{pmatrix}.
$$

Hence

$$
\det D\Phi=g_{x_n}.
$$

Since

$$
D\Phi^{-1}=(D\Phi)^{-1},
$$

we have

$$
\left|\det D\Phi^{-1}\right|
=
\frac1{|g_{x_n}|}.
$$

Therefore

$$
I=
\int
\frac{f(X(u))}
{|g_{x_n}(X(u))|}
\,du.
$$

XXX

⸻

4. Now relate du to surface area

This is the last step.

The surface parametrisation is

X(u)
=
(u_1,\ldots,u_{n-1},h(u,0)).

Its surface-area element is

$$
dS
=
\sqrt{\det(DX^T DX)}\,du.
$$

For this particular parametrisation,

$$
dS
=
\sqrt{1+\|\nabla h\|^2}\,du.
$$

Since

$$
g(u,h(u,0))=0,
$$

differentiating gives

$$
\frac{\partial h}{\partial u_i}
=
-\frac{g_{x_i}}{g_{x_n}}.
$$

Consequently,

$$
1+\|\nabla h\|^2
=
\frac{
g_{x_1}^2+\cdots+g_{x_n}^2
}{
g_{x_n}^2
}
=
\frac{\|\nabla g\|^2}{g_{x_n}^2}.
$$

Thus

$$
dS
=
\frac{\|\nabla g\|}{|g_{x_n}|}\,du,
$$

or equivalently

$$
\frac{du}{|g_{x_n}|}
=
\frac{dS}{\|\nabla g\|}.
$$

Substitute this into the previous expression:

$$
I
=
\int
f(X(u))
\frac{dS}{\|\nabla g(X(u))\|}.
$$

Since X(u) runs over the level surface g(\mathbf x)=0,

$$
\boxed{
\displaystyle
\int_{\mathbb R^n}
f(\mathbf x)\delta(g(\mathbf x))\,d\mathbf x
=
\int_{g(\mathbf x)=0}
\frac{f(\mathbf x)}
{\|\nabla g(\mathbf x)\|}
\,dS.
}
$$

So the logical chain is:

$$
\boxed{
\begin{aligned}
\text{change variables: }&t=g(\mathbf x)\\
&\Downarrow\\
\delta(g(\mathbf x))&\to\delta(t)\\
&\Downarrow\\
\text{1-D sifting in }t\\
&\Downarrow\\
\text{integral over }u\\
&\Downarrow\\
\text{convert }du\text{ to }dS\\
&\Downarrow\\
\frac{1}{\|\nabla g\|}\text{ appears.}
\end{aligned}}
$$

The part that is easy to miss is that the Jacobian 1/|g_{x_n}| is not yet 1/\|\nabla g\|. The latter only appears when we convert the remaining n-1 coordinate measure du into the intrinsic surface measure dS. That’s the geometric heart of the proof.
