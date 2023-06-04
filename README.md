# AlgebraicInference.jl

[![Build Status](https://github.com/samuelsonric/AlgebraicInference.jl/workflows/Tests/badge.svg)](https://github.com/samuelsonric/AlgebraicInference.jl/actions?query=workflow%3ATests)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://samuelsonric.github.io/AlgebraicInference.jl/dev/)
[![Code Coverage](https://codecov.io/gh/samuelsonric/AlgebraicInference.jl/branch/master/graph/badge.svg?token=FJJQQCTUCF)](https://codecov.io/gh/samuelsonric/AlgebraicInference.jl)

AlgebraicInference.jl is a library for performing Bayesian inference on wiring diagrams. It
builds on [Catlab.jl](https://algebraicjulia.github.io/Catlab.jl/dev/). See the
[documentation](https://samuelsonric.github.io/AlgebraicInference.jl/dev/) for example
notebooks and an API.

## Gaussian Systems

Gaussian systems were introduced by Jan Willems in his 2013 article *Open Stochastic
Systems*. A probability space $\Sigma = (\mathbb{R}^n, \mathcal{E}, P)$ is called an
$n$-variate Gaussian system with fiber $\mathbb{L} \subseteq \mathbb{R}^n$ if it is
isomorphic to a Gaussian measure on the quotient space $\mathbb{R}^n / \mathbb{L}$.

If $\mathbb{L} = \{0\}$, then $\Sigma$ is an $n$-variate normal distribution.

Every $n$-variate Gaussian system $\Sigma$ corresponds to a convex *energy function* 
$E: \mathbb{R}^n \to (0, \infty]$ of the form
```math
    E(x) = \begin{cases}
        \frac{1}{2} x^\mathsf{T} P x - x^\mathsf{T} p & Sx = s \\
        \infty                                        & \text{else},
    \end{cases}
```
where $P$ and $S$ are positive semidefinite matrices, $p \in \mathtt{image}(P)$, and
$s \in \mathtt{image}(S)$.

If $\Sigma$ is an $n$-variate normal distribution, then $E$ is its negative
log-density.

## Hypergraph Categories

There exists a hypergraph PROP whose morphisms $m \to n$ are $m + n$-variate Gaussian
systems. Hence, Gaussian systems can be composed using undirected wiring diagrams.

![inference](./inference.svg)

These wiring diagrams look like
[undirected graphical models](https://en.wikipedia.org/wiki/Graphical_model). One difference
is that wiring diagrams can contain half-edges, which specify which variables are
marginalized out during computation. Hence, a wiring diagram can be thought of as an
*inference problem*: a graphical model paired with a query.
