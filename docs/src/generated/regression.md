```@meta
EditURL = "../../literate/regression.jl"
```

# Linear Regression

````@example regression
using AlgebraicInference
using Catlab.Graphics, Catlab.Programs
using FillArrays
using LinearAlgebra
using StatsPlots
````

## Frequentist Linear Regression
Consider the Gauss-Markov linear model
```math
    y = X \beta + \epsilon,
```
where ``X`` is an ``n \times m`` matrix, ``\beta`` is an ``m \times 1`` vector, and
``\epsilon`` is an ``n \times 1`` normally distributed random vector with mean
``\mathbf{0}`` and covariance ``W``. If ``X`` has full column rank, then the best linear
unbiased estimator for ``\beta`` is the random vector
```math
    \hat{\beta} = X^+ (I - (Q W Q)^+ Q W)^\mathsf{T} y,
```
where ``X^+`` is the Moore-Penrose psuedoinverse of ``X``, and
```math
Q = I - X X^+.
```

References:
- Albert, Arthur. "The Gauss-Markov Theorem for Regression Models with Possibly Singular
  Covariances." *SIAM Journal on Applied Mathematics*, vol. 24, no. 2, 1973, pp. 182–87.

````@example regression
X = [
    1  0
    0  1
    0  0
]

W = [
    0  0  0
    0  1 .5
    0 .5  1
]

y = [
    1
    1
    1
]

Q = I - X * pinv(X)
β̂ = pinv(X) * (I - pinv(Q * W * Q) * Q * W)' * y
round.(β̂; digits=4)
````

To solve for ``\hat{\beta}`` using AlgebraicInference.jl, we construct an undirected
wiring diagram.

````@example regression
wd = @relation (a,) where (a::m, b::n, c::n, d::n) begin
    X(a, b)
    +(b, c, d)
    ϵ(c)
    y(d)
end

to_graphviz(wd; box_labels=:name, implicit_junctions=true)
````

Then we assign values to the boxes in `wd` and compute the result.

````@example regression
P = [
    1  0  0  1  0  0
    0  1  0  0  1  0
    0  0  1  0  0  1
]

hom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(
    :X => kernel(X, Zeros(3), Zeros(3, 3)),
    :+ => kernel(P, Zeros(3), Zeros(3, 3)),
    :ϵ => normal(Zeros(3), W),
    :y => normal(y, Zeros(3, 3)))

ob_map = Dict(
    :m => 2,
    :n => 3)

problem = InferenceProblem(wd, hom_map, ob_map)

Σ̂ = solve(problem)

β̂ = mean(Σ̂)

round.(β̂; digits=4)
````

## Bayesian Linear Regression
Let ``\rho = \mathcal{N}(m, V)`` be our prior belief about ``\beta``. Then our posterior
belief ``\hat{\rho}`` is a bivariate normal distribution with mean
```math
  \hat{m} = m - V X^\mathsf{T} (X V X^\mathsf{T} + W)^+ (X m - y)
```
and covariance
```math
  \hat{V} = V - V X^\mathsf{T} (X V X^\mathsf{T} + W)^+ X V.
```

````@example regression
V = [
    1  0
    0  1
]

m = [
    0
    0
]

m̂ = m - V * X' * pinv(X * V * X' + W) * (X * m - y)

round.(m̂; digits=4)
````

````@example regression
V̂ = V - V * X' * pinv(X * V * X' + W) * X * V

round.(V̂; digits=4)
````

To solve for ``\hat{\rho}`` using AlgebraicInference.jl, we construct an undirected
wiring diagram.

````@example regression
wd = @relation (a,) where (a::m, b::n, c::n, d::n) begin
    ρ(a)
    X(a, b)
    +(b, c, d)
    ϵ(c)
    y(d)
end

to_graphviz(wd; box_labels=:name, implicit_junctions=true)
````

Then we assign values to the boxes in `wd` and compute the result.

````@example regression
hom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(
    :ρ => normal(m, V),
    :X => kernel(X, Zeros(3), Zeros(3, 3)),
    :+ => kernel(P, Zeros(3), Zeros(3, 3)),
    :ϵ => normal(Zeros(3), W),
    :y => normal(y, Zeros(3, 3)))

ob_map = Dict(
    :m => 2,
    :n => 3)

problem = InferenceProblem(wd, hom_map, ob_map)

Σ̂ = solve(problem)

m̂ = mean(Σ̂)

round.(m̂; digits=4)
````

````@example regression
V̂ = cov(Σ̂)

round.(V̂; digits=4)
````

````@example regression
plot()
covellipse!(m, V, aspect_ratio=:equal, label="prior")
covellipse!(m̂, V̂, aspect_ratio=:equal, label="posterior")
````

