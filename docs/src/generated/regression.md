```@meta
EditURL = "<unknown>/literate/regression.jl"
```

# Linear Regression

````@example regression
using AlgebraicInference
using Catlab, Catlab.Graphics, Catlab.Programs
using LinearAlgebra
using StatsPlots
````

## Frequentist Linear Regression
Consider the Gauss-Markov linear model
```math
    y = X \beta + \epsilon,
```
where ``X`` is a ``n \times m`` matrix, ``\beta`` is an ``m \times 1`` vector, and ``\epsilon`` is an ``n \times 1`` normally distributed random vector with mean ``0`` and covariance ``W``. If ``X`` has full column rank, then the best linear unbiased estimator for ``\beta`` is the random variable
```math
    \hat{\beta} = X^+ (I - (Q W Q)^+ Q W)^\mathsf{T} y,
```
where ``X^+`` is the Moore-Penrose psuedoinverse of ``X``, and
```math
Q = I - X X^+.
```

References:
- Albert, Arthur. "The Gauss-Markov Theorem for Regression Models with Possibly Singular Covariances." *SIAM Journal on Applied Mathematics*, vol. 24, no. 2, 1973, pp. 182–87.

````@example regression
X = [ 1 0
      0 1
      0 0 ]

L = [ 1 0 0
      0 1 0
      0 0 1 ]

y = [ 1
      1
      1 ]

W = L * L'
Q = I - X * pinv(X)
β̂ = pinv(X) * (I - pinv(Q * W * Q) * Q * W)' * y
````

To solve for ``\hat{\beta}`` using `AlgebraicInference.jl`, we construct an undirected wiring diagram.

````@example regression
diagram = @relation (a₁, a₂) begin
    X(a₁, a₂, b₁, b₂, b₃)
    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)
    ϵ(c₁, c₂, c₃)
    y(d₁, d₂, d₃)
end

to_graphviz(diagram;
    box_labels         = :name,
    implicit_junctions = true,
)
````

Then we assign values to the boxes in `diagram` and compute the result.

````@example regression
P = [ 1 0 0 1 0 0
      0 1 0 0 1 0
      0 0 1 0 0 1 ]

hom_map = Dict(
    :X => System([-X I]),
    :+ => System([-P I]),
    :ϵ => ClassicalSystem(L),
    :y => ClassicalSystem(y),
)

β̂ = mean(oapply(diagram, hom_map))
````

## Bayesian Linear Regression
Let ``\rho = \mathcal{N}(m, V)`` be our prior belief about ``\beta``. Then our posterior belief ``\hat{\rho}`` is a bivariate normal distribution with mean
```math
  \hat{m} = m - V X^\mathsf{T} (X V X' + W)^+ (X m - y)
```
and covariance
```math
  \hat{V} = V - V X^\mathsf{T} (X V X' + W)^+ X V.
```

````@example regression
M = [ 1 0
      0 1 ]

m = [ 0
      0 ]

V = M * M'
m̂ = m - V * X' * pinv(X * V * X' + W) * (X * m - y)
````

````@example regression
V̂ = V - V * X' * pinv(X * V * X' + W) * X * V
````

To solve for ``\hat{\rho}`` using `AlgebraicInference.jl`, we construct an undirected wiring diagram.

````@example regression
diagram = @relation (a₁, a₂) begin
    ρ(a₁, a₂)
    X(a₁, a₂, b₁, b₂, b₃)
    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)
    ϵ(c₁, c₂, c₃)
    y(d₁, d₂, d₃)
end

to_graphviz(diagram;
    box_labels         = :name,
    implicit_junctions = true,
)
````

Then we assign values to the boxes in `diagram` and compute the result.

````@example regression
hom_map = Dict(
    :ρ => ClassicalSystem(M, m),
    :X => System([-X I]),
    :+ => System([-P I]),
    :ϵ => ClassicalSystem(L),
    :y => ClassicalSystem(y),
)

m̂ = mean(oapply(diagram, hom_map))
````

````@example regression
V̂ = cov(oapply(diagram, hom_map))
````

````@example regression
covellipse!(m, V, aspect_ratio=:equal, label="prior")
covellipse!(m̂, V̂, aspect_ratio=:equal, label="posterior")
````

