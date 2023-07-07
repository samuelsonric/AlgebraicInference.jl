# Library Reference
## Systems

```@docs
GaussianSystem
GaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Real)

normal
kernel

length(::GaussianSystem)
cov(::GaussianSystem)
invcov(::GaussianSystem)
var(::GaussianSystem)
mean(::GaussianSystem)

oapply(::AbstractUWD, ::AbstractVector{<:GaussianSystem}, ::AbstractVector)
```

## Problems
```@docs
InferenceProblem
MinDegree
MinFill

InferenceProblem{T₁, T₂}(::AbstractUWD, ::AbstractDict, ::AbstractDict) where {T₁, T₂}
InferenceProblem{T₁, T₂}(::AbstractUWD, ::AbstractVector, ::AbstractVector) where {T₁, T₂}
InferenceProblem{T₁, T₂}(::BayesNet, ::AbstractVector, ::AbstractDict) where {T₁, T₂}

solve(::InferenceProblem, alg)
init(::InferenceProblem, alg)
```

## Solvers

```@docs
InferenceSolver

solve(::InferenceSolver)
solve!(::InferenceSolver)
```
