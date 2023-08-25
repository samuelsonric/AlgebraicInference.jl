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

InferenceProblem{T₁, T₂, T₃}(::AbstractUWD, ::AbstractDict, ::AbstractDict) where {T₁, T₂, T₃}
InferenceProblem{T₁, T₂, T₃}(::AbstractUWD, ::AbstractVector, ::AbstractVector) where {T₁, T₂, T₃}
InferenceProblem{T₁, T₂, T₃}(::BayesNet, ::AbstractVector, ::AbstractDict) where {T₁, T₂, T₃}

solve(::InferenceProblem, alg::EliminationAlgorithm)
init(::InferenceProblem, alg::EliminationAlgorithm)
```

## Solvers

```@docs
InferenceSolver

solve(::InferenceSolver)
solve!(::InferenceSolver)
```

## Algorithms

```@docs
EliminationAlgorithm
MinDegree
MinFill
```

