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

InferenceProblem{T₁, T₂, T₃, T₄}(::AbstractUWD, ::AbstractDict, ::AbstractDict) where {T₁, T₂, T₃, T₄}
InferenceProblem{T₁, T₂, T₃, T₄}(::AbstractUWD, ::AbstractVector, ::AbstractVector) where {T₁, T₂, T₃, T₄}
InferenceProblem{T₁, T₂, T₃, T₄}(::BayesNet, ::AbstractVector, ::AbstractDict) where {T₁, T₂, T₃, T₄}

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

