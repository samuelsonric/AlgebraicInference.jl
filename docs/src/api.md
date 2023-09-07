# Library Reference
## Systems

```@docs
GaussianSystem
CanonicalForm

GaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Real)
CanonicalForm(::AbstractMatrix, ::AbstractVector)

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

InferenceProblem(::AbstractUWD, ::AbstractDict, ::AbstractDict)
InferenceProblem(::AbstractUWD, ::AbstractVector, ::AbstractVector)
InferenceProblem(::BayesNet, ::AbstractVector, ::AbstractDict)

solve(::InferenceProblem, alg::EliminationAlgorithm)
init(::InferenceProblem, alg::EliminationAlgorithm)
```

## Solvers

```@docs
InferenceSolver

solve!(::InferenceSolver)
```

## Algorithms

```@docs
EliminationAlgorithm
MinDegree
MinFill
```

