# Library Reference
## Systems

```@docs
GaussianSystem
GaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Any)

normal
kernel

length(::GaussianSystem)
cov(::GaussianSystem)
invcov(::GaussianSystem)
var(::GaussianSystem)
mean(::GaussianSystem)

oapply(::AbstractUWD, ::AbstractVector{<:GaussianSystem})
```

## Problems
```@docs
InferenceProblem
UWDProblem
MinWidth
MinFill

UWDProblem{T}(::AbstractUWD, ::AbstractDict, ::Union{Nothing, AbstractDict}) where T
UWDProblem{T}(::AbstractUWD, ::Any, ::Any) where T

solve(::InferenceProblem, alg)
init(::InferenceProblem, alg)
```

## Solvers

```@docs
InferenceSolver
UWDSolver

solve(::InferenceSolver)
solve!(::InferenceSolver)
```
