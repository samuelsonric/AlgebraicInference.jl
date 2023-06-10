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

oapply(::UndirectedWiringDiagram, ::AbstractDict{<:Any, <:GaussianSystem})
oapply(::UndirectedWiringDiagram, ::AbstractVector{<:GaussianSystem})
```

## Problems
```@docs
InferenceProblem
UWDProblem
MinWidth
MinFill

UWDProblem{T}(wd::AbstractUWD, bs) where T
UWDProblem{T}(wd::AbstractUWD, bm::AbstractDict) where T

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
