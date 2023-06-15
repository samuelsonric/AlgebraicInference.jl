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
MinDegree
MinFill

InferenceProblem{T}(::AbstractUWD, ::AbstractDict, ::Union{Nothing, AbstractDict}) where T
InferenceProblem{T}(::AbstractUWD, ::AbstractVector, ::Union{Nothing, AbstractVector}) where T

solve(::InferenceProblem, alg)
init(::InferenceProblem, alg)
```

## Solvers

```@docs
InferenceSolver

solve(::InferenceSolver)
solve!(::InferenceSolver)
```
