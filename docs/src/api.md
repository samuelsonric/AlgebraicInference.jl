# Library Reference
## Systems

```@docs
GaussianSystem
GaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Any)

canon(::AbstractMatrix, ::AbstractVector)
canon(::AbstractMatrix)
normal(::AbstractMatrix, ::AbstractVector)
normal(::AbstractMatrix)
normal(::AbstractVector)
kernel(::AbstractMatrix, ::AbstractVector, ::AbstractMatrix)
kernel(::AbstractMatrix, ::AbstractMatrix)
kernel(::AbstractMatrix)

length(::GaussianSystem)
cov(::GaussianSystem)
invcov(::GaussianSystem)
mean(::GaussianSystem)
⊗(::GaussianSystem, ::GaussianSystem)
+(::GaussianSystem, ::GaussianSystem)
*(::GaussianSystem, ::AbstractMatrix)
zero(::GaussianSystem)
pushfwd
marginal

oapply(::UndirectedWiringDiagram, ::AbstractDict{<:Any, <:GaussianSystem})
oapply(::UndirectedWiringDiagram, ::AbstractVector{<:GaussianSystem})
```

## Valuations

```@docs
Valuation
IdentityValuation
LabeledBox

domain
combine
project

inference_problem(::UndirectedWiringDiagram, ::AbstractDict)
inference_problem(::UndirectedWiringDiagram, ::AbstractVector)
```

## Architectures
```@docs
Architecture
architecture
answer_query
answer_query!
```

## Graphs

```@docs
primal_graph
minfill!
minwidth!
```
