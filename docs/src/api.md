# Library Reference

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

## Systems

```@docs
GaussianSystem

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
