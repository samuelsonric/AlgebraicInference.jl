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
