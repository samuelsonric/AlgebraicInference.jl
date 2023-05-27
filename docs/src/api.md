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
AbstractSystem
AbstractProgram
ClosedProgram
OpenProgram
System

ClosedProgram(::AbstractMatrix, ::AbstractVector)
ClosedProgram(::AbstractMatrix)
ClosedProgram(::AbstractVector)
OpenProgram(::ClosedProgram, ::AbstractMatrix, ::Int)
OpenProgram(::ClosedProgram, ::AbstractMatrix)
OpenProgram(::AbstractMatrix)
System(::ClosedProgram, ::AbstractMatrix)
System(::AbstractMatrix)

length(::AbstractSystem)
dof(::AbstractSystem)
fiber(::AbstractSystem)
mean(::AbstractSystem)
cov(::AbstractSystem)
*(::AbstractMatrix, ::AbstractSystem)
\(::AbstractMatrix, ::AbstractSystem)
⊗(::AbstractSystem, ::AbstractSystem)
oapply(::UndirectedWiringDiagram, ::AbstractDict{<:Any, <:AbstractSystem})
oapply(::UndirectedWiringDiagram, ::AbstractVector{<:AbstractSystem})
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
