# Library Reference

## Graphs

```@docs
construct_elimination_sequence
construct_join_tree
```

## Systems

```@docs
AbstractSystem
ClassicalSystem
System

ClassicalSystem(::AbstractMatrix, ::AbstractVector)
ClassicalSystem(::AbstractMatrix)
ClassicalSystem(::AbstractVector)
System(::AbstractMatrix, ::ClassicalSystem)
System(::AbstractMatrix)

length(::AbstractSystem)
dof(::AbstractSystem)
fiber(::AbstractSystem)
mean(::AbstractSystem)
cov(::AbstractSystem)
*(::AbstractMatrix, ::AbstractSystem)
\(::AbstractMatrix, ::AbstractSystem)
⊗(::AbstractSystem, ::AbstractSystem)
oapply(::UndirectedWiringDiagram, ::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}
oapply(::UndirectedWiringDiagram, ::AbstractVector{T}) where T <: AbstractSystem
```

## Valuations

```@docs
Variable
Valuation
LabeledBoxVariable
IdentityValuation
LabeledBox

domain
combine
project
neutral_valuation
eliminate

construct_inference_problem(::Type, ::UndirectedWiringDiagram, ::AbstractDict)
construct_inference_problem(::Type, ::UndirectedWiringDiagram, ::AbstractVector)
construct_factors
fusion_algorithm
collect_algorithm
shenoy_shafer_architecture!
```
