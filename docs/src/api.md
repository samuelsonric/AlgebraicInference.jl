# Library Reference

## Gaussian Systems

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
oapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: AbstractSystem}
oapply(composite::UndirectedWiringDiagram, boxes::AbstractVector{T}) where T <: AbstractSystem
```

## Valuations

```@docs
Valuation
LabeledBox

LabeledBox(::Any, ::OrderedSet)

d(::Valuation)
⊗(::Valuation, ::Valuation)
↓(::Valuation, ::AbstractSet)
-(::Valuation, ::Any)

construct_inference_problem(::UndirectedWiringDiagram, ::AbstractDict)
construct_inference_problem(::UndirectedWiringDiagram, ::AbstractVector)
construct_elimination_sequence(::AbstractSet{T}, ::AbstractSet) where T <: AbstractSet
fusion_algorithm(::AbstractSet{T}, ::Any) where T <: Valuation
```
