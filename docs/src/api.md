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

LabeledSystem

LabeledSystem(::OrderedSet, ::AbstractSystem)

d(::Valuation)
⊗(::Valuation, ::Valuation)
↓(::Valuation, ::AbstractSet)
```
