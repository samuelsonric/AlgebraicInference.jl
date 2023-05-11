# Library Reference

## Graphs

```@docs
construct_elimination_sequence(::AbstractSet{<:AbstractSet}, ::AbstractSet)
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
LabeledBox

LabeledBoxVariable{T}(::Any) where T
LabeledBox{T}(::Any, ::Vector) where T

domain(::Valuation)
combine(::Valuation{T}, ::Valuation{T}) where T
project(::Valuation{T}, ::AbstractSet{<:Variable{T}}) where T
neutral_element(::AbstractSet{<:Variable})
eliminate(::Valuation{T}, ::Variable{T}) where T

construct_inference_problem(::Type, ::UndirectedWiringDiagram, ::AbstractDict)
construct_inference_problem(::Type, ::UndirectedWiringDiagram, ::AbstractVector)
fusion_algorithm(::AbstractSet{<:Valuation{T}}, ::AbstractVector{<:Variable{T}}) where T
```
