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
one(::Type{<:Valuation})

inference_problem(::UndirectedWiringDiagram, ::Any)
inference_problem(::UndirectedWiringDiagram, ::AbstractDict)
```

## Join Trees
```@docs
JoinTree

JoinTree{T₁, T₂}(id, domain, factor) where {T₁, T₂ <: Valuation{T₁}}
JoinTree{T₁, T₂}(kb, order) where {T₁, T₂ <: Valuation{T₁}}
JoinTree(kb, order)

solve(::JoinTree, ::Any)
solve!(::JoinTree, ::Any)
```

## Graphs

```@docs
primal_graph
minfill!
minwidth!
```
