"""
    Valuation

Abstract type for valuations.

Subtypes should support the following methods:
- [`d(ϕ::Valuation)`](@ref)
- [`↓(ϕ::Valuation, x::AbstractSet)`](@ref)
- [`⊗(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation end

"""
    LabeledBox{T₁, T₂} <: Valuation
"""
struct LabeledBox{T₁, T₂} <: Valuation
    box::T₁
    labels::OrderedSet{T₂}

    function LabeledBox(box::T₁, labels::OrderedSet{T₂}) where {T₁, T₂}
        new{T₁, T₂}(box, labels)
    end

    function LabeledBox(box::T₁, labels::OrderedSet{T₂}) where {T₁ <: AbstractSystem, T₂}
        @assert length(box) == length(labels)
        new{T₁, T₂}(box, labels)
    end
end

"""
    LabeledBox(box, labels::OrderedSet)
"""
LabeledBox(box, labels::OrderedSet)

"""
    d(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
d(ϕ::Valuation)

function d(ϕ::LabeledBox)
    Set(ϕ.labels)
end

"""
    ↓(ϕ::Valuation, x::AbstractSet)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
↓(ϕ::Valuation, x::AbstractSet)

function ↓(ϕ::LabeledBox, x::AbstractSet)
    ↓(oapply, ϕ, x)
end

function ↓(f, ϕ::LabeledBox, x::AbstractSet)
    @assert x ⊆ ϕ.labels
    labels = OrderedSet(x)
    indices = Dict(label => i 
                   for (i, label) in enumerate(ϕ.labels))
    composite = UntypedUWD(length(labels))
    add_box!(composite, length(ϕ.labels))
    add_junctions!(composite, length(ϕ.labels))
    for (i, label) in enumerate(ϕ.labels)
        set_junction!(composite, i, i; outer=false)
    end
    for (i, label) in enumerate(labels)
        set_junction!(composite, i, indices[label]; outer=true)
    end
    box = f(composite, [ϕ.box])
    LabeledBox(box, labels)
end

"""
    ⊗(ϕ₁::Valuation, ϕ₂::Valuation)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
⊗(ϕ₁::Valuation, ϕ₂::Valuation)

function ⊗(ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    ⊗(oapply, ϕ₁, ϕ₂)
end

function ⊗(f, ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    labels = ϕ₁.labels ∪ ϕ₂.labels
    indices = Dict(label => i 
                   for (i, label) in enumerate(labels))
    composite = UntypedUWD(length(labels))
    add_box!(composite, length(ϕ₁.labels))
    add_box!(composite, length(ϕ₂.labels))
    add_junctions!(composite, length(labels))
    for (i, label) in enumerate(flatten([ϕ₁.labels, ϕ₂.labels]))
        set_junction!(composite, i, indices[label]; outer=false)
    end
    for (i, label) in enumerate(labels)
        set_junction!(composite, i, i; outer=true)
    end
    box = f(composite, [ϕ₁.box, ϕ₂.box])
    LabeledBox(box, labels)
end

"""
    -(ϕ::Valuation, X)

Perform the variable elimination ``\\phi^{-X}``.
"""
function -(ϕ::Valuation, X)
    @assert X in d(ϕ)
    ϕ ↓ setdiff(d(ϕ), [X])
end

"""
    construct_inference_problem(composite::UndirectedWiringDiagram,
                                box_map::AbstractDict)
"""
function construct_inference_problem(composite::UndirectedWiringDiagram,
                                     box_map::AbstractDict)
    boxes = [box_map[x]
             for x in subpart(composite, :name)]
    construct_inference_problem(composite, boxes)
end

"""
    construct_inference_problem(composite::UndirectedWiringDiagram,
                                boxes::AbstractVector)
"""
function construct_inference_problem(composite::UndirectedWiringDiagram,
                                     boxes::AbstractVector)
    @assert nboxes(composite) == length(boxes)
    labels = [OrderedSet{Int}()
              for box in boxes]
    for i in ports(composite; outer=false)
        push!(labels[box(composite, i)], junction(composite, i; outer=false))
    end
    factors = Set(LabeledBox(box, label)
                  for (box, label) in zip(boxes, labels))
    query = Set(junction(composite, i; outer=true)
                for i in ports(composite; outer=true))
    factors, query
end

"""
    construct_elimination_sequence(domains::AbstractSet{T},
                                   query::AbstractSet) where T <: AbstractSet
"""
function construct_elimination_sequence(domains::AbstractSet{T},
                                        query::AbstractSet) where T <: AbstractSet
    E = domains; x = query
    Xs = setdiff(∪(E...), x)
    if isempty(Xs)
        return []
    else
        X = argmin(Xs) do X
            Eₓ = Set(s
                     for s in E
                     if X in s)
            length(∪(Eₓ...))
        end
        Eₓ = Set(s
                for s in E
                if X in s)
        sₓ = ∪(Eₓ...)
        F = setdiff(E, Eₓ) ∪ [setdiff(sₓ, [X])]
        return [X, construct_elimination_sequence(F, x)...]
    end
end

"""
    fusion_algorithm(factors::AbstractSet{T},
                     elimination_sequence) where T <: Valuation

An implementation of Shenoy's fusion algorithm (algorithm 3.1 in *Generic Inference*).
"""
function fusion_algorithm(factors::AbstractSet{T},
                          elimination_sequence) where T <: Valuation
    Ψ = factors
    for X in elimination_sequence
        Ψₓ = Set(ϕ
                 for ϕ in Ψ
                 if X in d(ϕ))
        ϕₓ = reduce(⊗, Ψₓ)
        Ψ = setdiff(Ψ, Ψₓ) ∪ [ϕₓ - X]
    end
    reduce(⊗, Ψ)
end
