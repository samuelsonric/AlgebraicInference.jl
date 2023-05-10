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
    labels::Vector{T₂}

    function LabeledBox(box::T₁, labels::Vector{T₂}) where {T₁, T₂}
        new{T₁, T₂}(box, labels)
    end

    function LabeledBox(box::T₁, labels::Vector{T₂}) where {T₁ <: AbstractSystem, T₂}
        @assert length(box) == length(labels)
        new{T₁, T₂}(box, labels)
    end
end

"""
    LabeledBox(box, labels)
"""
function LabeledBox(box, labels)
    LabeledBox(box, collect(labels))
end

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

"""
    ↓(F, ϕ::LabeledBox, x::AbstractSet)

Perform the projection ``\\phi^{\\downarrow x}`` using a **Cospan**-alebra ``F``.
"""
function ↓(F, ϕ::LabeledBox, x::AbstractSet)
    @assert x ⊆ d(ϕ)
    port_labels = ϕ.labels
    outer_port_labels = collect(x)
    junction_labels = collect(Set(port_labels))
    junction_indices = Dict(label => i
                            for (i, label) in enumerate(junction_labels))
    composite = UntypedUWD(length(outer_port_labels))
    add_box!(composite, length(ϕ.labels))
    add_junctions!(composite, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=true)
    end
    box = F(composite, [ϕ.box])
    LabeledBox(box, outer_port_labels)
end

"""
    ⊗(ϕ₁::Valuation, ϕ₂::Valuation)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
⊗(ϕ₁::Valuation, ϕ₂::Valuation)

function ⊗(ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    ⊗(oapply, ϕ₁, ϕ₂)
end

"""
    ⊗(F, ϕ₁::LabeledBox, ϕ₂::LabeledBox)

Perform the combination ``\\phi_1 \\otimes \\phi_2`` using a **Cospan**-algebra ``F``.
"""
function ⊗(F, ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    port_labels = [ϕ₁.labels; ϕ₂.labels]
    outer_port_labels = collect(Set(port_labels))
    junction_labels = outer_port_labels
    junction_indices = Dict(label => i
                            for (i, label) in enumerate(junction_labels))
    composite = UntypedUWD(length(outer_port_labels))
    add_box!(composite, length(ϕ₁.labels)); add_box!(composite, length(ϕ₂.labels))
    add_junctions!(composite, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=true)
    end
    box = F(composite, [ϕ₁.box, ϕ₂.box])
    LabeledBox(box, outer_port_labels)
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
    construct_inference_problem(F,
                                composite::UndirectedWiringDiagram,
                                box_map::AbstractDict)
"""
function construct_inference_problem(F,
                                     composite::UndirectedWiringDiagram,
                                     box_map::AbstractDict)
    boxes = [box_map[x]
             for x in subpart(composite, :name)]
    construct_inference_problem(F, composite, boxes)
end

construct_inference_problem(a, b) = construct_inference_problem(oapply, a, b)

"""
    construct_inference_problem(F,
                                composite::UndirectedWiringDiagram,
                                boxes::AbstractVector)

Let ``F`` be a **Cospan**-algebra, ``f`` an operation in **Cospan** of the form
```math
    B \\xleftarrow{\\mathtt{box}} R \\xrightarrow{\\mathtt{junc}} J
    \\xleftarrow{\\mathtt{junc'}} Q,
```
where ``\\mathtt{junc'}`` is injective, and ``(b_1, \\dots, b_n)`` a sequence of fillers for
the boxes in ``f``. Then `construct_inference_problem(composite, boxes)` constructs a
knowledge base ``\\{\\phi_1, \\dots, \\phi_n \\}`` and query ``x \\subseteq J`` such that
```math
    (\\phi_1 \\otimes \\dots \\otimes \\phi_n)^{\\downarrow x} \\cong F(f)(b_1, \\dots, b_n).
```
"""
function construct_inference_problem(F,
                                     composite::UndirectedWiringDiagram,
                                     boxes::AbstractVector{T}) where T
    @assert nboxes(composite) == length(boxes)
    query = Set(junction(composite, i; outer=true)
                for i in ports(composite; outer=true))
    neutral_element_labels = setdiff(query, Set(junction(composite, i; outer=false)
                                                for i in ports(composite; outer=false)))
    neutral_element = let composite
        outer_port_labels = collect(neutral_element_labels)
        junction_labels = outer_port_labels
        junction_indices = Dict(label => i
                                for (i, label) in enumerate(junction_labels))
        composite = UntypedUWD(length(outer_port_labels))
        add_junctions!(composite, length(junction_labels))
        for (i, label) in enumerate(outer_port_labels)
            set_junction!(composite, i, junction_indices[label]; outer=true)
        end
        box = F(composite, T[])
        LabeledBox(box, outer_port_labels)
    end
    labels = [Int[]
              for box in boxes]
    for i in ports(composite; outer=false)
        push!(labels[box(composite, i)], junction(composite, i; outer=false))
    end
    knowledge_base = Set(LabeledBox(box, label)
                         for (box, label) in zip(boxes, labels)) ∪ [neutral_element]
    knowledge_base, query
end

"""
    construct_elimination_sequence(edges::AbstractSet{T},
                                   query::AbstractSet) where T <: AbstractSet

Construct an elimination sequence using the "One Step Look Ahead - Smallest Clique"
heuristic.

Let ``(V, E)`` be a hypergraph and ``x \\subseteq V`` a query. Then
`construct_elimination_sequence(edges, query)` constructs an ordering ``(X_1, \\dots, X_m)``
of the vertices in ``V - x``.

References:
- Lehmann, N. 2001. *Argumentation System and Belief Functions*. Ph.D. thesis, Department
  of Informatics, University of Fribourg.
"""
function construct_elimination_sequence(edges::AbstractSet{T},
                                        query::AbstractSet) where T <: AbstractSet
    E = edges; x = query
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
    fusion_algorithm(knowledge_base::AbstractSet{T},
                     elimination_sequence) where T <: Valuation

An implementation of Shenoy's fusion algorithm.

Let ``\\{ \\phi_1, \\dots, \\phi_n \\}`` be a knowledge base and ``(X_1, \\dots, X_m)``
an elimination sequence. Then `fusion_algorithm(knowledge_base, elimination_sequence)`
computes the valuation ``\\phi^{\\downarrow x}``, where 
```math
    \\phi = \\phi_1 \\otimes \\dots \\otimes \\phi_n
```
and
```math
    x = d(\\phi) - \\{ X_i \\mid 1 \\leq i \\leq m \\}.
```

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
function fusion_algorithm(knowledge_base::AbstractSet{T},
                          elimination_sequence) where T <: Valuation
    Ψ = knowledge_base
    for X in elimination_sequence
        Ψₓ = Set(ϕ
                 for ϕ in Ψ
                 if X in d(ϕ))
        ϕₓ = reduce(⊗, Ψₓ)
        Ψ = setdiff(Ψ, Ψₓ) ∪ [ϕₓ - X]
    end
    reduce(⊗, Ψ)
end
