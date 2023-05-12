"""
    Variable{T}

Abtract type for variables in a valuation algebra. See [`Valuation`](@ref).
"""
abstract type Variable{T} end

"""
    Valuation{T}

Abstract type for valuations in a valuation algebra. For any type `T`, the types
`Valuation{T}` and `Variable{T}` should form a stable valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T`](@ref)
- [`project(ϕ::Valuation{T}, x::AbstractSet{<:Variable{T}}) where T`](@ref)
- [`neutral_element(x::AbstractSet{<:Variable})`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation{T} end

"""
    LabeledBoxVariable{T} <: Variable{T}
"""
struct LabeledBoxVariable{T} <: Variable{T}
    value::Int

    @doc """
        LabeledBoxVariable{T}(value::Int) where T
    """
    function LabeledBoxVariable{T}(value::Int) where T
        new{T}(value)
    end
end

"""
    LabeledBox{T₁, T₂} <: Valuation{T₁}
"""
struct LabeledBox{T₁, T₂} <: Valuation{T₁}
    box::T₂
    labels::Vector{Int}

    @doc """
        LabeledBox{T}(box, labels::Vector{Int}) where T
    """
    function LabeledBox{T₁}(box::T₂, labels::Vector{Int}) where {T₁, T₂}
        new{T₁, T₂}(box, labels)
    end
end

"""
    domain(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
domain(ϕ::Valuation)

function domain(ϕ::LabeledBox{T}) where T
    Var = LabeledBoxVariable{T}
    Set{Var}(Var(X) for X in ϕ.labels)
end

"""
    combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

function combine(ϕ₁::LabeledBox{T}, ϕ₂::LabeledBox{T}) where T
    Val = LabeledBox{T}; Var = LabeledBoxVariable{T}
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
    box = oapply(composite, [ϕ₁.box, ϕ₂.box])
    Val(box, outer_port_labels)
end

"""
    project(ϕ::Valuation{T}, x::AbstractSet{<:Variable{T}}) where T

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation{T}, x::AbstractSet{<:Variable{T}}) where T

function project(ϕ::LabeledBox{T}, x::AbstractSet{<:LabeledBoxVariable{T}}) where T
    @assert x ⊆ domain(ϕ)
    Val = LabeledBox{T}; Var = LabeledBoxVariable{T}
    port_labels = ϕ.labels
    outer_port_labels = [X.value for X in x]
    junction_labels = [label for label in Set(port_labels)]
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
    box = oapply(composite, [ϕ.box])
    Val(box, outer_port_labels)
end

"""
    neutral_element(x::AbstractSet{<:Variable})

Construct the neutral element ``\\phi^{\\downarrow x}``.
"""
neutral_element(x::AbstractSet{<:Variable})

function neutral_element(x::AbstractSet{<:LabeledBoxVariable{T}}) where T  
    Val = LabeledBox{T}; Var = LabeledBoxVariable{T}
    outer_port_labels = [X.value for X in x]
    junction_labels = outer_port_labels
    junction_indices = Dict(label => i
                            for (i, label) in enumerate(junction_labels))
    composite = UntypedUWD(length(outer_port_labels))
    add_junctions!(composite, length(junction_labels))
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=true)
    end
    box = oapply(composite, T[])
    Val(box, outer_port_labels)
end

"""
    eliminate(ϕ::Valuation{T}, X::Variable{T}) where T

Perform the variable elimination ``\\phi^{-X}``.
"""
function eliminate(ϕ::Valuation{T}, X::Variable{T}) where T
    @assert X in domain(ϕ)
    x = setdiff(domain(ϕ), [X])
    project(ϕ, x)
end

"""
    construct_inference_problem(::Type,
                                composite::UndirectedWiringDiagram,
                                box_map::AbstractDict) where T
"""
function construct_inference_problem(::Type{T},
                                     composite::UndirectedWiringDiagram,
                                     box_map::AbstractDict) where T
    boxes = [box_map[x] for x in subpart(composite, :name)]
    construct_inference_problem(T, composite, boxes)
end

"""
    construct_inference_problem(::Type,
                                composite::UndirectedWiringDiagram,
                                boxes::AbstractVector)

Let ``f`` be an operation in **Cospan** of the form
```math
    B \\xleftarrow{\\mathtt{box}} P \\xrightarrow{\\mathtt{junc}} J
    \\xleftarrow{\\mathtt{junc'}} Q,
```
where ``\\mathtt{junc'}: Q \\to J`` is injective, and ``(b_1, \\dots, b_n)`` a sequence of
fillers for the boxes in ``f``. Then `construct_inference_problem(T, composite, boxes)`
constructs a knowledge base ``\\{\\phi_1, \\dots, \\phi_n\\}`` and query ``x \\subseteq J``
such that
```math
    (\\phi_1 \\otimes \\dots \\otimes \\phi_n)^{\\downarrow x} \\cong
    F(f)(b_1, \\dots, b_n),
```
where ``F`` is the **Cospan**-algebra computed by `oapply`.
"""
function construct_inference_problem(::Type{T},
                                     composite::UndirectedWiringDiagram,
                                     boxes::AbstractVector) where T
    @assert nboxes(composite) == length(boxes)
    @assert length(ports(composite; outer=true)) ==
            length(Set(junction(composite, i; outer=true)
                       for i in ports(composite; outer=true)))
    Val = LabeledBox{T}; Var = LabeledBoxVariable{T}
    labels = [Int[] for box in boxes]
    for i in ports(composite; outer=false)
        push!(labels[box(composite, i)], junction(composite, i; outer=false))
    end
    knowledge_base = [Val(box, label) for (box, label) in zip(boxes, labels)]
    query = Set(Var(junction(composite, i; outer=true))
                for i in ports(composite; outer=true))
    variables = Set(Var(junction(composite, i; outer=false))
                    for i in ports(composite; outer=false))
    e = neutral_element(setdiff(query, variables))
    [knowledge_base..., e], query
end

"""
    fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                     elimination_sequence::AbstractVector{<:Variable{T}}) where T

Perform the fusion algorithm.

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
function fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                          elimination_sequence::AbstractVector{<:Variable{T}}) where T
    Val = Valuation{T}
    Ψ = Vector{Val}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in Ψ]
        ϕ = eliminate(reduce(combine, Ψ[mask]), X)
        keepat!(Ψ, .!mask); push!(Ψ, ϕ)
    end
    reduce(combine, Ψ)
end


"""
    collect_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                      assignment_map::AbstractVector{<:Integer},
                      labels::AbstractVector{<:AbstractSet{<:Variable{T}}},
                      edges::AbstractSet{<:AbstractSet{<:Integer}},
                      query::AbstractSet{<:Variable{T}}) where T

Perform the collect algorithm.

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
function collect_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                           assignment_map::AbstractVector{<:Integer},
                           labels::AbstractVector{<:AbstractSet{<:Variable{T}}},
                           edges::AbstractSet{<:AbstractSet{<:Integer}},
                           query::AbstractSet{<:Variable{T}}) where T
    Val = Valuation{T}
    Ψ = knowledge_base; a = assignment_map; λ = labels; E = edges; x = query
    V = length(λ)
    Ψ = Val[neutral_element(label) for label in labels]
    for (i, j) in enumerate(assignment_map)
        Ψ[j] = combine(Ψ[j], knowledge_base[i])
    end
    for i in 1:V - 1
        j = ch(V, E, i)
        Ψ[j] = combine(Ψ[j], project(Ψ[i], domain(Ψ[i]) ∩ λ[j]))
    end
    project(Ψ[V], x)
end
