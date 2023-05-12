"""
    Variable

Abtract type for variables in a valuation algebra. See [`Valuation`](@ref).
"""
abstract type Variable end

"""
    Valuation{T <: Variable}

Abstract type for valuations in a valuation algebra. For any type `T <: Variable`, the types
`Valuation{T}` and `T` should form a stable valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation{T} where T <: Variable)`](@ref)
- [`combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T <: Variable`](@ref)
- [`project(ϕ::Valuation{T}, x::AbstractSet{T}) where T <: Variable`](@ref)
- [`neutral_element(x::AbstractSet{T}) where T <: Variable`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation{T} end

struct LabeledBoxVariable{T} <: Variable
    id::Int

    @doc """
        LabeledBoxVariable{T}(id::Int) where T
    """
    function LabeledBoxVariable{T}(id::Int) where T
        new{T}(id)
    end
end

"""
    LabeledBox{T₁, T₂} <: Valuation{LabeledBoxVariable{T₁}}
"""
struct LabeledBox{T₁, T₂} <: Valuation{LabeledBoxVariable{T₁}}
    labels::Vector{LabeledBoxVariable{T₁}}
    box::T₂

    @doc """
        LabeledBox(labels::Vector{LabeledBoxVariable{T}}, box) where T
    """
    function LabeledBox(labels::Vector{LabeledBoxVariable{T₁}}, box::T₂) where {T₁, T₂}
        new{T₁, T₂}(labels, box)
    end
end

"""
    domain(ϕ::Valuation{T}) where T <: Variable

Get the domain of ``\\phi``.
"""
domain(ϕ::Valuation{T}) where T <: Variable

function domain(ϕ::LabeledBox{T}) where T
    Set(ϕ.labels)
end

"""
    combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T <: Variable

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T <: Variable

function combine(ϕ₁::LabeledBox{T}, ϕ₂::LabeledBox{T}) where T
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
    LabeledBox(outer_port_labels, box)
end

"""
    project(ϕ::Valuation{T}, x::AbstractSet{T}) where T <: Variable

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation{T}, x::AbstractSet{T}) where T <: Variable

function project(ϕ::LabeledBox{T}, x::AbstractSet{LabeledBoxVariable{T}}) where T
    @assert x ⊆ domain(ϕ)
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
    box = oapply(composite, [ϕ.box])
    LabeledBox(outer_port_labels, box)
end

"""
    neutral_element(x::AbstractSet{T}) where T <: Variable

Construct the neutral element ``\\phi^{\\downarrow x}``.
"""
neutral_element(x::AbstractSet{T}) where T <: Variable

function neutral_element(x::AbstractSet{LabeledBoxVariable{T}}) where T  
    outer_port_labels = collect(x)
    junction_labels = outer_port_labels
    junction_indices = Dict(label => i
                            for (i, label) in enumerate(junction_labels))
    composite = UntypedUWD(length(outer_port_labels))
    add_junctions!(composite, length(junction_labels))
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(composite, i, junction_indices[label]; outer=true)
    end
    box = oapply(composite, T[])
    LabeledBox(outer_port_labels, box)
end

"""
    eliminate(ϕ::Valuation{T}, X::T) where T <: Variable

Perform the variable elimination ``\\phi^{-X}``.
"""
function eliminate(ϕ::Valuation{T}, X::T) where T <: Variable
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
    Var = LabeledBoxVariable{T}
    labels = [Var[] for box in boxes]
    for i in ports(composite; outer=false)
        push!(labels[box(composite, i)], Var(junction(composite, i; outer=false)))
    end
    knowledge_base = [LabeledBox(label, box) for (label, box) in zip(labels, boxes)]
    query = Set(Var(junction(composite, i; outer=true))
                for i in ports(composite; outer=true))
    variables = Set(Var(junction(composite, i; outer=false))
                    for i in ports(composite; outer=false))
    e = neutral_element(setdiff(query, variables))
    [knowledge_base..., e], query
end

#=
function construct_join_tree_factors(knowledge_base::AbstractVector{<:Valuation{T}},
                                     join_tree::LabeledGraph{<:Variable{T}},
                                     assignment_map::AbstractVector{<:Integer}) where T
    Ψ = knowledge_base; λ = join_tree.labels; a = assignment_map
    join_tree_factors = Valuation{T}[neutral_element(x) for x in λ]
    for (i, j) in enumerate(a)
        join_tree_factors[j] = combine(join_tree_factors[j], Ψ[i])
    end
    join_tree_factors
end  
=#

"""
    fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                     elimination_sequence::AbstractVector{T}) where T <: Variable

Perform the fusion algorithm.

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
function fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                          elimination_sequence::AbstractVector{T}) where T <: Variable
    Ψ = Vector{Valuation{T}}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in Ψ]
        ϕ = eliminate(reduce(combine, Ψ[mask]), X)
        keepat!(Ψ, .!mask); push!(Ψ, ϕ)
    end
    reduce(combine, Ψ)
end

#=
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
function collect_algorithm(join_tree_factors::AbstractVector{<:Valuation{T}},
                           join_tree::LabeledGraph{<:Variable{T}},
                           query::AbstractSet{<:Variable{T}}) where T
    Ψ = join_tree_factors; λ = join_tree.labels; E = join_tree.edges; x = query
    V = length(λ)
    for i in 1:V - 1
        j = child(join_tree, i)
        Ψ[j] = combine(Ψ[j], project(Ψ[i], domain(Ψ[i]) ∩ λ[j]))
    end
    project(Ψ[V], x)
end
=#
