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

function construct_join_tree_factors(knowledge_base::AbstractVector{<:Valuation{T₁}},
                                     assignment_map::AbstractVector{Int},
                                     join_tree_domains::AbstractVector{T₂},
                                     join_tree::Node{Int}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    join_tree_factors = Valuation{T₁}[neutral_element(x) for x in join_tree_domains]
    for (i, j) in enumerate(assignment_map)
        join_tree_factors[j] = combine(join_tree_factors[j], knowledge_base[i])
    end
    join_tree_factors
end  

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
    factors = Vector{Valuation{T}}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in factors]
        fused_factor = eliminate(reduce(combine, factors[mask]), X)
        keepat!(factors, .!mask); push!(factors, fused_factor)
    end
    reduce(combine, factors)
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
function collect_algorithm(join_tree_factors::AbstractVector{<:Valuation{T₁}},
                           join_tree::Node{Int},
                           query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    @assert query ⊆ domain(join_tree_factors[join_tree.id])
    factor = join_tree_factors[join_tree.id]
    for sub_tree in children(join_tree)
        message = collect_algorithm(join_tree_factors,
                                    sub_tree,
                                    domain(factor) ∩ domain(join_tree_factors[sub_tree.id])) 
        factor = combine(factor, message)
    end
    project(factor, query)
end 
