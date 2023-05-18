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
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T`](@ref)
- [`project(ϕ::Valuation{T}, x::AbstractSet{T}) where T`](@ref)
- [`neutral_valuation(x::AbstractSet{T}) where T <: Variable`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*;
  Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation{T} end

"""
    IdentityValuation{T} <: Valuation{T}

The identity element ``e``.
"""
struct IdentityValuation{T} <: Valuation{T} end

"""
    LabeledBoxVariable{T} <: Variable
"""
struct LabeledBoxVariable{T} <: Variable
    id::Int
end

"""
    LabeledBox{T₁, T₂} <: Valuation{LabeledBoxVariable{T₁}}
"""
struct LabeledBox{T₁, T₂} <: Valuation{LabeledBoxVariable{T₁}}
    labels::OrderedSet{LabeledBoxVariable{T₁}}
    box::T₂
end

"""
    domain(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
domain(ϕ::Valuation)

function domain(ϕ::IdentityValuation{T}) where T
    Set{T}()
end

function domain(ϕ::LabeledBox{T}) where T
    Set(ϕ.labels)
end

"""
    combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

function combine(ϕ₁::IdentityValuation{T}, ϕ₂::Valuation{T}) where T
    ϕ₂
end

function combine(ϕ₁::Valuation{T}, ϕ₂::IdentityValuation{T}) where T
    ϕ₁
end

function combine(ϕ₁::IdentityValuation{T}, ϕ₂::IdentityValuation{T}) where T
    ϕ₁
end

function combine(ϕ₁::LabeledBox{T}, ϕ₂::LabeledBox{T}) where T
    port_labels = [ϕ₁.labels..., ϕ₂.labels...]
    outer_port_labels = ϕ₁.labels ∪ ϕ₂.labels
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

function combine(ϕ₁::LabeledBox{AbstractSystem}, ϕ₂::LabeledBox{AbstractSystem})
    labels = ϕ₁.labels ∪ ϕ₂.labels
    box = [i == j for i in [ϕ₁.labels..., ϕ₂.labels...], j in labels] \ (ϕ₁.box ⊗ ϕ₂.box)
    LabeledBox(labels, box)
end

"""
    project(ϕ::Valuation{T}, x::AbstractSet{T}) where T

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation{T}, x::AbstractSet{T}) where T

function project(ϕ::IdentityValuation{T}, x::AbstractSet{T}) where T
    @assert isempty(x)
    ϕ
end

function project(ϕ::LabeledBox{T}, x::AbstractSet{LabeledBoxVariable{T}}) where T
    @assert x ⊆ ϕ.labels
    port_labels = ϕ.labels
    outer_port_labels = OrderedSet(x)
    junction_labels = port_labels
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

function project(ϕ::LabeledBox{AbstractSystem}, x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    labels = OrderedSet(x)
    box = [i == j for i in labels, j in ϕ.labels] * ϕ.box
    LabeledBox(labels, box)
end

function project(ϕ::LabeledBox{AbstractSystem, <:System}, x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    labels = OrderedSet(x)
    U = [i == j for i in labels, j in ϕ.labels]
    V = nullspace((I - U' * U) * ϕ.box.R')'
    box = System(V * ϕ.box.R * U', V * ϕ.box.ϵ)
    LabeledBox(labels, box)
end

"""
    neutral_valuation(x::AbstractSet{T}) where T <: Variable

Construct the neutral element ``\\phi^{\\downarrow x}``.
"""
neutral_valuation(x::AbstractSet{T}) where T <: Variable

function neutral_valuation(x::AbstractSet{LabeledBoxVariable{T}}) where T  
    outer_port_labels = OrderedSet(x)
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

#FIXME
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
    labels = [OrderedSet{Var}() for box in boxes]
    for i in ports(composite; outer=false)
        push!(labels[box(composite, i)], Var(junction(composite, i; outer=false)))
    end
    knowledge_base = [LabeledBox(label, box) for (label, box) in zip(labels, boxes)]
    query = Set(Var(junction(composite, i; outer=true))
                for i in ports(composite; outer=true))
    variables = Set(Var(junction(composite, i; outer=false))
                    for i in ports(composite; outer=false))
    e = neutral_valuation(setdiff(query, variables))
    [knowledge_base..., e], query
end

"""
    construct_factors(knowledge_base::AbstractVector{<:Valuation{T₁}},
                      assignment_map::AbstractVector{Int},
                      domains::AbstractVector{T₂},
                      tree::Node{Int};
                      identity=false) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
"""
function construct_factors(knowledge_base::AbstractVector{<:Valuation{T₁}},
                           assignment_map::AbstractVector{Int},
                           domains::AbstractVector{T₂},
                           tree::Node{Int};
                           identity=false) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    id = IdentityValuation{T₁}()
    factors = Valuation{T₁}[]
    for x in domains
        e = identity ? id : neutral_valuation(x)
        push!(factors, e)
    end
    for (i, j) in enumerate(assignment_map)
        factors[j] = combine(factors[j], knowledge_base[i])
    end
    factors
end  

"""
    fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                     elimination_sequence::AbstractVector{T}) where T <: Variable


An implementation of the fusion algorithm.
"""
function fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                          elimination_sequence::AbstractVector{T}) where T <: Variable
    fused_factors = Vector{Valuation{T}}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in fused_factors]
        factor = eliminate(reduce(combine, fused_factors[mask]), X)
        keepat!(fused_factors, .!mask); push!(fused_factors, factor)
    end
    reduce(combine, fused_factors)
end

"""
    collect_algorithm(factors::AbstractVector{<:Valuation{T₁}},
                      domains::AbstractVector{T₂},
                      tree::Node{Int},
                      query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}

An implementation of the collect algorithm.
"""
function collect_algorithm(factors::AbstractVector{<:Valuation{T₁}},
                           domains::AbstractVector{T₂},
                           tree::Node{Int},
                           query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    @assert query ⊆ domains[tree.id]
    factor = factors[tree.id]
    for subtree in tree.children
        message = message_to_parent(factors, domains, subtree)
        factor = combine(factor, message)
    end
    project(factor, query)
end

"""
    shenoy_shafer_architecture!(mailboxes::AbstractDict{Tuple{Int, Int}, Valuation{T₁}},
                                factors::AbstractVector{<:Valuation{T₁}},
                                domains::AbstractVector{T₂},
                                tree::Node{Int},
                                query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}

An implementation of the Shenoy-Shafer architecture.
"""
function shenoy_shafer_architecture!(mailboxes::AbstractDict{Tuple{Int, Int}, Valuation{T₁}},
                                     factors::AbstractVector{<:Valuation{T₁}},
                                     domains::AbstractVector{T₂},
                                     tree::Node{Int},
                                     query::AbstractSet{T₁}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    for tree in PreOrderDFS(tree)
        if query ⊆ domains[tree.id]        
            factor = factors[tree.id]
            for subtree in tree.children
                message = message_to_parent!(mailboxes, factors, domains, subtree)
                factor = combine(factor, message)
            end
            if isdefined(tree, :parent)
                message = message_from_parent!(mailboxes, factors, domains, tree)
                factor = combine(factor, message)
            end
            return project(factor, query)
        end 
    end
    error()
end
