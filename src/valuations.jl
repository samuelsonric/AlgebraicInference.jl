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

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:Kernel})
    Var = LabeledBoxVariable{AbstractSystem}
    src_labels = OrderedSet{Var}(); tgt_labels = OrderedSet{Var}()
    n = size(ϕ₂.box.L, 2)
    for (i, X) in enumerate(ϕ₂.labels)
        if i <= n
            push!(src_labels, X)
        else
            push!(tgt_labels, X)
        end
    end
    if src_labels ⊆ ϕ₁.labels
        Γ₁ = ϕ₁.box.Γ; Γ₂ = ϕ₂.box.ϵ.Γ
        μ₁ = ϕ₁.box.μ; μ₂ = ϕ₂.box.ϵ.μ
        L = ϕ₂.box.L * [X == Y for X in src_labels, Y in ϕ₁.labels]
        Γ = [Γ₁     Γ₁ * L'
             L * Γ₁ L * Γ₁ * L' + Γ₂]
        μ = [μ₁
             L * μ₁ + μ₂]
        LabeledBox(ϕ₁.labels ∪ tgt_labels, ClassicalSystem(Γ, μ))
    else
        ϕ₂ = LabeledBox(ϕ₂.labels, System(ϕ₂.box))
        combine(ϕ₁, ϕ₂)    
    end
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:System})
    if ϕ₂.labels ⊆ ϕ₁.labels
        Γ₁ = ϕ₁.box.Γ; Γ₂ = ϕ₂.box.ϵ.Γ
        μ₁ = ϕ₁.box.μ; μ₂ = ϕ₂.box.ϵ.μ
        R = ϕ₂.box.R * [X == Y for X in ϕ₂.labels, Y in ϕ₁.labels]
        # K = (qr(R * Γ₁ * R' + Γ₂, Val(true)) \ (R * Γ₁))'
        K = Γ₁ * R' * pinv(R * Γ₁ * R' + Γ₂)
        Γ = (I - K * R) * Γ₁ * (I - K * R)' + K * Γ₂ * K'
        μ = (I - K * R) * μ₁ + K * μ₂
        LabeledBox(ϕ₁.labels, ClassicalSystem(Γ, μ))
    else
        ϕ₁ = LabeledBox(ϕ₁.labels, System(ϕ₁.box))
        combine(ϕ₁, ϕ₂)
    end
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:Union{System, Kernel}},
                 ϕ₂::LabeledBox{AbstractSystem, <:ClassicalSystem})
    combine(ϕ₂, ϕ₁)
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
    fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                     elimination_sequence::AbstractVector{T}) where T <: Variable


An implementation of the fusion algorithm.
"""
function fusion_algorithm(knowledge_base::Vector{<:Valuation{T}},
                          elimination_sequence::Vector{T}) where T <: Variable
    fused_factors = Vector{Valuation{T}}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in fused_factors]
        factor = eliminate(reduce(combine, fused_factors[mask]), X)
        keepat!(fused_factors, .!mask); push!(fused_factors, factor)
    end
    reduce(combine, fused_factors)
end
