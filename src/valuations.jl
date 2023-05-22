"""
    Variable

Abtract type for variables in a valuation algebra. See [`Valuation`](@ref).
"""
abstract type Variable end

"""
    Valuation{T <: Variable}

Abstract type for valuations in a valuation algebra. For any type `T <: Variable`, the types
`Valuation{T}` and `Set{T}` should form a stable valuation algebra.

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

    @doc """
        LabeledBox(labels::OrderedSet{LabeledBoxVariable{T}}, box) where T
    """
    function LabeledBox(labels::OrderedSet{LabeledBoxVariable{T₁}}, box::T₂) where {T₁, T₂}
       new{T₁, T₂}(labels, box)
    end
end

"""
    LabeledBox(labels::Vector{LabeledBoxVariable{T}}, box) where T
"""
function LabeledBox(labels::Vector{LabeledBoxVariable{T}}, box) where T
    old_labels = labels; labels = OrderedSet(old_labels)
    if length(labels) < length(old_labels)
        port_labels = old_labels
        outer_port_labels = labels
        junction_labels = outer_port_labels
        junction_indices = Dict(label => i
                                for (i, label) in enumerate(junction_labels))
        composite = UntypedUWD(length(outer_port_labels))
        add_box!(composite, length(old_labels))
        add_junctions!(composite, length(junction_labels))
        for (i, label) in enumerate(port_labels)
            set_junction!(composite, i, junction_indices[label]; outer=false)
        end
        for (i, label) in enumerate(outer_port_labels)
            set_junction!(composite, i, junction_indices[label]; outer=true)
        end
        box = oapply(composite, [box])
    end
    LabeledBox(labels, box)
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
    l₁ = ϕ₁.labels; Σ₁ = ϕ₁.box
    l₂ = ϕ₂.labels; Σ₂ = ϕ₂.box
    lu = l₁ ∪ l₂
    LabeledBox(lu, [X == Y for X in [l₁..., l₂...], Y in lu] \ (Σ₁ ⊗ Σ₂))
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:System})
    l₁ = ϕ₁.labels; Σ₁ = ϕ₁.box
    l₂ = ϕ₂.labels; Σ₂ = ϕ₂.box
    if l₂ ⊆ l₁
        Γ₁ = Σ₁.Γ; Γ₂ = Σ₂.ϵ.Γ
        μ₁ = Σ₁.μ; μ₂ = Σ₂.ϵ.μ
        R₂ = Σ₂.R * [X == Y for X in l₂, Y in l₁]
        K = Γ₁ * R₂' * pinv(R₂ * Γ₁ * R₂' + Γ₂)
        Γ = (I - K * R₂) * Γ₁ * (I - K * R₂)' + K * Γ₂ * K'
        μ = (I - K * R₂) * μ₁ + K * μ₂
        LabeledBox(l₁, ClassicalSystem(Γ, μ))
    else
        ϕ₁ = LabeledBox(l₁, System(Σ₁))
        combine(ϕ₁, ϕ₂)
    end
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:ClassicalSystem})
    l₁ = ϕ₁.labels; Σ₁ = ϕ₁.box
    l₂ = ϕ₂.labels; Σ₂ = ϕ₂.box
    lu = l₁ ∪ l₂; li = l₁ ∩ l₂
    U = [
        1/√2 * [X == Y for X in l₁, Y in li]
       -1/√2 * [X == Y for X in l₂, Y in li]
    ]
    V = [
        1/2 * [(X == Y)((X ∉ l₂) + 1) for X in l₁, Y in lu]
        1/2 * [(X == Y)((X ∉ l₁) + 1) for X in l₂, Y in lu]
    ]
    Σ = Σ₁ ⊗ Σ₂; Γ = Σ.Γ
    LabeledBox(lu, V' * (I - Γ * U * pinv(U' * Γ * U) * U') * Σ)
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:Kernel})
    l₁ = ϕ₁.labels; Σ₁ = ϕ₁.box
    l₂ = ϕ₂.labels; Σ₂ = ϕ₂.box
    n₂ = dof(Σ₂); s₂ = OrderedSet(take(l₂, n₂)); t₂ = OrderedSet(drop(l₂, n₂))
    if s₂ ⊆ l₁
        lu = l₁ ∪ t₂; li = l₁ ∩ t₂
        Γ₁ = Σ₁.Γ; Γ₂ = Σ₂.ϵ.Γ
        μ₁ = Σ₁.μ; μ₂ = Σ₂.ϵ.μ
        L₂ = Σ₂.L * [X == Y for X in s₂, Y in l₁]
        Γ = [
            Γ₁      Γ₁ * L₂'
            L₂ * Γ₁ L₂ * Γ₁ * L₂' + Γ₂
        ]
        μ = [
            μ₁
            L₂ * μ₁ + μ₂
        ]
        U = [
            1/√2 * [X == Y for X in l₁, Y in li]
           -1/√2 * [X == Y for X in t₂, Y in li]
        ]
        V = [
            1/2 * [(X == Y) * ((X ∉ t₂) + 1) for X in l₁, Y in lu]
            1/2 * [(X == Y) * ((X ∉ l₁) + 1) for X in t₂, Y in lu]
        ]
        Σ = ClassicalSystem(Γ, μ)
        LabeledBox(lu, V' * (I - Γ * U * pinv(U' * Γ * U) * U') * Σ)
    else
        ϕ₁ = LabeledBox(l₁, Kernel(Σ₁))
        combine(ϕ₁, ϕ₂)
    end  
end      

function combine(ϕ₁::LabeledBox{AbstractSystem, <:Kernel},
                 ϕ₂::LabeledBox{AbstractSystem, <:Kernel})
    l₁ = ϕ₁.labels; Σ₁ = ϕ₁.box
    l₂ = ϕ₂.labels; Σ₂ = ϕ₂.box
    n₁ = dof(Σ₁); s₁ = OrderedSet(take(l₁, n₁)); t₁ = OrderedSet(drop(l₁, n₁))
    n₂ = dof(Σ₂); s₂ = OrderedSet(take(l₂, n₂)); t₂ = OrderedSet(drop(l₂, n₂))
    if isdisjoint(l₁, t₂)
        sd = setdiff(s₂, t₁); su = s₁ ∪ sd
        Γ₁ = Σ₁.ϵ.Γ; Γ₂ = Σ₂.ϵ.Γ
        μ₁ = Σ₁.ϵ.μ; μ₂ = Σ₂.ϵ.μ
        L₁ = Σ₁.L
        L₂₁ = Σ₂.L * [X == Y for X in s₂, Y in t₁]
        L₂₂ = Σ₂.L * [X == Y for X in s₂, Y in sd]
        Γ = [
            Γ₁       Γ₁ * L₂₁'
            L₂₁ * Γ₁ L₂₁ * Γ₁ * L₂₁' + Γ₂
        ]
        μ = [
            μ₁
            L₂₁ * μ₁ + μ₂
        ]
        L = [
            L₁       zeros(length(t₁), length(sd))
            L₂₁ * L₁ L₂₂
        ] * [X == Y for X in [s₁..., sd...], Y in su]
        LabeledBox(lu ∪ t₁ ∪ t₂, Kernel(L, ClassicalSystem(Γ, μ)))
    else
        ϕ₁ = LabeledBox(l₁, System(Σ₁))
        ϕ₂ = LabeledBox(l₂, System(Σ₂))
        combine(ϕ₁, ϕ₂)
    end
end

function combine(ϕ₁::LabeledBox{AbstractSystem},
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
    l₁ = ϕ.labels; Σ₁ = ϕ.box
    l₂ = OrderedSet(x)
    LabeledBox(l₂, [X == Y for X in l₂, Y in l₁] * Σ₁)
end

function project(ϕ::LabeledBox{AbstractSystem, <:System},
                 x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    l₁ = ϕ.labels; Σ₁ = ϕ.box
    l₂ = OrderedSet(x)
    R₁ = Σ₁.R; ϵ₁ = Σ₁.ϵ
    U = [X == Y for X in l₂, Y in l₁]
    V = nullspace((I - U' * U) * R₁')'
    LabeledBox(l₂, System(V * R₁ * U', V * ϵ₁))
end

function project(ϕ::LabeledBox{AbstractSystem, <:Kernel},
                 x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    l₁ = ϕ.labels; Σ₁ = ϕ.box 
    l₂ = OrderedSet(x)
    n₁ = dof(Σ₁); s₁ = OrderedSet(take(l₁, n₁)); t₁ = OrderedSet(drop(l₁, n₁))
    if s₁ ⊆ l₂
        ti = t₁ ∩ l₂
        L₁ = Σ₁.L; ϵ₁ = Σ₁.ϵ
        U = [X == Y for X in ti, Y in t₂]
        LabeledBox(s₁ ∪ ti, Kernel(U * L₁, U * ϵ₁))
    else
        ϕ = LabeledBox(l₁, System(Σ₁))
        project(ϕ, x)
    end
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
    knowledge_base_labels = [Var[] for box in boxes]
    for i in ports(composite; outer=false)
        labels = knowledge_base_labels[box(composite, i)]
        push!(labels, Var(junction(composite, i; outer=false)))
    end
    knowledge_base = Valuation{Var}[]
    for (labels, box) in zip(knowledge_base_labels, boxes)
        push!(knowledge_base, LabeledBox(labels, box))
    end
    query = Set(Var(junction(composite, i; outer=true))
                for i in ports(composite; outer=true))
    variables = Set(Var(junction(composite, i; outer=false))
                    for i in ports(composite; outer=false))
    difference = setdiff(query, variables)
    if !isempty(difference)
        push!(knowledge_base, neutral_valuation(difference))
    end
    knowledge_base, query
end

"""
    fusion_algorithm(knowledge_base::AbstractVector{<:Valuation{T}},
                     elimination_sequence::AbstractVector{T}) where T

Eliminate variables from a knowledge base using the fusion algorithm.

Let ``\\{\\phi_1, \\dots, \\phi_n\\}`` be a knowledge base and  ``s = (X_1, \\dots, X_m)``
an elimination sequence. Then `fusion_algorithm(knowledge_base, elimination_sequence)` solves
the inference problem
```math
(\\phi_1 \\dots \\otimes \\dots \\phi_n)^{-s}.
```
"""
function fusion_algorithm(knowledge_base::Vector{<:Valuation{T}},
                          elimination_sequence::Vector{T}) where T
    fused_factors = Vector{Valuation{T}}(knowledge_base)
    for X in elimination_sequence
        mask = [X in domain(ϕ) for ϕ in fused_factors]
        factor = eliminate(reduce(combine, fused_factors[mask]), X)
        keepat!(fused_factors, .!mask); push!(fused_factors, factor)
    end
    reduce(combine, fused_factors)
end
