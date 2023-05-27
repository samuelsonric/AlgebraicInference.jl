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
    labels::Vector{Int}
    box::T₂

    @doc """
        LabeledBox(labels::OrderedSet{LabeledBoxVariable{T}}, box) where T
    """
    function LabeledBox{T₁}(labels::Vector{Int}, box::T₂) where {T₁, T₂}
       new{T₁, T₂}(labels, box)
    end
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
    V = LabeledBoxVariable{T}
    Set{V}(V(X) for X in ϕ.labels)
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
    junction_indices = Dict(
        label => i
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

function combine(ϕ₁::LabeledBox{AbstractSystem, <:OpenProgram},
                 ϕ₂::LabeledBox{AbstractSystem, <:OpenProgram})
    Σ₁ = ϕ₁.box; Γ₁ = Σ₁.ϵ.Γ; μ₁ = Σ₁.ϵ.μ; L₁ = Σ₁.L; o₁ = Σ₁.o
    Σ₂ = ϕ₂.box; Γ₂ = Σ₂.ϵ.Γ; μ₂ = Σ₂.ϵ.μ; L₂ = Σ₂.L; o₂ = Σ₂.o
    n₁ = size(L₁, 2); l₁ = ϕ₁.labels; s₁ = l₁[1:n₁]; t₁ = l₁[n₁+1:end]
    n₂ = size(L₂, 2); l₂ = ϕ₂.labels; s₂ = l₂[1:n₂]; t₂ = l₂[n₂+1:end]
    ds₁ = Set(s₁); dt₁ = Set(t₁)
    ds₂ = Set(s₂); dt₂ = Set(t₂)
    if isdisjoint(ds₁, dt₂)
        s = collect(ds₁ ∪ setdiff(ds₂, dt₁))
        t = collect(dt₁ ∪ dt₂)
        _t₁ = [t₁; -o₁:-1]
        _t₂ = [t₂; -o₂-o₁:-1-o₁]
        _t  = [t;  -o₂-o₁:-1; collect(dt₁ ∩ dt₂)]
        m = length(t); _m = length(_t)
        U  = [X == Y for X in s₁, Y in s]
        V₁ = [X == Y for X in s₂, Y in _t₁]
        V₂ = [X == Y for X in s₂, Y in s]
        W₁ = [X == Y for X in _t, Y in _t₁]
        W₂ = [
            !(X in dt₁ && X in dt₂) ? X == Y : i > m ? -(X == Y) : 0
            for (i, X) in enumerate(_t), Y in _t₂]
        K = W₁ + W₂ * L₂ * V₁
        l = [s; t]
        Γ = K * Γ₁ * K' + W₂ * Γ₂ * W₂'
        μ = K * μ₁ + W₂ * μ₂
        L = K * L₁ * U + W₂ * L₂ * V₂
        o = _m - m
        LabeledBox{AbstractSystem}(l, OpenProgram(ClassicalSystem(Γ,μ), L, o))
    elseif isdisjoint(ds₂, dt₁)
        combine(ϕ₂, ϕ₁)
    else
        error()
    end
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:OpenProgram})

    ϕ₁ = LabeledBox{AbstractSystem}(ϕ₁.labels, OpenProgram(ϕ₁.box))
    ϕ = combine(ϕ₁, ϕ₂)
    Σ = ϕ.box; Γ = Σ.ϵ.Γ; μ = Σ.ϵ.μ; L = Σ.L
    if size(L, 2) == 0
        n = length(ϕ.labels)
        Γ₁₁ = Γ[1:n, 1:n]
        Γ₁₂ = Γ[1:n, n+1:end]
        Γ₂₂ = Γ[n+1:end, n+1:end]
        μ₁ = μ[1:n]
        μ₂ = μ[n+1:end]
        _K = Γ₁₂ * pinv(Γ₂₂)
        _Γ = Γ₁₁ - _K * Γ₁₂'
        _μ = μ₁  - _K * μ₂
        _ϕ = LabeledBox{AbstractSystem}(ϕ.labels, ClassicalSystem(_Γ, _μ))
    else
        _ϕ = ϕ
    end
    _ϕ 
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:OpenProgram},
                 ϕ₂::LabeledBox{AbstractSystem, <:ClassicalSystem})
    combine(ϕ₂, ϕ₁)
end

function combine(ϕ₁::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 ϕ₂::LabeledBox{AbstractSystem, <:ClassicalSystem})
    ϕ₂ = LabeledBox{AbstractSystem}(ϕ₂.labels, OpenProgram(ϕ₂.box))
    combine(ϕ₁, ϕ₂)
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
    junction_indices = Dict(
        label => i
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

function project(ϕ::LabeledBox{AbstractSystem, <:ClassicalSystem},
                 x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    Σ = ϕ.box; Γ = Σ.Γ; μ = Σ.μ
    l = ϕ.labels
    dx = Set(X.id for X in x)
    mask = [X in dx for X in l]
    _Γ = Γ[mask, mask]
    _μ = μ[mask]
    _l = l[mask]
    LabeledBox{AbstractSystem}(_l, ClassicalSystem(_Γ, _μ))
end

function project(ϕ::LabeledBox{AbstractSystem, <:OpenProgram},
                 x::AbstractSet{LabeledBoxVariable{AbstractSystem}})
    Σ = ϕ.box; Γ = Σ.ϵ.Γ; μ = Σ.ϵ.μ; L = Σ.L; o = Σ.o
    n = size(L, 2); l = ϕ.labels; s = l[1:n]; t = l[n+1:end]
    ds = Set(s); dx = Set(X.id for X in x)
    if ds ⊆ dx 
         mask = [X in dx for X in t]
        _mask = [mask; trues(o)]
        _l = [s; t[mask]]
        _Γ = Γ[_mask, _mask]
        _μ = μ[_mask]
        _L = L[_mask, :]
        _o = o
        LabeledBox{AbstractSystem}(_l, OpenProgram(ClassicalSystem(_Γ, _μ), _L, _o))
    else
        error()
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
    junction_indices = Dict(
        label => i
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
    @assert (
        length(ports(composite; outer=true))
        == length(Set(
            junction(composite, i; outer=true)
            for i in ports(composite; outer=true))))
    knowledge_base_labels = [Int[] for box in boxes]
    for i in ports(composite; outer=false)
        labels = knowledge_base_labels[box(composite, i)]
        push!(labels, junction(composite, i; outer=false))
    end
    knowledge_base = LabeledBox{T}[]
    for (labels, box) in zip(knowledge_base_labels, boxes)
        push!(knowledge_base, LabeledBox{T}(labels, box))
    end
    query = Set(
        LabeledBoxVariable{T}(junction(composite, i; outer=true))
        for i in ports(composite; outer=true))
    variables = Set(
        LabeledBoxVariable{T}(junction(composite, i; outer=false))
        for i in ports(composite; outer=false))
    difference = setdiff(query, variables)
    if !isempty(difference)
        push!(knowledge_base, neutral_valuation(difference))
    end
    knowledge_base, query
end
