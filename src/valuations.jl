"""
    Valuation{T}

Abstract type for valuations in a valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)
- [`project(ϕ::Valuation, x)`](@ref)

Valuations are parametrized by the type of the variables in their variable system. If
`ϕ <: Valuation{T}`, then `domain(ϕ)` should return a container with element type `T`.

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
    LabeledBox{T₁, T₂} <: Valuation{T₁}
"""
struct LabeledBox{T₁, T₂} <: Valuation{T₁}
    labels::Vector{T₁}
    box::T₂

    @doc """
        LabeledBox(labels::Vector, box)
    """
    function LabeledBox(labels::Vector{T₁}, box::T₂) where {T₁, T₂}
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

function domain(ϕ::LabeledBox)
    ϕ.labels
end

"""
    combine(ϕ₁::Valuation, ϕ₂::Valuation)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
combine(ϕ₁::Valuation, ϕ₂::Valuation)

function combine(ϕ₁::IdentityValuation, ϕ₂::Valuation)
    ϕ₂
end

function combine(ϕ₁::Valuation, ϕ₂::IdentityValuation)
    ϕ₁
end

function combine(ϕ₁::IdentityValuation, ϕ₂::IdentityValuation)
    ϕ₁
end

function combine(ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    port_labels = [ϕ₁.labels..., ϕ₂.labels...]
    outer_port_labels = ϕ₁.labels ∪ ϕ₂.labels
    junction_labels = outer_port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(ϕ₁.labels)); add_box!(wd, length(ϕ₂.labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    box = oapply(wd, [ϕ₁.box, ϕ₂.box])
    LabeledBox(outer_port_labels, box)
end

function combine(ϕ₁::LabeledBox{<:Any, <:OpenProgram}, ϕ₂::LabeledBox{<:Any, <:OpenProgram})
    Σ₁ = ϕ₁.box; Γ₁ = Σ₁.ϵ.Γ; μ₁ = Σ₁.ϵ.μ; L₁ = Σ₁.L; o₁ = Σ₁.o
    Σ₂ = ϕ₂.box; Γ₂ = Σ₂.ϵ.Γ; μ₂ = Σ₂.ϵ.μ; L₂ = Σ₂.L; o₂ = Σ₂.o
    n₁ = size(L₁, 2); l₁ = ϕ₁.labels; s₁ = l₁[1:n₁]; t₁ = l₁[n₁+1:end]
    n₂ = size(L₂, 2); l₂ = ϕ₂.labels; s₂ = l₂[1:n₂]; t₂ = l₂[n₂+1:end]
    if isdisjoint(s₁, t₂)
        s = s₁ ∪ setdiff(s₂, t₁)
        t = t₁ ∪ t₂
        _t₁ = [t₁; -o₁:-1]
        _t₂ = [t₂; -o₂-o₁:-1-o₁]
        _t  = [t;  -o₂-o₁:-1; t₁ ∩ t₂]
        m = length(t); _m = length(_t)
        U  = [X == Y for X in s₁, Y in s]
        V₁ = [X == Y for X in s₂, Y in _t₁]
        V₂ = [X == Y for X in s₂, Y in s]
        W₁ = [X == Y for X in _t, Y in _t₁]
        W₂ = [
            !(X in t₁ && X in t₂) ? X == Y : i > m ? -(X == Y) : 0
            for (i, X) in enumerate(_t), Y in _t₂]
        K = W₁ + W₂ * L₂ * V₁
        l = [s; t]
        Γ = K * Γ₁ * K' + W₂ * Γ₂ * W₂'
        μ = K * μ₁ + W₂ * μ₂
        L = K * L₁ * U + W₂ * L₂ * V₂
        o = _m - m
        LabeledBox(l, OpenProgram(ClosedProgram(Γ,μ), L, o))
    elseif isdisjoint(s₂, t₁)
        combine(ϕ₂, ϕ₁)
    else
        error()
    end
end

function combine(ϕ₁::LabeledBox{<:Any, <:ClosedProgram}, ϕ₂::LabeledBox{<:Any, <:OpenProgram})

    ϕ₁ = LabeledBox(ϕ₁.labels, OpenProgram(ϕ₁.box))
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
        _ϕ = LabeledBox(ϕ.labels, ClosedProgram(_Γ, _μ))
    else
        _ϕ = ϕ
    end
    _ϕ 
end

function combine(ϕ₁::LabeledBox{<:Any, <:OpenProgram}, ϕ₂::LabeledBox{<:Any, <:ClosedProgram})
    combine(ϕ₂, ϕ₁)
end

function combine(ϕ₁::LabeledBox{<:Any, <:ClosedProgram}, ϕ₂::LabeledBox{<:Any, <:ClosedProgram})
    ϕ₂ = LabeledBox(ϕ₂.labels, OpenProgram(ϕ₂.box))
    combine(ϕ₁, ϕ₂)
end

"""
    project(ϕ::Valuation, x)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation, x)

function project(ϕ::IdentityValuation, x)
    @assert isempty(x)
    ϕ
end

function project(ϕ::LabeledBox, x)
    @assert x ⊆ ϕ.labels
    port_labels = ϕ.labels
    outer_port_labels = collect(x)
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(ϕ.labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    box = oapply(wd, [ϕ.box])
    LabeledBox(outer_port_labels, box)
end

function project(ϕ::LabeledBox{<:Any, <:ClosedProgram}, x)
    Σ = ϕ.box; Γ = Σ.Γ; μ = Σ.μ
    l = ϕ.labels
    mask = [X in x for X in l]
    _Γ = Γ[mask, mask]
    _μ = μ[mask]
    _l = l[mask]
    LabeledBox(_l, ClosedProgram(_Γ, _μ))
end

function project(ϕ::LabeledBox{<:Any, <:OpenProgram}, x)
    Σ = ϕ.box; Γ = Σ.ϵ.Γ; μ = Σ.ϵ.μ; L = Σ.L; o = Σ.o
    n = size(L, 2); l = ϕ.labels; s = l[1:n]; t = l[n+1:end]
    if s ⊆ x 
         mask = [X in x for X in t]
        _mask = [mask; trues(o)]
        _l = [s; t[mask]]
        _Γ = Γ[_mask, _mask]
        _μ = μ[_mask]
        _L = L[_mask, :]
        _o = o
        LabeledBox(_l, OpenProgram(ClosedProgram(_Γ, _μ), _L, _o))
    else
        error()
    end
end

"""
    inference_problem(wd::UndirectedWiringDiagram, box_map::AbstractDict)

See [`inference_problem(wd::UndirectedWiringDiagram, boxes::AbstractVector)`](@ref).
"""
function inference_problem(wd::UndirectedWiringDiagram, box_map::AbstractDict)
    boxes = [box_map[x] for x in subpart(wd, :name)]
    inference_problem(wd, boxes)
end

#FIXME
"""
    inference_problem(wd::UndirectedWiringDiagram, boxes::AbstractVector)

Let ``f`` be an operation in **Cospan** of the form
```math
    B \\xleftarrow{\\mathtt{box}} P \\xrightarrow{\\mathtt{junc}} J
    \\xleftarrow{\\mathtt{junc'}} Q
```
and ``(b_1, \\dots, b_n)`` a sequence of fillers for the boxes in ``f``. Then
`inference_problem(wd, boxes)` constructs a knowledge base
``\\{\\phi_1, \\dots, \\phi_n\\}`` and query ``x \\subseteq J`` such that
```math
    (\\phi_1 \\otimes \\dots \\otimes \\phi_n)^{\\downarrow x} \\cong
    F(f)(b_1, \\dots, b_n),
```
where ``F`` is the **Cospan**-algebra computed by `oapply`.

The operation ``f`` must satify must satisfy the following constraints:
- ``\\mathtt{junc'}`` is injective.
- ``\\mathtt{image}(\\mathtt{junc'}) \\subseteq \\mathtt{image}(\\mathtt{junc})``
- For all ``x, y \\in P``, ``\\mathtt{box}(x) = \\mathtt{box}(y)`` and
  ``\\mathtt{junc}(x) = \\mathtt{junc}(y)`` implies that ``x = y``. 
"""
function inference_problem(wd::UndirectedWiringDiagram, boxes::AbstractVector)
    @assert nboxes(wd) == length(boxes)
    kb_labels = [Int[] for box in boxes]
    for i in ports(wd; outer=false)
        push!(kb_labels[box(wd, i)], junction(wd, i; outer=false))
    end
    kb = [
        LabeledBox(labels, box)
        for (labels, box) in zip(kb_labels, boxes)]
    query = Set(
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true))
    kb, query
end
