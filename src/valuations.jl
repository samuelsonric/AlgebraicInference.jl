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

function length(ϕ::LabeledBox)
    length(ϕ.labels)
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

function combine(ϕ₁::LabeledBox{<:Any, <:GaussianSystem}, ϕ₂::LabeledBox{<:Any, <:GaussianSystem})
    l = ϕ₁.labels ∪ ϕ₂.labels
    LabeledBox(l, extend(l, ϕ₁).box + extend(l, ϕ₂).box)
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

function project(ϕ::LabeledBox{<:Any, <:GaussianSystem}, x)
    P, S = ϕ.box.P, ϕ.box.S
    p, s = ϕ.box.p, ϕ.box.s

    m = [X in x for X in ϕ.labels]
    n = .!m

    _P = pinv(P[n, n])
    _S = pinv(S[n, n])

    Π = _P * P[n, m]
    Σ = _S * S[n, m]
    π = _P * p[n]
    σ = _S * s[n]

    V = nullspace(S[n, n]; atol=1e-10)
    Ω = P[n, n] - P[n, n] * V * pinv(V' * P[n, n] * V) * V' * P[n, n]

    LabeledBox(ϕ.labels[m], GaussianSystem(
        P[m, m] - P[m, n] * Π + (Π - Σ)' * Ω * (Π - Σ),
        S[m, m] - S[m, n] * Σ,
        p[m] - P[m, n] * π + (Π - Σ)' * Ω * (π - σ),
        s[m] - S[m, n] * σ))
end

#=
function project(ϕ::LabeledBox{<:Any, <:GaussianSystem}, x)
    P, S = ϕ.box.P, ϕ.box.S
    p, s = ϕ.box.p, ϕ.box.s

    m = [X in x for X in ϕ.labels]
    n = .!m

    _P = qr(P[n, n], Val(true))
    _S = qr(S[n, n], Val(true))

    Π = _P \ P[n, m]
    Σ = _S \ S[n, m]
    π = _P \ p[n]
    σ = _S \ s[n]

    V = nullspace(S[n, n]; atol=1e-10)
    Ω = P[n, n] - P[n, n] * V * pinv(V' * P[n, n] * V) * V' * P[n, n]

    LabeledBox(ϕ.labels[m], GaussianSystem(
        P[m, m] - P[m, n] * Π + (Π - Σ)' * Ω * (Π - Σ),
        S[m, m] - S[m, n] * Σ,
        p[m] - P[m, n] * π + (Π - Σ)' * Ω * (π - σ),
        s[m] - S[m, n] * σ))
end
=#

function extend(l, ϕ::LabeledBox{<:Any, <:GaussianSystem{
    <:AbstractMatrix{T₁},
    <:AbstractMatrix{T₂},
    <:AbstractVector{T₃},
    <:AbstractVector{T₄}}}) where {T₁, T₂, T₃, T₄}
    
    n = length(l); _n = length(ϕ)
    P = zeros(T₁, n, n)
    S = zeros(T₂, n, n)
    p = zeros(T₃, n)
    s = zeros(T₄, n)
    
    for _i in 1:_n
        i = findfirst(X -> X == ϕ.labels[_i], l)
        P[i, i] = ϕ.box.P[_i, _i]
        S[i, i] = ϕ.box.S[_i, _i]        
        p[i] = ϕ.box.p[_i]
        s[i] = ϕ.box.s[_i]
        
        for _j in _i+1:_n
            j = findfirst(X -> X == ϕ.labels[_j], l)
            P[i, j] = ϕ.box.P[_i, _j]
            S[i, j] = ϕ.box.S[_i, _j]
            P[j, i] = ϕ.box.P[_j, _i]
            S[j, i] = ϕ.box.S[_j, _i]
        end
    end
    
    LabeledBox(l, GaussianSystem(P, S, p, s))
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

function inference_problem(wd::UntypedRelationDiagram{<:Any, T}, boxes::AbstractVector) where T
    @assert nboxes(wd) == length(boxes)
    kb_labels = [T[] for box in boxes]
    for i in ports(wd; outer=false)
        push!(kb_labels[box(wd, i)], subpart(wd, junction(wd, i; outer=false), :variable))
    end
    kb = [
        LabeledBox(labels, box)
        for (labels, box) in zip(kb_labels, boxes)]
    query = Set(
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true))
    kb, query
end
