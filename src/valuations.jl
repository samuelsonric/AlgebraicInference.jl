"""
    Valuation{T}

Abstract type for valuations in a valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)
- [`project(ϕ::Valuation, x)`](@ref)

Valuations are parametrized by the type of the variables in their variable system. If
`isa(ϕ, Valuation{T})`, then `domain(ϕ)` should return a container with element type `T`.
"""
abstract type Valuation{T} end

"""
    IdentityValuation{T} <: Valuation{T}

The identity element ``e``.
"""
struct IdentityValuation{T} <: Valuation{T} end

"""
    LabeledBox{T₁, T₂} <: Valuation{T₁}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
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

function convert(::Type{LabeledBox{T₁, T₂}}, ϕ::LabeledBox) where {T₁, T₂}
    LabeledBox(convert(Vector{T₁}, ϕ.labels), convert(T₂, ϕ.box))
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
    T[]
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
    LabeledBox(l, extend(l, ϕ₁.labels, ϕ₁.box) + extend(l, ϕ₂.labels, ϕ₂.box))
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
    @assert x ⊆ ϕ.labels
    m = [X in x for X in ϕ.labels]
    LabeledBox(ϕ.labels[m], marginal(ϕ.box, m))
end

"""
    one(T::Type{<:Valuation})

Construct an identity element of type `T`.
"""
one(T::Type{<:Valuation})

function one(::Type{Union{IdentityValuation{T₁}, T₂}}) where {T₁, T₂ <: Valuation{T₁}}
    IdentityValuation{T₁}()
end

function one(::Type{LabeledBox{T₁, GaussianSystem{T₂, T₃, T₄, T₅, T₆}}}) where {
    T₁, T₂, T₃, T₄, T₅, T₆}
    
    LabeledBox(T₁[], GaussianSystem(
        convert(T₂, [;;]),
        convert(T₃, [;;]),
        convert(T₄, []),
        convert(T₅, []),
        zero(T₆)))
end

"""
    inference_problem(wd::UndirectedWiringDiagram, boxes)

Translate an undirected wiring diagram
```math
    B \\xleftarrow{\\mathtt{box}} P \\xrightarrow{\\mathtt{junc}} J
    \\xleftarrow{\\mathtt{junc'}} Q
```
into an inference problem in a valuation algebra.

The diagram must satisfy the following constraints:
- ``\\mathtt{junc'}`` is injective.
- ``\\mathtt{image}(\\mathtt{junc'}) \\subseteq \\mathtt{image}(\\mathtt{junc})``
- For all ``x, y \\in P``, ``\\mathtt{box}(x) = \\mathtt{box}(y)`` and
  ``\\mathtt{junc}(x) = \\mathtt{junc}(y)`` implies that ``x = y``. 
"""
function inference_problem(wd::UndirectedWiringDiagram, boxes)
    inference_problem(wd, collect(boxes))
end

"""
    inference_problem(wd::UndirectedWiringDiagram, box_map::AbstractDict)

See [`inference_problem(wd::UndirectedWiringDiagram, boxes)`](@ref).
"""
function inference_problem(wd::UndirectedWiringDiagram, box_map::AbstractDict)
    boxes = [box_map[x] for x in subpart(wd, :name)]
    inference_problem(wd, boxes)
end

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
