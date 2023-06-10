"""
    Valuation{T}

Abstract type for valuations in a stable valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)
- [`project(ϕ::Valuation, x)`](@ref)
- [`duplicate(ϕ::Valuation, x)`](@ref)
- [`one(::Type{<:Valuation}, x)`](@ref)

Valuations are parametrized by the type of the variables in their variable system. If
`isa(ϕ, Valuation{T})`, then `domain(ϕ)` should return a container with element type `T`.
"""
abstract type Valuation{T} end

"""
    UWDBox{T₁, T₂} <: Valuation{T₂}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct UWDBox{T₁, T₂} <: Valuation{T₂}
    box::T₁
    labels::Vector{T₂}
end

"""
    UWDBox{T₁, T₂}(labels, box, unique::Bool=true) where {T₁, T₂}
"""
function UWDBox{T₁, T₂}(box, labels, unique::Bool) where {T₁, T₂}
    if unique || length(labels) == length(Set(labels))
        UWDBox{T₁, T₂}(box, labels)
    else
        port_labels = labels
        outer_port_labels = collect(Set(labels))
        junction_labels = outer_port_labels
        junction_indices = Dict(
            label => i
            for (i, label) in enumerate(junction_labels))
        wd = UntypedUWD(length(outer_port_labels))
        add_box!(wd, length(port_labels))
        add_junctions!(wd, length(junction_labels))
        for (i, label) in enumerate(port_labels)
            set_junction!(wd, i, junction_indices[label]; outer=false)
        end
        for (i, label) in enumerate(outer_port_labels)
            set_junction!(wd, i, junction_indices[label]; outer=true)
        end
        UWDBox{T₁, T₂}(oapply(wd, [box]), outer_port_labels)
    end
end

function convert(::Type{UWDBox{T₁, T₂}}, ϕ::UWDBox) where {T₁, T₂}
    UWDBox{T₁, T₂}(ϕ.box, ϕ.labels)
end

"""
    length(ϕ::Valuation)

Get the size of the domain of ``\\phi``.
"""
function length(ϕ::Valuation)
    length(domain(ϕ))
end

"""
    domain(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
domain(ϕ::Valuation)

function domain(ϕ::UWDBox)
    ϕ.labels
end

"""
    combine(ϕ₁::Valuation, ϕ₂::Valuation)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
combine(ϕ₁::Valuation, ϕ₂::Valuation)

function combine(ϕ₁::UWDBox{T₁, T₂}, ϕ₂::UWDBox{T₁, T₂}) where {T₁, T₂}
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
    UWDBox{T₁, T₂}(box, outer_port_labels)
end

function combine(ϕ₁::UWDBox{T₁, T₂}, ϕ₂::UWDBox{T₁, T₂}) where {T₁ <: GaussianSystem, T₂}
    l = ϕ₁.labels ∪ ϕ₂.labels
    UWDBox{T₁, T₂}(extend(ϕ₁.box, ϕ₁.labels, l) + extend(ϕ₂.box, ϕ₂.labels, l), l)
end

"""
    project(ϕ::Valuation, x)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation, x)

function project(ϕ::UWDBox{T₁, T₂}, x) where {T₁, T₂}
    @assert x ⊆ ϕ.labels
    port_labels = ϕ.labels
    outer_port_labels = collect(x)
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(port_labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, i; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    box = oapply(wd, [ϕ.box])
    UWDBox{T₁, T₂}(box, outer_port_labels)
end

function project(ϕ::UWDBox{T₁, T₂}, x) where {T₁ <: GaussianSystem, T₂}
    @assert x ⊆ ϕ.labels
    m = [X in x for X in ϕ.labels]
    UWDBox{T₁, T₂}(marginal(ϕ.box, m), ϕ.labels[m])
end

"""
    one(T::Type{<:Valuation})

Construct the neutral element ``e_\\lozenge``.
"""
function one(T::Type{<:Valuation})
    one(T, [])
end

"""
    one(T::Type{<:Valuation}, x)

Construct the neutral element ``e_x``.
"""
one(T::Type{<:Valuation}, x)

function one(::Type{UWDBox{T₁, T₂}}, x) where {T₁, T₂}
    n = length(x)
    wd = UntypedUWD(n)
    add_junctions!(wd, n)
    for i in 1:n
        set_junction!(wd, i, i; outer=true)
    end
    UWDBox{T₁, T₂}(oapply(wd, T₁[]), x)
end

function one(::Type{UWDBox{T₁, T₂}}, x) where {T₁ <: GaussianSystem, T₂}
    n = length(x)
    UWDBox{T₁, T₂}(zero(T₁, n), x)
end

"""
    duplicate(ϕ::Valuation, x)
"""
duplicate(ϕ::Valuation, x)

function duplicate(ϕ::UWDBox, x)
    port_labels = ϕ.labels
    outer_port_labels = x
    junction_labels = port_labels
    junction_indices = Dict(
        label => i
        for (i, label) in enumerate(junction_labels))
    wd = UntypedUWD(length(outer_port_labels))
    add_box!(wd, length(port_labels))
    add_junctions!(wd, length(junction_labels))
    for (i, label) in enumerate(port_labels)
        set_junction!(wd, i, i; outer=false)
    end
    for (i, label) in enumerate(outer_port_labels)
        set_junction!(wd, i, junction_indices[label]; outer=true)
    end
    oapply(wd, [ϕ.box])
end
