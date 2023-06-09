"""
    Valuation{T}

Abstract type for valuations in a stable valuation algebra.

Subtypes should specialize the following methods:
- [`domain(ϕ::Valuation)`](@ref)
- [`combine(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)
- [`project(ϕ::Valuation, x)`](@ref)
- [`one(::Type{<:Valuation}, x)`](@ref)

Valuations are parametrized by the type of the variables in their variable system. If
`isa(ϕ, Valuation{T})`, then `domain(ϕ)` should return a container with element type `T`.
"""
abstract type Valuation{T} end

"""
    UWDBox{T₁, T₂} <: Valuation{T₁}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct UWDBox{T₁, T₂} <: Valuation{T₁}
    labels::Vector{T₁}
    box::T₂
end

"""
    UWDBox{T₁, T₂}(labels, box, unique::Bool=true) where {T₁, T₂}
"""
function UWDBox{T₁, T₂}(labels, box, unique::Bool) where {T₁, T₂}
    if unique || length(labels) == length(Set(labels))
        UWDBox{T₁, T₂}(labels, box)
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
        UWDBox{T₁, T₂}(outer_port_labels, oapply(wd, [box]))
    end
end

"""
    UWDBox(labels, box, unique::Bool=true)
"""
function UWDBox(labels, box, unique::Bool)
    if unique || length(labels) == length(Set(labels))
        UWDBox(labels, box)
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
        UWDBox(outer_port_labels, oapply(wd, [box]))
    end
end

function convert(::Type{UWDBox{T₁, T₂}}, ϕ::UWDBox) where {T₁, T₂}
    UWDBox{T₁, T₂}(ϕ.labels, ϕ.box)
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

function combine(ϕ₁::UWDBox, ϕ₂::UWDBox)
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
    UWDBox(outer_port_labels, box)
end

function combine(ϕ₁::UWDBox{<:Any, <:GaussianSystem}, ϕ₂::UWDBox{<:Any, <:GaussianSystem})
    l = ϕ₁.labels ∪ ϕ₂.labels
    UWDBox(l, extend(ϕ₁.box, ϕ₁.labels, l) + extend(ϕ₂.box, ϕ₂.labels, l))
end

"""
    project(ϕ::Valuation, x)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
project(ϕ::Valuation, x)

function project(ϕ::UWDBox, x)
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
    UWDBox(outer_port_labels, box)
end

function project(ϕ::UWDBox{<:Any, <:GaussianSystem}, x)
    @assert x ⊆ ϕ.labels
    m = [X in x for X in ϕ.labels]
    UWDBox(ϕ.labels[m], marginal(ϕ.box, m))
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
    box = oapply(wd, T₂[])
    UWDBox{T₁, T₂}(x, box)
end

function one(::Type{UWDBox{T₁, GaussianSystem{T₂, T₃, T₄, T₅, T₆}}}, x) where {
    T₁, T₂, T₃, T₄, T₅, T₆}
    n = length(x)
    UWDBox{T₁, GaussianSystem{T₂, T₃, T₄, T₅, T₆}}(x, GaussianSystem(
        Zeros(n, n),
        Zeros(n, n),
        Zeros(n),
        Zeros(n),
        0))
end
