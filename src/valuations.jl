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
    UWDBox{T₁, T₂} <: Valuation{T₁}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct UWDBox{T₁, T₂} <: Valuation{T₁}
    labels::Vector{T₁}
    box::T₂

    @doc """
        UWDBox{T₁, T₂}(labels, box) where {T₁, T₂}
    """
    function UWDBox{T₁, T₂}(labels, box) where {T₁, T₂}
        new{T₁, T₂}(labels, box)
    end
end

"""
    UWDBox(labels, box, unique=true)
"""
function UWDBox(labels, box, unique)
    unique ? UWDBox(labels, box) : begin
        port_labels = labels
        outer_port_labels = collect(Set(labels))
        junction_labels = outer_port_labels
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
        UWDBox(outer_port_labels, oapply(wd, [box]))
    end
end

function UWDBox(labels, box)
    UWDBox(collect(labels), box)
end

function UWDBox(labels::Vector{T₁}, box::T₂) where {T₁, T₂}
   UWDBox{T₁, T₂}(labels, box)
end

function convert(::Type{UWDBox{T₁, T₂}}, ϕ::UWDBox) where {T₁, T₂}
    UWDBox{T₁, T₂}(ϕ.labels, ϕ.box)
end

function length(ϕ::UWDBox)
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

function domain(ϕ::UWDBox)
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

function project(ϕ::IdentityValuation, x)
    @assert isempty(x)
    ϕ
end

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

Construct an identity element of type `T`.
"""
one(T::Type{<:Valuation})

function one(::Type{Union{IdentityValuation{T₁}, T₂}}) where {T₁, T₂ <: Valuation{T₁}}
    IdentityValuation{T₁}()
end

function one(::Type{UWDBox{T₁, T₂}}) where {T₁, T₂}
    UWDBox{T₁, T₂}([], oapply(UntypedUWD(), T₂[]))
end

function one(::Type{UWDBox{T₁, GaussianSystem{T₂, T₃, T₄, T₅, T₆}}}) where {
    T₁, T₂, T₃, T₄, T₅, T₆}
    UWDBox(T₁[], GaussianSystem{T₂, T₃, T₄, T₅, T₆}([;;], [;;], [], [], 0))
end
