"""
    Valuation

Abstract type for valuations.

Subtypes should support the following methods:
- [`d(ϕ::Valuation)`](@ref)
- [`↓(ϕ::Valuation, x::AbstractSet)`](@ref)
- [`⊗(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*; Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation end

"""
    LabeledSystem <: Valuation

A labeled Gaussian system.
"""
struct LabeledSystem{T₁, T₂ <: AbstractSystem} <: Valuation
    labels::OrderedSet{T₁}
    system::T₂

    function LabeledSystem(labels::OrderedSet{T₁}, system::T₂) where {T₁, T₂ <: AbstractSystem}
        @assert length(labels) == length(system)
        new{T₁, T₂}(labels, system)
    end
end

"""
    LabeledSystem(labels::OrderedSet, system::AbstractSystem)

Construct a labeled Gaussian system.
"""
LabeledSystem(labels::OrderedSet, system::AbstractSystem)

"""
    d(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
d(ϕ::Valuation)

function d(ϕ::LabeledSystem)
    Set(ϕ.labels)
end

"""
    ⊗(ϕ₁::Valuation, ϕ₂::Valuation)

Compute the combination ``\\phi_1 \\otimes \\phi_2``.
"""
⊗(ϕ₁::Valuation, ϕ₂::Valuation)

function ⊗(ϕ₁::LabeledSystem, ϕ₂::LabeledSystem)
    labels = ϕ₁.labels ∪ ϕ₂.labels

    M = [
        i == j
        for i in flatten((ϕ₁.labels, ϕ₂.labels))
        for j in labels
    ]

    system = M \ (ϕ₁.system ⊗ ϕ₂.system)
    LabeledSystem(labels, system)
end

"""
    ↓(ϕ::Valuation, x::AbstractSet)

Compute the projection ``\\phi^{\\downarrow x}``.
"""
↓(ϕ::Valuation, x::AbstractSet)

function ↓(ϕ::LabeledSystem, x::AbstractSet)
    @assert x ⊆ labels(ϕ)
    labels = OrderedSet(x)

    M = [
        i == j
        for i in ϕ.labels
        for j in labels
    ]

    system = M * ϕ.system
    LabeledSystem(labels, system)
end
