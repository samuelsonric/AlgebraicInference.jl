"""
    Valuation

Abstract type for valuations.

Subtypes should support the following methods:
- [`d(ϕ::Valuation)`](@ref)
- [`↓(ϕ::Valuation, x::AbstractSet)`](@ref)
- [`↑(ϕ::Valuation, x::AbstractSet)`](@ref)
- [`⊗(ϕ₁::Valuation, ϕ₂::Valuation)`](@ref)

References:
- Pouly, M.; Kohlas, J. *Generic Inference. A Unified Theory for Automated Reasoning*; Wiley: Hoboken, NJ, USA, 2011.
"""
abstract type Valuation end

"""
    LabeledBox <: Valuation
"""
struct LabeledBox{T₁, T₂} <: Valuation
    box::T₁
    labels::OrderedSet{T₂}
end

"""
    LabeledBox(box, labels::OrderedSet)
"""
LabeledBox(box, labels::OrderedSet)

"""
    d(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
d(ϕ::Valuation)

function d(ϕ::LabeledBox)
    Set(ϕ.labels)
end

"""
    ↓(ϕ::Valuation, x::AbstractSet)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
↓(ϕ::Valuation, x::AbstractSet)

function ↓(ϕ::LabeledBox, x::AbstractSet)
    ↓(oapply, ϕ, x)
end

function ↓(f, ϕ::LabeledBox, x::AbstractSet)
    @assert x ⊆ ϕ.labels
    labels = OrderedSet(x)
    indices = Dict( label => i 
                    for (i, label) in enumerate(ϕ.labels) )
    composite = UntypedUWD(length(labels))
    add_box!(composite, length(ϕ.labels))
    add_junctions!(composite, length(ϕ.labels))
    for (i, label) in enumerate(ϕ.labels)
        set_junction!(composite, i, i; outer=false)
    end
    for (i, label) in enumerate(labels)
        set_junction!(composite, i, indices[label]; outer=true)
    end
    box = f(composite, [ϕ.box])
    LabeledBox(box, labels)
end

"""
    ↑(ϕ::Valuation, x::AbstractSet)

Perform the vacuous extension ``\\phi^{\\uparrow x}``.
"""
↑(ϕ::Valuation, x::AbstractSet)

function ↑(ϕ::LabeledBox, x::AbstractSet)
    ↑(oapply, ϕ, x)
end

function ↑(f, ϕ::LabeledBox, x::AbstractSet)
    @assert ϕ.labels ⊆ x
    labels = OrderedSet(x)
    indices = Dict( label => i 
                    for (i, label) in enumerate(labels) )
    composite = UntypedUWD(length(labels))
    add_box!(composite, length(ϕ.labels))
    add_junctions!(composite, length(labels))
    for (i, label) in enumerate(ϕ.labels)
        set_junction!(composite, i, indices[label]; outer=false)
    end
    for (i, label) in enumerate(labels)
        set_junction!(composite, i, i; outer=true)
    end
    box = f(composite, [ϕ.box])
    LabeledBox(box, labels)
end

"""
    ⊗(ϕ₁::Valuation, ϕ₂::Valuation)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
⊗(ϕ₁::Valuation, ϕ₂::Valuation)

function ⊗(ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    ⊗(oapply, ϕ₁, ϕ₂)
end

function ⊗(f, ϕ₁::LabeledBox, ϕ₂::LabeledBox)
    labels = ϕ₁.labels ∪ ϕ₂.labels
    indices = Dict( label => i 
                    for (i, label) in enumerate(labels) )
    composite = UntypedUWD(length(labels))
    add_box!(composite, length(ϕ₁.labels))
    add_box!(composite, length(ϕ₂.labels))
    add_junctions!(composite, length(labels))
    for (i, label) in enumerate(flatten([ϕ₁.labels, ϕ₂.labels]))
        set_junction!(composite, i, indices[label]; outer=false)
    end
    for (i, label) in enumerate(labels)
        set_junction!(composite, i, i; outer=true)
    end
    box = f(composite, [ϕ₁.box, ϕ₂.box])
    LabeledBox(box, labels)
end

"""
    ⊗(ϕs::Valuations...)

Combine one more valuations.
"""
function ⊗(ϕs::Valuation...)
    reduce(⊗, ϕs)
end

"""
    -(ϕ::Valuation, X)

Perform the variable elimination ``\\phi^{-X}``.
"""
function -(ϕ::Valuation, X)
    @assert X in d(ϕ)
    x = d(ϕ)
    delete!(x, X)
    ϕ ↓ x
end

#################################################################################
# Algorithms from *Generic Inference. A Unified Theory for Automated Reasoning* #
#################################################################################

"""
    fusion_algorithm(factors::AbstractSet{T},
                     elimination_sequence::OrderedSet) where T <: Valuation

Algorithm 3.1: The Fusion Algorithm
"""
function fusion_algorithm(factors::AbstractSet{T},
                          elimination_sequence::OrderedSet) where T <: Valuation
    Ψ = factors
    for Y in elimination_sequence
        Γ = [ ϕᵢ
              for ϕᵢ in Ψ
              if Y in d(ϕᵢ) ]
        ψ = ⊗(Γ...)
        Ψ = setdiff(Ψ, Γ) ∪ [ψ - Y]
    end
    ⊗(Ψ...)
end

"""
    join_tree_construction(domains::AbstractSet{T},
                           elimination_sequence::OrderedSet) where T <: AbstractSet

Algorithm 3.2: Join Tree Construction
"""
function join_tree_construction(domains::AbstractSet{T},
                                elimination_sequence::OrderedSet) where T <: AbstractSet
    λ = Dict{UUID, T}(); color = Dict{UUID, Bool}()
    V = Set{UUID}(); E = Set{Set{UUID}}()
    l = domains
    for Xᵢ in elimination_sequence
        sᵢ = ∪(( s
                 for s in l
                 if Xᵢ in s )...)
        l  = setdiff(l, ( s 
                          for s in l
                          if Xᵢ in s )) ∪ [setdiff(sᵢ, [Xᵢ])]
        i  = uuid1(); λ[i] = sᵢ; color[i] = true
        for j in V
            if Xᵢ in λ[j] && color[j]
                E = E ∪ [Set([i, j])]
                color[j] = false
            end
        end
        V = V ∪ [i]
    end
    i = uuid1(); λ[i] = ∪(l...)
    for j in V
        if color[j]
            E = E ∪ [Set([i, j])]
            color[j] = false
        end
    end
    V = V ∪ [i]
    V, E, λ
end
