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

function ch(V::Int, E::Set{Set{Int}}, i::Int)
    @assert i < V
    for j in i + 1:V
        if Set([i, j]) in E
            return j
        end
    end
    error()
end

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
        ϕ = ⊗(Γ...)
        Ψ = setdiff(Ψ, Γ) ∪ [ϕ - Y]
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
    λ = T[]; color = Bool[]
    V = 0; E = Set{Set{Int}}()
    l = domains
    for Xᵢ in elimination_sequence
        sᵢ = ∪(( s
                 for s in l
                 if Xᵢ in s )...)
        setdiff!(l, ( s 
                      for s in l
                      if Xᵢ in s )); push!(l, setdiff(sᵢ, [Xᵢ]))
        i = V + 1; push!(λ, sᵢ); push!(color, true)
        for j in 1:V
            if Xᵢ in λ[j] && color[j]
                push!(E, Set([i, j]))
                color[j] = false
            end
        end
        V += 1
    end
    i = V + 1; push!(λ, ∪(l...))
    for j in 1:V
        if color[j]
            push!(E, Set([i, j]))
            color[j] = false
        end
    end
    V += 1
    V, E, λ
end
