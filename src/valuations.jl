"""
    Valuation{T}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct Valuation{T}
    hom::T
    labels::Vector{Int}
    index::Dict{Int, Int}

    function Valuation{T}(hom, labels, index) where T
        @assert length(labels) == length(index)
        new{T}(hom, labels, index)
    end
end

function Valuation{T}(hom, labels) where T
    n = length(labels)
    Valuation{T}(hom, labels, Dict(zip(labels, 1:n)))
end

function convert(::Type{Valuation{T}}, ϕ::Valuation) where T
    Valuation{T}(ϕ.hom, ϕ.labels, ϕ.index)
end

"""
    length(ϕ::Valuation)

Get the size of the domain of ``\\phi``.
"""
function length(ϕ::Valuation)
    length(ϕ.labels)
end

"""
    domain(ϕ::Valuation)

Get the domain of ``\\phi``.
"""
function domain(ϕ::Valuation)
    ϕ.labels
end

"""
    combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T
    n₁ = length(ϕ₁)
    n₂ = length(ϕ₂)
    ls = copy(ϕ₁.labels)

    is₁ = 1:n₁
    is₂ = map(ϕ₂.labels) do l
        get(ϕ₁.index, l) do
            push!(ls, l)
            length(ls)
        end
    end

    n = length(ls)
    wd = UntypedUWD(n)
    add_box!(wd, n₁)
    add_box!(wd, n₂)
    add_junctions!(wd, n)
    set_junction!(wd, 1:n; outer=true)
    set_junction!(wd, [is₁; is₂]; outer=false)
    Valuation{T}(oapply(wd, [ϕ₁.hom, ϕ₂.hom]), ls)
end

function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation{T}) where T <: GaussianSystem
    hom, labels = combine(ϕ₁.hom, ϕ₂.hom, ϕ₁.labels, ϕ₂.labels, ϕ₁.index)
    Valuation{T}(hom, labels)
end

"""
    project(ϕ::Valuation, x)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
function project(ϕ::Valuation{T}, x) where T
    @assert x ⊆ ϕ.labels
    n = length(ϕ); m = length(x)
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> ϕ.index[l], x), n)
    Valuation{T}(oapply(wd, [ϕ.hom]), x)
end

function project(ϕ::Valuation{T}, x) where T <: GaussianSystem
    @assert x ⊆ ϕ.labels
    m = in.(ϕ.labels, [x])
    Valuation{T}(marginal(ϕ.hom, m), ϕ.labels[m])
end

"""
    extend(ϕ::Valuation, objects, x)

Perform the vacuous extension ``\\phi^{\\uparrow x}``
"""
function extend(ϕ::Valuation{T}, objects, x) where T
    @assert ϕ.labels ⊆ x
    n = length(ϕ); m = length(x); i = n
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> get(_ -> i+=1, ϕ.index, l), x), m)
    Valuation{T}(oapply(wd, [ϕ.hom]), x)
end

"""
    expand(ϕ::Valuation, objects, x)
"""
function expand(ϕ::Valuation{T}, x) where T
    n = length(ϕ); m = length(x)
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> ϕ.index[l], x), n)
    Valuation{T}(oapply(wd, [ϕ.hom]), x)
end

"""
    one(::Type{Valuation{T}}) where T

Construct an identity element.
"""
one(::Type{Valuation{T}}) where T

function one(::Type{Valuation{T}}) where {L, T <: StructuredMulticospan{L}}
    Valuation{T}(id(munit(StructuredCospanOb{L})), [])
end

function one(::Type{Valuation{T}}) where T <: GaussianSystem
    Valuation{T}(zero(T, 0), [])
end
