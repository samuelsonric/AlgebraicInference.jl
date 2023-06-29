"""
    Valuation{T₁, T₂}

A filler for a box in an undirected wiring diagram, labeled with the junctions to which the
box is incident.
"""
struct Valuation{T}
    morphism::T
    labels::Vector{Int}
    index::LittleDict{Int, Int, Vector{Int}, Vector{Int}}

    function Valuation{T}(morphism, labels, index) where T
        @assert length(labels) == length(index)
        new{T}(morphism, labels, index)
    end
end

# FIXME
function Valuation{T}(morphism, labels) where T
    n = length(labels)
    index = LittleDict{Int, Int, Vector{Int}, Vector{Int}}(labels, 1:n)
    Valuation{T}(morphism, labels, index)
end

function convert(::Type{Valuation{T}}, ϕ::Valuation) where T
    Valuation{T}(ϕ.morphism, ϕ.labels, ϕ.index)
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
    combine(ϕ₁::Valuation, ϕ₂::Valuation, objects)

Perform the combination ``\\phi_1 \\otimes \\phi_2``.
"""
function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation, objects) where T
    ls = copy(ϕ₁.labels)
    ix = copy(ϕ₁.index)
    n₁ = length(ϕ₁)
    n₂ = length(ϕ₂)

    is = map(ϕ₂.labels) do l
        get!(ix, l) do
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
    set_junction!(wd, [1:n₁; is]; outer=false)
    Valuation{T}(oapply(wd, [ϕ₁.morphism, ϕ₂.morphism], objects[ls]), ls, ix)
end

function combine(ϕ₁::Valuation{T}, ϕ₂::Valuation, objects) where T <: GaussianSystem

    cs = cumsum(objects[ϕ₁.labels])
    ls = copy(ϕ₁.labels)
    ix = copy(ϕ₁.index)
    is = Int[]
    n₁ = n = length(ϕ₁.morphism)

    for l₂ in ϕ₂.labels
        o = objects[l₂]
        i = get!(ix, l₂) do
            n += o
            push!(cs, n)
            push!(ls, l₂)
            length(ls)
        end
        append!(is, cs[i] - o + 1:cs[i])
    end
    
    P = zeros(n, n)
    S = zeros(n, n)
    p = zeros(n)
    s = zeros(n)
    
    Σ₁ = ϕ₁.morphism
    P[1:n₁, 1:n₁] = Σ₁.P
    S[1:n₁, 1:n₁] = Σ₁.S
    p[1:n₁] = Σ₁.p
    s[1:n₁] = Σ₁.s

    Σ₂ = ϕ₂.morphism
    P[is, is] .+= Σ₂.P
    S[is, is] .+= Σ₂.S
    p[is] .+= Σ₂.p
    s[is] .+= Σ₂.s
  
    σ = Σ₁.σ + Σ₂.σ
    Valuation{T}(GaussianSystem(P, S, p, s, σ), ls, ix)
end

"""
    project(ϕ::Valuation, variables, objects)

Perform the projection ``\\phi^{\\downarrow x}``.
"""
function project(ϕ::Valuation{T}, variables, objects) where T
    n = length(ϕ)
    m = length(variables)
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> ϕ.index[l], variables), n)
    Valuation{T}(oapply(wd, [ϕ.morphism], objects[ϕ.labels]), variables)
end

function project(ϕ::Valuation{T}, variables, objects) where T <: GaussianSystem
    ls = Int[]; is₁ = Int[]; is₂ = Int[]
    n = 0
    
    for l in ϕ.labels
        o = objects[l]
        if l in variables
            push!(ls, l)
            is = is₁
        else
            is = is₂
        end
        append!(is, n + 1:n + o)
        n += o
    end

    Σ = ϕ.morphism
    P₁₁ = Σ.P[is₁, is₁]; P₁₂ = Σ.P[is₁, is₂]
    S₁₁ = Σ.S[is₁, is₁]; S₁₂ = Σ.S[is₁, is₂]

    P₂₁ = Σ.P[is₂, is₁]; P₂₂ = Σ.P[is₂, is₂]
    S₂₁ = Σ.S[is₂, is₁]; S₂₂ = Σ.S[is₂, is₂]

    p₁ = Σ.p[is₁]; p₂ = Σ.p[is₂]
    s₁ = Σ.s[is₁]; s₂ = Σ.s[is₂]

    σ₁ = Σ.σ

    K = KKT(P₂₂, S₂₂)
    A = solve!(K, P₂₁, S₂₁)
    a = solve!(K, p₂,  s₂)

    P = P₁₁ + A'  * P₂₂ * A - P₁₂ * A - A' * P₂₁
    S = S₁₁ - A'  * S₂₂ * A
    p = p₁  + A'  * P₂₂ * a - P₁₂ * a - A' * p₂
    s = s₁  - S₁₂ * a
    σ = σ₁  - s₂' * a

    Valuation{T}(GaussianSystem(P, S, p, s, σ), ls)
end

"""
    extend(ϕ::Valuation, variables, objects)

Perform the vacuous extension ``\\phi^{\\uparrow x}``
"""
function extend(ϕ::Valuation{T}, variables, objects) where T
    ls = copy(ϕ.labels)
    ix = copy(ϕ.index)

    for l in variables
        get!(ix, l) do
            push!(ls, l)
            length(ls)
        end
    end

    n = length(ϕ)
    m = length(variables)
    wd = cospan_diagram(UntypedUWD, 1:n, 1:m, m)
    Valuation{T}(oapply(wd, [ϕ.morphism], objects[ls]), ls, ix)
end

function extend(ϕ::Valuation{T}, variables, objects) where T <: GaussianSystem
    ls = copy(ϕ.labels)
    ix = copy(ϕ.index)

    n = 0

    for l in variables
        get!(ix, l) do
            push!(ls, l)
            length(ls)
        end
        n += objects[l]
    end

    P = zeros(n, n)
    S = zeros(n, n)
    p = zeros(n)
    s = zeros(n)

    Σ = ϕ.morphism
    m = length(Σ)
    P[1:m, 1:m] = Σ.P
    S[1:m, 1:m] = Σ.S
    p[1:m] = Σ.p
    s[1:m] = Σ.s
 
    σ = ϕ.morphism.σ
    Valuation{T}(GaussianSystem(P, S, p, s, σ), ls, ix)
end

"""
    expand(ϕ::Valuation, variables, objects)
"""
function expand(ϕ::Valuation{T}, variables, objects) where T
    n = length(ϕ)
    wd = cospan_diagram(UntypedUWD, 1:n, map(l -> ϕ.index[l], variables), n)
    convert(T, oapply(wd, [ϕ.morphism], objects[ϕ.labels]))
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
