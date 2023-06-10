"""
    GaussianSystem{
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix, 
        T₃ <: AbstractVector,
        T₄ <: AbstractVector,
        T₅}

A Gaussian system.
"""
struct GaussianSystem{
    T₁ <: AbstractMatrix,
    T₂ <: AbstractMatrix,
    T₃ <: AbstractVector,
    T₄ <: AbstractVector}

    P::T₁
    S::T₂
    p::T₃
    s::T₄
    σ::Float64

    @doc """
        GaussianSystem{T₁, T₂, T₃, T₄}(P, S, p, s, σ) where {
            T₁ <: AbstractMatrix,
            T₂ <: AbstractMatrix,
            T₃ <: AbstractVector,
            T₄ <: AbstractVector}

    Construct a Gaussian system by specifying its energy function. 

    Set ``\\sigma = s^\\mathsf{T} S^+ s``, where ``S^+`` is the Moore-Penrose
    psuedoinverse of ``S``.
    """
    function GaussianSystem{T₁, T₂, T₃, T₄}(P, S, p, s, σ) where {
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix,
        T₃ <: AbstractVector,
        T₄ <: AbstractVector}
    
        m = checksquare(P)
        n = checksquare(S)
        @assert m == n == length(p) == length(s)
        new{T₁, T₂, T₃, T₄}(P, S, p, s, σ)
    end
end

const DenseGaussianSystem{T} = GaussianSystem{Matrix{T}, Matrix{T}, Vector{T}, Vector{T}}

"""
    GaussianSystem(
        P::AbstractMatrix,
        S::AbstractMatrix,
        p::AbstractVector,
        s::AbstractVector,
        σ)

Construct a Gaussian system by specifying its energy function. 

Set ``\\sigma = s^\\mathsf{T} S^+ s``, where ``S^+`` is the Moore-Penrose
psuedoinverse of ``S``.
"""
function GaussianSystem(P::T₁, S::T₂, p::T₃, s::T₄, σ) where {
    T₁ <: AbstractMatrix,
    T₂ <: AbstractMatrix,
    T₃ <: AbstractVector,
    T₄ <: AbstractVector}
    
    GaussianSystem{T₁, T₂, T₃, T₄}(P, S, p, s, σ)
end


function convert(::Type{GaussianSystem{T₁, T₂, T₃, T₄}}, Σ::GaussianSystem) where {
    T₁, T₂, T₃, T₄}
    GaussianSystem{T₁, T₂, T₃, T₄}(Σ.P, Σ.S, Σ.p, Σ.s, Σ.σ)
end

"""
    normal(Σ::AbstractMatrix, μ::AbstractVector)

Construct a multivariate normal distribution with covariance matrix `Σ` and mean vector `μ`.
"""
function normal(Σ::AbstractMatrix, μ::AbstractVector)
    V = nullspace(Σ)
    P = pinv(Σ)
    S = V * V'
    GaussianSystem(P, S, P * μ, S * μ, dot(μ, S * μ))
end

function normal(Σ::Eye, μ::AbstractVector)
    n = length(μ)
    GaussianSystem(Eye(n), Zeros(n, n), μ, Zeros(n), 0)
end

function normal(Σ::Zeros{<:Any, 2}, μ::AbstractVector)
    n = length(μ)
    GaussianSystem(Zeros(n, n), Eye(n), Zeros(n), μ, dot(μ, μ))
end

"""
    kernel(Σ::AbstractMatrix, μ::AbstractVector, L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\mathcal{N}(Lx + \\mu, \\Sigma).
```
"""
function kernel(Σ::AbstractMatrix, μ::AbstractVector, L::AbstractMatrix)
    normal(Σ, μ) * [-L I]
end

"""
    length(Σ::GaussianSystem)

Get the dimension of `Σ`.
"""
function length(Σ::GaussianSystem)
    size(Σ.P, 1)
end

"""
    cov(Σ::GaussianSystem)

Get the covariance matrix of `Σ`.
"""
function cov(Σ::GaussianSystem)
    n = length(Σ)
    K = KKT(Σ.P, Σ.S)
    solve!(K, I(n), zeros(n, n))
end

"""
    var(Σ::GaussianSystem)

Get the variances of `Σ`.
"""
function var(Σ::GaussianSystem)
    diag(cov(Σ))
end

"""
    mean(Σ::GaussianSystem)

Get the mean vector of `Σ`.
"""
function mean(Σ::GaussianSystem)
    n = length(Σ)
    K = KKT(Σ.P, Σ.S)
    solve!(K, Σ.p, Σ.s)
end

"""
    invcov(Σ::GaussianSystem)

Get the information matrix of `Σ`.
"""
function invcov(Σ::GaussianSystem)
    Σ.P
end

"""
    ⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)

Compute the tensor product of `Σ₁` and `Σ₂`.
"""
function ⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    GaussianSystem(
        Σ₁.P ⊕ Σ₂.P,
        Σ₁.S ⊕ Σ₂.S,
        [Σ₁.p; Σ₂.p],
        [Σ₁.s; Σ₂.s],
        Σ₁.σ + Σ₂.σ)
end

"""
    *(Σ::GaussianSystem, M::AbstractMatrix)

Construct a Gaussian system with energy function ``E'(x) = E(Mx),`` where ``E`` is the
energy function of `Σ`.
"""
function *(Σ::GaussianSystem, M::AbstractMatrix)
    @assert size(M, 1) == length(Σ)
    GaussianSystem(
        M' * Σ.P * M,
        M' * Σ.S * M,
        M' * Σ.p,
        M' * Σ.s,
        Σ.σ)
end

"""
    +(Σ₁::GaussianSystem, Σ₂::GaussianSystem)

Construct a Gaussian system by summing the energy functions of `Σ₁` and `Σ₂`.
"""
function +(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    @assert length(Σ₁) == length(Σ₂)    
    GaussianSystem(
        Σ₁.P + Σ₂.P,
        Σ₁.S + Σ₂.S,
        Σ₁.p + Σ₂.p,
        Σ₁.s + Σ₂.s,
        Σ₁.σ + Σ₂.σ)
end

"""
    zero(Σ::GaussianSystem)

Construct a Gaussian system with energy function ``E(x) = 0``.
"""
function zero(Σ::GaussianSystem)
    GaussianSystem(zero(Σ.P), zero(Σ.S), zero(Σ.p), zero(Σ.s), 0)
end

function zero(::Type{GaussianSystem{T₁, T₂, T₃, T₄}}, n) where {T₁, T₂, T₃, T₄}
    GaussianSystem{T₁, T₂, T₃, T₄}(Zeros(n, n), Zeros(n, n), Zeros(n), Zeros(n), 0)
end

"""
    pushfwd(Σ::GaussianSystem, M::AbstractMatrix)

Compute the pushforward ``M_*\\Sigma``.
"""
function pushfwd(Σ::GaussianSystem, M::AbstractMatrix)
    @assert size(M, 2) == length(Σ)
    P, S = Σ.P, Σ.S
    p, s = Σ.p, Σ.s
    σ = Σ.σ

    V = nullspace(M')
    K = KKT(P, [S; M])

    m, n = size(M)
    A = solve!(K, zeros(n, m), [zeros(n, m); I(m)])
    a = solve!(K, p, [s; zeros(m)])

    GaussianSystem(
        A' * P * A,
        A' * S * A + V * V',
        A' * (p - P * a),
        A' * (s - S * a),
        a' * (s - S * a) * -1 + σ - s' * a)
end

"""
    marginal(Σ::GaussianSystem, m::AbstractVector{Bool})

Compute the marginal of `Σ` along the indices specified by `m`.
"""
function marginal(Σ::GaussianSystem, m::AbstractVector{Bool})
    P, S = Σ.P, Σ.S
    p, s = Σ.p, Σ.s
    σ = Σ.σ

    n = .!m
    K = KKT(P[n, n], S[n, n])

    A = solve!(K, P[n, m], S[n, m])
    a = solve!(K, p[n],    s[n])

    GaussianSystem(
        P[m, m] + A' * P[n, n] * A - P[m, n] * A - A' * P[n, m],
        S[m, m] - S[m, n] * A,
        p[m]    + A' * P[n, n] * a - P[m, n] * a - A' * p[n],
        s[m]    - S[m, n] * a,
        σ       - s[n]'   * a)
end

"""
    oapply(wd::UndirectedWiringDiagram, box_map::AbstractDict{<:Any, <:GaussianSystem})

See [`oapply(wd::UndirectedWiringDiagram, boxes::AbstractVector{<:GaussianSystem})`](@ref).
"""
function oapply(wd::UndirectedWiringDiagram, box_map::AbstractDict{<:Any, <:GaussianSystem})
    boxes = [box_map[x] for x in subpart(wd, :name)]
    oapply(wd, boxes)
end

"""
    oapply(wd::UndirectedWiringDiagram, boxes::AbstractVector{<:GaussianSystem})

Compose Gaussian systems according to the undirected wiring diagram `wd`.
"""
function oapply(wd::UndirectedWiringDiagram, boxes::AbstractVector{<:GaussianSystem})
    @assert nboxes(wd) == length(boxes)
    L = Bool[
        junction(wd, i; outer=false) == j
        for i in ports(wd; outer=false),
            j in junctions(wd)]
    R = Bool[
        junction(wd, i; outer=true ) == j
        for i in ports(wd; outer=true ),
            j in junctions(wd)]
    Σ = reduce(⊗, boxes; init=GaussianSystem(Bool[;;], Bool[;;], Bool[], Bool[], false))
    pushfwd(Σ * L, R)
 end
