"""
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}

A Gaussian system.
"""
struct GaussianSystem{T₁, T₂, T₃, T₄, T₅}
    P::T₁
    S::T₂
    p::T₃
    s::T₄
    σ::T₅

    @doc """
        GaussianSystem{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ) where {
            T₁ <: AbstractMatrix,
            T₂ <: AbstractMatrix,
            T₃ <: AbstractVector,
            T₄ <: AbstractVector,
            T₅ <: Real}

    Construct a Gaussian system by specifying its energy function. 

    You should set `σ` equal to ``s^\\mathsf{T} S^+ s``, where ``S^+`` is the Moore-Penrose
    psuedoinverse of ``S``.
    """
    function GaussianSystem{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ) where {
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix,
        T₃ <: AbstractVector,
        T₄ <: AbstractVector,
        T₅ <: Real}
    
        m = checksquare(P)
        n = checksquare(S)
        @assert m == n == length(p) == length(s)
        new{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ)
    end
end

const CanonicalForm{T₁, T₂} = GaussianSystem{
    T₁,
    ZerosMatrix{Bool, Tuple{OneTo{Int}, OneTo{Int}}},
    T₂,
    ZerosVector{Bool, Tuple{OneTo{Int}}},
    Bool}

const DenseGaussianSystem{T} = GaussianSystem{
    Matrix{T},
    Matrix{T},
    Vector{T},
    Vector{T},
    T}

const DenseCanonicalForm{T} = CanonicalForm{
    Matrix{T},
    Vector{T}}

"""
    GaussianSystem(
        P::AbstractMatrix,
        S::AbstractMatrix,
        p::AbstractVector,
        s::AbstractVector,
        σ::Real)

Construct a Gaussian system by specifying its energy function. 

You should set `σ` equal to ``s^\\mathsf{T} S^+ s``, where ``S^+`` is the Moore-Penrose
psuedoinverse of ``S``.
"""
function GaussianSystem(P::T₁, S::T₂, p::T₃, s::T₄, σ::T₅) where {
    T₁ <: AbstractMatrix,
    T₂ <: AbstractMatrix,
    T₃ <: AbstractVector,
    T₄ <: AbstractVector,
    T₅ <: Real}
    
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ)
end

function Base.convert(::Type{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}, Σ::GaussianSystem) where {
    T₁, T₂, T₃, T₄, T₅}
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Σ.P, Σ.S, Σ.p, Σ.s, Σ.σ)
end

function Base.convert(::Type{CanonicalForm{T₁, T₂}}, Σ::GaussianSystem) where {T₁, T₂}
    @assert iszero(Σ.S)
    @assert iszero(Σ.s)
    @assert iszero(Σ.σ)

    n = length(Σ)
    CanonicalForm{T₁, T₂}(Σ.P, Zeros(n, n), Σ.p, Zeros(n), 0)
end 

function Base.convert(::Type{T}, d::NormalCanon) where T <: GaussianSystem
    convert(T, GaussianSystem([d.λ;;], Zeros(1, 1), [d.η], Zeros(1), 0))
end

function Base.convert(::Type{T}, d::Normal) where T <: GaussianSystem
    convert(T, normal(d.μ, d.σ))
end

function Base.convert(::Type{T}, cpd::LinearGaussianCPD) where T <: GaussianSystem
    convert(T, kernel(cpd.a, cpd.b, cpd.σ))
end

function Base.convert(::Type{T}, cpd::StaticCPD) where T <: GaussianSystem
    convert(T, cpd.d)
end

"""
    normal(μ::AbstractVector, Σ::AbstractMatrix)

Construct a multivariate normal distribution with mean vector `μ` and covariance matrix `Σ`.
"""
function normal(μ::AbstractVector, Σ::AbstractMatrix)
    V = nullspace(Σ)
    P = pinv(Σ)
    S = V * V'
    GaussianSystem(P, S, P * μ, S * μ, dot(μ, S * μ))
end

"""
    normal(μ::Real, σ::Real)

Construct a normal distribution with mean `μ` and standard deviation `σ`.
"""
function normal(μ::Real, σ::Real)
    normal([μ], [σ^2;;])
end

function normal(μ::AbstractVector, Σ::Eye)
    n = length(μ)
    GaussianSystem(Eye(n), Zeros(n, n), μ, Zeros(n), 0)
end

function normal(μ::AbstractVector, Σ::ZerosMatrix)
    n = length(μ)
    GaussianSystem(Zeros(n, n), Eye(n), Zeros(n), μ, dot(μ, μ))
end

"""
    kernel(L::AbstractMatrix, μ::AbstractVector, Σ::AbstractMatrix)

Construct a conditional distribution of the form
``(y \\mid x) \\sim \\mathcal{N}(Lx + \\mu, \\Sigma).``
"""
function kernel(L::AbstractMatrix, μ::AbstractVector, Σ::AbstractMatrix)
    normal(μ, Σ) * [-L I]
end

"""
    kernel(l::AbstractVector, μ::Real, σ::Real)

Construct a conditional distribution of the form
``(y \\mid x) \\sim \\mathcal{N}(l^\\mathsf{T}x + \\mu, \\sigma^2).``
"""
function kernel(l::AbstractVector, μ::Real, σ::Real)
    kernel(reshape(l, 1, length(l)), [μ], [σ^2;;])
end

"""
    length(Σ::GaussianSystem)

Get the dimension of `Σ`.
"""
function Base.length(Σ::GaussianSystem)
    size(Σ.P, 1)
end

"""
    cov(Σ::GaussianSystem)

Get the covariance matrix of `Σ`.
"""
function Statistics.cov(Σ::GaussianSystem)
    n = length(Σ)
    K = KKT(Σ.P, Σ.S)
    solve!(K, Eye(n), Zeros(n, n))
end

"""
    var(Σ::GaussianSystem)

Get the variances of `Σ`.
"""
function Statistics.var(Σ::GaussianSystem)
    diag(cov(Σ))
end

"""
    mean(Σ::GaussianSystem)

Get the mean vector of `Σ`.
"""
function Statistics.mean(Σ::GaussianSystem)
    n = length(Σ)
    K = KKT(Σ.P, Σ.S)
    solve!(K, Σ.p, Σ.s)
end

"""
    invcov(Σ::GaussianSystem)

Get the precision matrix of `Σ`.
"""
function Distributions.invcov(Σ::GaussianSystem)
    Σ.P
end

"""
    ⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)

Compute the tensor product of `Σ₁` and `Σ₂`.
"""
function Catlab.:⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
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
function Base.:*(Σ::GaussianSystem, M::AbstractMatrix)
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
function Base.:+(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
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
function Base.zero(Σ::GaussianSystem)
    GaussianSystem(zero(Σ.P), zero(Σ.S), zero(Σ.p), zero(Σ.s), 0)
end

function Base.zero(::Type{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}, n) where {T₁, T₂, T₃, T₄, T₅}
    @assert n >= 0
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Zeros(n, n), Zeros(n, n), Zeros(n), Zeros(n), 0)
end

"""
    pushforward(Σ::GaussianSystem, M::AbstractMatrix)

Compute the pushforward ``M_*\\Sigma``.
"""
function pushforward(Σ::GaussianSystem, M::AbstractMatrix)
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
    oapply(wd::AbstractUWD, homs::AbstractVector{<:GaussianSystem}, obs::AbstractVector)

Compose Gaussian systems according to the undirected wiring diagram `wd`.
"""
function Catlab.oapply(wd::AbstractUWD, homs::AbstractVector{<:GaussianSystem}, obs::AbstractVector)
    @assert nboxes(wd) == length(homs)
    @assert njunctions(wd) == length(obs)

    juncs = collect(subpart(wd, :junction))
    query = collect(subpart(wd, :outer_junction))

    n = sum(obs)
    L = falses(sum(obs[juncs]), n)
    R = falses(sum(obs[query]), n)

    cms = cumsum(obs)

    for ((i, j), m) in zip(enumerate(juncs), cumsum(obs[juncs]))
        o = obs[j]
        L[m - o + 1:m, cms[j] - o + 1:cms[j]] = I(o)
    end

    for ((i, j), m) in zip(enumerate(query), cumsum(obs[query]))
        o = obs[j]
        R[m - o + 1:m, cms[j] - o + 1:cms[j]] = I(o)
    end

    Σ = reduce(⊗, homs; init=zero(DenseGaussianSystem{Bool}, 0))
    pushforward(Σ * L, R)
end
