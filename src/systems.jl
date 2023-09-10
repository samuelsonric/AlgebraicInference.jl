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


function GaussianSystem(d::MvNormalCanon)
    CanonicalForm(d.J, d.h)
end


function GaussianSystem(d::NormalCanon)
    CanonicalForm([d.λ;;], [d.η])
end


function GaussianSystem(d::MvNormal)
    normal(d.μ, d.Σ)
end 


function GaussianSystem(d::Normal)
    normal(d.μ, d.σ)
end 


function GaussianSystem(cpd::LinearGaussianCPD)
    kernel(cpd.a, cpd.b, cpd.σ)
end 


function GaussianSystem(cpd::StaticCPD)
    GaussianSystem(cpd.d)
end


function GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Σ) where {
    T₁, T₂, T₃, T₄, T₅}

    convert(GaussianSystem{T₁, T₂, T₃, T₄, T₅}, GaussianSystem(Σ))
end


"""
    CanonicalForm{T₁, T₂}(K, h) where {T₁ <: AbstractMatrix, T₂ <: AbstractVector}

Construct the canonical form ``\\mathcal{C}(K, h, g)``, where the normalization constant
``g`` is inferred from ``K`` and ``h``.
"""
function CanonicalForm{T₁, T₂}(K, h) where {T₁ <: AbstractMatrix, T₂ <: AbstractVector}
    n = size(K, 1)
    CanonicalForm{T₁, T₂}(K, Zeros(K), h, Zeros(h), 0)
end


"""
    CanonicalForm(K::AbstractMatrix, h::AbstractVector)

Construct the canonical form ``\\mathcal{C}(K, h, g)``, where the normalization constant
``g`` is inferred from ``K`` and ``h``.
"""
function CanonicalForm(K::T₁, h::T₂) where {T₁ <: AbstractMatrix, T₂ <: AbstractVector}
    CanonicalForm{T₁, T₂}(K, h)
end


function Base.convert(::Type{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}, Σ::GaussianSystem) where {
    T₁, T₂, T₃, T₄, T₅}

    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Σ.P, Σ.S, Σ.p, Σ.s, Σ.σ)
end


function Base.convert(::Type{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}, Σ) where {
    T₁, T₂, T₃, T₄, T₅}

    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Σ)
end


function Base.convert(::Type{CanonicalForm{T₁, T₂}}, Σ::GaussianSystem) where {T₁, T₂}
    @assert iszero(Σ.S)
    @assert iszero(Σ.s)
    @assert iszero(Σ.σ)

    n = length(Σ)
    CanonicalForm{T₁, T₂}(Σ.P, Zeros(n, n), Σ.p, Zeros(n), 0)
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
    normal(μ, Σ) *  [-L I]
end


"""
    kernel(l::AbstractVector, μ::Real, σ::Real)

Construct a conditional distribution of the form
``(y \\mid x) \\sim \\mathcal{N}(l^\\mathsf{T}x + \\mu, \\sigma^2).``
"""
function kernel(l::AbstractVector, μ::Real, σ::Real)
    normal(μ, σ) * [-l' I]
end


"""
    length(Σ::GaussianSystem)

Get the dimension of `Σ`.
"""
function Base.length(Σ::GaussianSystem)
    size(Σ.P, 1)
end


"""
    cov(Σ::GaussianSystem; atol::Real=1e-8)

Get the covariance matrix of `Σ`.
"""
function Statistics.cov(Σ::GaussianSystem; atol::Real=1e-8)
    U = nullspace(Σ.S; atol)
    Xt_A_X(pinv(Xt_A_X(Σ.P, U); atol), U')
end


"""
    var(Σ::GaussianSystem; atol::Real=1e-8)

Get the variances of `Σ`.
"""
function Statistics.var(Σ::GaussianSystem; atol::Real=1e-8)
    diag(cov(Σ; atol))
end


"""
    mean(Σ::GaussianSystem; atol::Real=1e-8)

Get the mean vector of `Σ`.
"""
function Statistics.mean(Σ::GaussianSystem; atol::Real=1e-8)
    n = length(Σ)
    K = KKT(Σ.P, Σ.S; atol)
    solve!(K, Σ.p, Σ.s)
end


"""
    invcov(Σ::GaussianSystem)

Get the precision matrix of `Σ`.
"""
function Distributions.invcov(Σ::GaussianSystem)
    Σ.P
end


# Compute the tensor product of Σ₁ and Σ₂.
function Catlab.:⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    GaussianSystem(
        cat(Σ₁.P, Σ₂.P; dims=(1, 2)),
        cat(Σ₁.S, Σ₂.S; dims=(1, 2)),
        cat(Σ₁.p, Σ₂.p; dims=(1,)),
        cat(Σ₁.s, Σ₂.s; dims=(1,)),
        Σ₁.σ + Σ₂.σ)
end


# Construct a system with energy function
# E'(x) = E(Mx),
# where E is the energy function of Σ.
function Base.:*(Σ::GaussianSystem, M::AbstractMatrix)
    @assert size(M, 1) == length(Σ)
    GaussianSystem(
        Xt_A_X(Σ.P, M),
        Xt_A_X(Σ.S, M),
        M' * Σ.p,
        M' * Σ.s,
        Σ.σ)
end


# Construct a system with energy function
# E'(x) = E₁(x) + E₂(x),
# where E₁ and E₂ are the energy functions of Σ₁ and Σ₂.
function Base.:+(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    @assert length(Σ₁) == length(Σ₂)    
    GaussianSystem(
        Σ₁.P + Σ₂.P,
        Σ₁.S + Σ₂.S,
        Σ₁.p + Σ₂.p,
        Σ₁.s + Σ₂.s,
        Σ₁.σ + Σ₂.σ)
end


# Construct a vacuous Gaussian system.
function Base.zero(Σ::GaussianSystem)
    GaussianSystem(zero(Σ.P), zero(Σ.S), zero(Σ.p), zero(Σ.s), zero(Σ.σ))
end


# Construct a vacuous Gaussian system.
function Base.zero(::Type{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}, n::Integer) where {
    T₁, T₂, T₃, T₄, T₅}

    @assert n >= 0
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(Zeros(n, n), Zeros(n, n), Zeros(n), Zeros(n), 0)
end


# Compute the pushforward M#Σ
function pushforward(Σ::GaussianSystem, M::AbstractMatrix; atol::Real=1e-8)
    @assert length(Σ) == size(M, 2)

    K = KKT(Σ.P, Σ.S + M' * M; atol)
    A = solve!(K, Zeros(M)', M')
    a = solve!(K, Σ.p, Σ.s)

    GaussianSystem(
        Xt_A_X(Σ.P, A),
        I - M * A,
        A' * (Σ.p - Σ.P * a),
        M * a,
        Σ.σ - Σ.s' * a)
end


function disintegrate(Σ::GaussianSystem, i₁::AbstractVector, i₂::AbstractVector; atol::Real=1e-8)
    P₁₁ = Σ.P[i₁, i₁]; P₁₂ = Σ.P[i₁, i₂]; P₂₂ = Σ.P[i₂, i₂]
    S₁₁ = Σ.S[i₁, i₁]; S₁₂ = Σ.S[i₁, i₂]; S₂₂ = Σ.S[i₂, i₂]

    P₂₁ = P₁₂'
    S₂₁ = S₁₂'

    p₁ = Σ.p[i₁]; p₂ = Σ.p[i₂]
    s₁ = Σ.s[i₁]; s₂ = Σ.s[i₂]

    σ₁ = σ₂ = Σ.σ

    K = KKT(P₂₂, S₂₂; atol)
    A = solve!(K, P₂₁, S₂₁)
    a = solve!(K, p₂,  s₂)

    Σ₁ = GaussianSystem(
        P₁₁ - P₁₂ * A - A' * P₂₁ + Xt_A_X(P₂₂, A),
        S₁₁ - S₁₂ * A,
        p₁  - P₁₂ * a - A' * p₂  + A' * P₂₂ * a,
        s₁  - S₁₂ * a,
        σ₁  - s₂' * a)

    Σ₂ = GaussianSystem(P₂₂, S₂₂, p₂, s₂, σ₂)

    Σ₁, Σ₂, -A, a
end


function combine(
    Σ₁::GaussianSystem,
    Σ₂::GaussianSystem,
    i₁::AbstractVector,
    i₂::AbstractVector,
    n::Integer)

    P = zeros(n, n)
    S = zeros(n, n)
    p = zeros(n)
    s = zeros(n)

    P[i₁, i₁] .+= Σ₁.P
    S[i₁, i₁] .+= Σ₁.S
    p[i₁] .+= Σ₁.p
    s[i₁] .+= Σ₁.s

    P[i₂, i₂] .+= Σ₂.P
    S[i₂, i₂] .+= Σ₂.S
    p[i₂] .+= Σ₂.p
    s[i₂] .+= Σ₂.s

    σ = Σ₁.σ + Σ₂.σ

    GaussianSystem(P, S, p, s, σ)
end


function permute(Σ::GaussianSystem, i::AbstractVector)
    P = Σ.P[i, i]
    S = Σ.S[i, i]
    p = Σ.p[i]
    s = Σ.s[i]
    σ = Σ.σ

    GaussianSystem(P, S, p, s, σ)
end


function observe(
    Σ::GaussianSystem,
    v::AbstractVector,
    i₁::AbstractVector,
    i₂::AbstractVector)

    P = Σ.P[i₁, i₁]
    S = Σ.S[i₁, i₁]
    p = Σ.p[i₁] - Σ.P[i₁, i₂] * v
    s = Σ.s[i₁] - Σ.S[i₁, i₂] * v
    σ = Σ.σ + dot(v, Σ.S[i₂, i₂] * v - 2Σ.s[i₂])

    GaussianSystem(P, S, p, s, σ)
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
