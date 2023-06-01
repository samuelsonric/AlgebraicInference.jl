struct GaussianSystem{
    T₁ <: AbstractMatrix,
    T₂ <: AbstractMatrix,
    T₃ <: AbstractVector,
    T₄ <: AbstractVector}

    P::T₁
    S::T₂
    p::T₃
    s::T₄

    function GaussianSystem(P::T₁, S::T₂, p::T₃, s::T₄) where {
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix,
        T₃ <: AbstractVector,
        T₄ <: AbstractVector}
    
        m = checksquare(P)
        n = checksquare(S)
        @assert m == n == length(p) == length(s)
        new{T₁, T₂, T₃, T₄}(P, S, p, s)
    end
end

"""
    normal(Σ::AbstractMatrix, μ::AbstractVector)

Construct a multivariate normal distribution with covariance matrix `Σ` and mean vector `μ`.
"""
function normal(Σ::AbstractMatrix, μ::AbstractVector)
    V = nullspace(Σ)
    P = pinv(Σ)
    S = V * V'
    GaussianSystem(P, S, -P * μ, -S * μ)
end

"""
    normal(Σ::AbstractMatrix)

Construct a centered multivariate normal distribution with covariance matrix `Σ`.
"""
function normal(Σ::AbstractMatrix)
    n = size(Σ, 1)
    normal(Σ, zeros(n))
end

"""
    normal(μ::AbstractVector)

Construct a Dirac distribution centered at `μ`.
"""
function normal(μ::AbstractVector)
    n = length(μ)
    normal(zeros(n, n), μ)
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
    kernel(Σ::AbstractMatrix, L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\mathcal{N}(Lx, \\Sigma).
```
"""
function kernel(Σ::AbstractMatrix, L::AbstractMatrix)
    n = size(Σ, 1)
    kernel(Σ, zeros(n), L)
end

"""
    kernel(L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\delta_{Lx}.
```
"""
function kernel(L::AbstractMatrix)
    n = size(L, 2)
    kernel(zeros(n, n), L)
end

"""
    length(Σ::GaussianSystem)

Get the dimension of `Σ`.
"""
function length(Σ::GaussianSystem)
    size(Σ.P, 1)
end

function *(Σ::GaussianSystem, M::AbstractMatrix)
    @assert size(M, 1) == length(Σ)
    GaussianSystem(
        M' * Σ.P * M,
        M' * Σ.S * M,
        M' * Σ.p,
        M' * Σ.s)
end

function +(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    @assert length(Σ₁) == length(Σ₂)    
    GaussianSystem(
        Σ₁.P + Σ₂.P,
        Σ₁.S + Σ₂.S,
        Σ₁.p + Σ₂.p,
        Σ₁.s + Σ₂.s)
end

function zero(Σ::GaussianSystem)
    GaussianSystem(
        zero(Σ.P),
        zero(Σ.S),
        zero(Σ.p),
        zero(Σ.s))
end

"""
    mean(Σ::GaussianSystem)

Get the mean vector of `Σ`.
"""
function mean(Σ::GaussianSystem)
    C = cov(Σ)
    (C * Σ.P - I) * pinv(Σ.S) * Σ.s - C * Σ.p
end

"""
    cov(Σ::GaussianSystem)

Get the covariance matrix of `Σ`.
"""
function cov(Σ::GaussianSystem)
    V = nullspace(Σ.S; atol=1e-10)
    V * pinv(V' * Σ.P * V) * V'
end
