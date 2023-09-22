struct GaussianSampler{T₁, T₂, T₃, T₄} <: Sampleable{Multivariate, Continuous}
    chol::Cholesky{T₁, T₂}
    V::T₃
    μ::T₄
end


function GaussianSampler(Σ::GaussianSystem)
    K = KKT(Σ.P, Σ.S)

    μ = solve!(K, Σ.p, Σ.s)
    V = K.V₂
    chol = K.cache₂.cacheval

    GaussianSampler(chol, V, μ)
end


function GaussianSystem(Σ::GaussianSampler)
    P = Xt_A_X(AbstractMatrix(Σ.chol), Σ.V')
    S = I - Xt_A_X(I, Σ.V')
    p = P * Σ.μ
    s = S * Σ.μ
    σ = dot(s, Σ.μ)
    
    GaussianSystem(P, S, p, s, σ)
end


const DenseGaussianSampler{T} = GaussianSampler{
    T,
    Matrix{T},
    Matrix{T},
    Vector{T}}


function Base.:(==)(Σ₁::GaussianSampler, Σ₂::GaussianSampler)
    Σ₁.chol == Σ₂.chol &&
    Σ₁.V == Σ₂.V &&
    Σ₁.μ == Σ₂.μ 
end


function Base.convert(::Type{GaussianSampler{T₁, T₂, T₃, T₄}}, Σ::GaussianSampler) where {
    T₁, T₂, T₃, T₄}

    GaussianSampler{T₁, T₂, T₃, T₄}(Σ.chol, Σ.V, Σ.μ)
end


function Distributions.sampler(Σ::GaussianSystem)
    GaussianSampler(Σ)    
end


function Base.length(Σ::GaussianSampler)
    size(Σ.V, 1)
end


function Statistics.mean(Σ::GaussianSampler)
    Σ.μ
end


function Statistics.cov(Σ::GaussianSampler)
    C = inv(Σ.chol)
    Xt_A_X(C, Σ.V')
end


function Distributions.invcov(Σ::GaussianSampler)
    P = AbstractMatrix(Σ.chol)
    Xt_A_X(P, Σ.V')
end


function Statistics.var(Σ::GaussianSampler)
    diag(cov(Σ))
end


function Distributions._rand!(rng::AbstractRNG, Σ::GaussianSampler, x::AbstractVector)
    n = size(Σ.V, 2)
    mul!(x, Σ.V, Σ.chol.U \ randn(rng, n))
    x .+ Σ.μ
end


function disintegrate(Σ::GaussianSystem, i₁::AbstractVector, i₂::AbstractVector; atol::Real=1e-8)
    P₁₁, P₂₂, P₁₂, P₂₁ = blocks(Σ.P, i₁, i₂)
    S₁₁, S₂₂, S₁₂, S₂₁ = blocks(Σ.S, i₁, i₂)

    p₁, p₂ = blocks(Σ.p, i₁, i₂)
    s₁, s₂ = blocks(Σ.s, i₁, i₂)

    σ₁ = Σ.σ

    K = KKT(P₂₂, S₂₂; atol)
    A = solve!(K, P₂₁, S₂₁)
    a = solve!(K, p₂,  s₂)

    P = P₁₁ - P₁₂ * A - A' * P₂₁ + Xt_A_X(P₂₂, A)
    S = S₁₁ - S₁₂ * A
    p = p₁  - P₁₂ * a - A' * p₂  + A' * P₂₂ * a
    s = s₁  - S₁₂ * a
    σ = σ₁  - s₂' * a

    chol = K.cache₂.cacheval
    V = K.V₂
    μ = a

    Σ = GaussianSystem(P, S, p, s, σ)
    f = GaussianConditional(GaussianSampler(chol, V, μ), -A)

    Σ, f
end
