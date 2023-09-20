struct GaussianSampler{T₁, T₂, T₃} <: Sampleable{Multivariate, Continuous}
    U::T₁
    V::T₂
    μ::T₃
end


function GaussianSampler(Σ::GaussianSystem)
    K = KKT(Σ.P, Σ.S)

    μ = solve!(K, Σ.p, Σ.s)
    V = K.V₂
    U = K.cache₂.cacheval.U

    GaussianSampler(U, V, μ)
end


function GaussianSystem(Σ::GaussianSampler)
    P = Xt_A_X(Xt_A_X(I, Σ.U), Σ.V')
    S = I - Xt_A_X(I, Σ.V')
    p = P * Σ.μ
    s = S * Σ.μ
    σ = dot(s, Σ.μ)
    
    GaussianSystem(P, S, p, s, σ)
end


const DenseGaussianSampler{T} = GaussianSampler{
    UpperTriangular{T, Matrix{T}},
    Matrix{T},
    Vector{T}}


function Base.convert(::Type{GaussianSampler{T₁, T₂, T₃}}, Σ::GaussianSampler) where {
    T₁, T₂, T₃}

    GaussianSampler{T₁, T₂, T₃}(Σ.U, Σ.V, Σ.μ)
end


function Statistics.mean(Σ::GaussianSampler)
    Σ.μ
end


function Base.length(Σ::GaussianSampler)
    size(Σ.V, 1)
end


function Distributions._rand!(rng::AbstractRNG, Σ::GaussianSampler, x::AbstractVector)
    n = size(Σ.V, 2)
    mul!(x, Σ.V, Σ.U \ randn(rng, n))
end


function Distributions.sampler(Σ::GaussianSystem)
    GaussianSampler(Σ)    
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

    U = K.cache₂.cacheval.U
    V = K.V₂
    μ = a

    Σ = GaussianSystem(P, S, p, s, σ)
    f = GaussianConditional(GaussianSampler(U, V, μ), -A)

    Σ, f
end
