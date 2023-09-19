struct GaussianConditional{T₁, T₂}
    Σ::T₁
    M::T₂
end


function GaussianSystem(f::GaussianConditional)
    f.Σ * [-f.M I]
end


function Base.convert(::Type{GaussianConditional{T₁, T₂}}, f::GaussianConditional) where {T₁, T₂}
    GaussianConditional{T₁, T₂}(f.Σ, f.M)
end


function Base.rand(rng::AbstractRNG, f::GaussianConditional, x::AbstractVector)
    rand(rng, f.Σ) + f.M * x
end


function Base.rand(f::GaussianConditional, x::AbstractVector)
    rand(default_rng(), f, x)
end


function Statistics.mean(f::GaussianConditional, x::AbstractVector)
    mean(f.Σ) + f.M * x
end
