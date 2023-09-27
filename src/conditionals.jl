# The conditional distribution of a Gaussian system.
struct GaussianConditional{T₁, T₂}
    Σ::T₁
    M::T₂
end


function Base.convert(::Type{GaussianConditional{T₁, T₂}}, f::GaussianConditional) where {T₁, T₂}
    GaussianConditional{T₁, T₂}(f.Σ, f.M)
end


###############################
# Valuation Algebra inferface #
###############################


function cpdrand(
    rng::AbstractRNG,
    hom::GaussianConditional,
    srcobs::AbstractVector,
    tgtobs::AbstractVector,
    x::AbstractVector{Vector{T}}) where T

    _x = reduce(vcat, x; init=T[])
    _y = rand(rng, hom.Σ) + hom.M * _x

    tgtcms = cumsum(tgtobs)

    n = length(tgtobs)
    y = Vector{Vector{T}}(undef, n)

    for i in 1:n
        y[i] = _y[tgtcms[i] - tgtobs[i] + 1:tgtcms[i]]
    end

    y    
end


function cpdmean(
    hom::GaussianConditional,
    srcobs::AbstractVector,
    tgtobs::AbstractVector,
    x::AbstractVector{Vector{T}}) where T

    _x = reduce(vcat, x; init=T[])
    _y = mean(hom.Σ) + hom.M * _x

    tgtcms = cumsum(tgtobs)

    n = length(tgtobs)
    y = Vector{Vector{T}}(undef, n)

    for i in 1:n
        y[i] = _y[tgtcms[i] - tgtobs[i] + 1:tgtcms[i]]
    end

    y    
end


function cpdtype(::Type{<:GaussianSystem{<:AbstractMatrix{T}}}) where T
    GaussianConditional{DenseGaussianSampler{T}, Matrix{T}}
end
