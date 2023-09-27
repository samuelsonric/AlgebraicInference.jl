###############################
# Valuation Algebra interface #
###############################


function combine(
    hom₁::AbstractArray,
    hom₂::AbstractArray,
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    n = length(obs)

    dims₁ = ones(Int, n)
    dims₂ = ones(Int, n)

    dims₁[i₁] .*= obs[i₁]
    dims₂[i₂] .*= obs[i₂]    

    hom₁ = reshape(hom₁, dims₁...)
    hom₂ = reshape(hom₂, dims₂...)

    hom₁ .* hom₂
end


function combine(
    hom₁::AbstractArray{<:Any, 0},
    hom₂::AbstractArray{<:Any, 0},
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    fill(hom₁ .* hom₂)
end


function invert(hom::AbstractArray{<:Real})
    pinv.(hom)
end


function invert(hom::AbstractArray{<:Real, 0})
    fill(pinv.(hom))
end


function invert(hom::AbstractArray{BoolRig})
    hom
end


function project(
    hom::AbstractArray,
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    reshape(sum(hom, dims=i₂), obs[i₁]...)
end


function reduce_to_context(
    hom::AbstractArray,
    ctx::Integer,
    i₁::AbstractVector,
    y₂::Integer,
    obs::AbstractVector)

    n = length(obs)

    i = Vector{Union{Colon, Int}}(undef, n)
    i[i₁] .= Colon()
    i[y₂]  = ctx

    hom[i...]
end


function reduce_to_context(
    hom::AbstractVector,
    ctx::Integer,
    i₁::AbstractVector,
    y₂::Integer,
    obs::AbstractVector)

    fill(hom[y₂])
end


function permute(
    hom::AbstractArray,
    i::AbstractVector,
    obs::AbstractVector)

    permutedims(hom, i)
end


function permute(
    hom::AbstractArray{<:Any, 0},
    i::AbstractVector,
    obs::AbstractVector)

    hom
end


function unit(::Type{T}) where T <: AbstractArray
    convert(T, ones())
end


function ctxtype(::Type{<:AbstractArray})
    Int
end
