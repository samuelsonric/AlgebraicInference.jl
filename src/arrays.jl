###############################
# Valuation Algebra interface #
###############################


function combine(
    hom₁::Array,
    hom₂::Array,
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    n = length(obs)

    dims₁ = ones(Int, n)
    dims₂ = ones(Int, n)

    dims₁[i₁] .*= size(hom₁)
    dims₂[i₂] .*= size(hom₂)    

    hom₁ = reshape(hom₁, dims₁...)
    hom₂ = reshape(hom₂, dims₂...)

    hom₁ .* hom₂
end


function combine(
    hom₁::Array{<:Any, 0},
    hom₂::Array{<:Any, 0},
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    fill(hom₁ .* hom₂)
end


function invert(hom::Array{<:Real})
    pinv.(hom)
end


function invert(hom::Array{<:Real, 0})
    fill(pinv.(hom))
end


function invert(hom::Array{BoolRig})
    hom
end


function project(
    hom::Array,
    i₁::AbstractVector,
    i₂::AbstractVector,
    obs::AbstractVector)

    reshape(sum(hom, dims=i₂), obs[i₁]...)
end


function reduce_to_context(
    hom::Array,
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
    hom::Vector,
    ctx::Integer,
    i₁::AbstractVector,
    y₂::Integer,
    obs::AbstractVector)

    fill(hom[y₂])
end


function permute(
    hom::Array,
    i::AbstractVector,
    obs::AbstractVector)

    permutedims(hom, i)
end


function permute(
    hom::Array{<:Any, 0},
    i::AbstractVector,
    obs::AbstractVector)

    hom
end


function unit(::Type{Array{T}}) where T
    ones(T)
end


function ctxtype(::Type{Array{T}}) where T
    Int
end


function cpdtype(::Type{Array{T}}) where T
    Array{T}
end


function ctxcat(::Type{Array{T}}, ctx::AbstractVector) where T
    ctx
end
