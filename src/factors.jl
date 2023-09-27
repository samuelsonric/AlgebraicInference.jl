# A valuation in a valuation algebra.
#
# To implement the valuation algebra interface, specialize the methods
# - combine
# - project
# - permute
# - unit
# With these, you can use the Shenoy-Shafer and idempotent architectures.
#
# In order to use the Lauritzen-Spiegelhalter or HUGIN architectures, specialise the method
# - inverse
# 
# In order to supply evidence, specialise the methods
# - ctxtype
# - reduce_to_context
#
# In order to sample, specialise the methods
# - disintegrate
# - cpdtype
# - cpdrand
# - ctxtype
# - ctxcat
struct Factor{T₁, T₂, T₃}
    hom::T₂
    obs::Vector{T₃}
    vars::Vector{Int}

    function Factor{T₁, T₂, T₃}(hom, obs, vars) where {T₁, T₂, T₃}
        @assert length(obs) == length(vars)
        new{T₁, T₂, T₃}(hom, obs, vars)
    end
end


function Factor{T₁}(hom::T₂, obs::AbstractVector{T₃}, vars) where {T₁, T₂, T₃}
    Factor{T₁, T₂, T₃}(hom, obs, vars)
end


function Base.convert(::Type{Factor{T₁, T₂, T₃}}, fac::Factor{T₁}) where {T₁, T₂, T₃}
    Factor{T₁, T₂, T₃}(fac.hom, fac.obs, fac.vars)
end


# Get the number of variables in the domain
# d(fac)
function Base.length(fac::Factor)
    length(fac.vars)
end


# Compute the reduction
# (fac ⊗ ctx) ↓ d(fac)
function reduce_to_context(fac::Factor{false, T₁, T₂}, ctx::Pair) where {T₁, T₂}
    i₁ = Int[]
    x₂ = -1

    for (x, y) in enumerate(fac.vars)
        if y != ctx.first
            push!(i₁, x)
        else
            x₂ = x
        end
    end

    hom = reduce_to_context(fac.hom, ctx.second, i₁, x₂, fac.obs)
    obs = fac.obs[i₁]
    vars = fac.vars[i₁]

    Factor{false, T₁, T₂}(hom, obs, vars)
end


# Compute an equivalent factor with sorted variables.
function Base.sort(fac::Factor{false, T₁, T₂}) where {T₁, T₂}
    i = sortperm(fac.vars)
 
    hom = permute(fac.hom, i, fac.obs)
    obs = fac.obs[i]
    vars = fac.vars[i]

    Factor{true, T₁, T₂}(hom, obs, vars)
end


# Compute the inverse
# fac⁻¹
function invert(fac::Factor{true, T₁, T₂}) where {T₁, T₂}
    hom = invert(fac.hom)
    obs = fac.obs
    vars = fac.vars

    Factor{true, T₁, T₂}(hom, obs, vars)
end


# Compute the combination
# fac₁ ⊗ fac₂
function combine(fac₁::Factor{true, T₁, T₂}, fac₂::Factor) where {T₁, T₂}
    vars  = sort(fac₁.vars ∪ fac₂.vars)

    i₁ = Vector{Int}(undef, length(fac₁))
    i₂ = Vector{Int}(undef, length(fac₂))

    for (x₁, y₁) in enumerate(fac₁.vars)
        i₁[x₁] = searchsortedfirst(vars, y₁)
    end

    for (x₂, y₂) in enumerate(fac₂.vars)
        i₂[x₂] = searchsortedfirst(vars, y₂)
    end

    obs = Vector{T₂}(undef, length(vars))
    obs[i₁] = fac₁.obs
    obs[i₂] = fac₂.obs

    hom = combine(fac₁.hom, fac₂.hom, i₁, i₂, obs)

    Factor{true, T₁, T₂}(hom, obs, vars)
end


# Compute the projection
# fac ↓ vars
function project(fac::Factor{true, T₁, T₂}, vars::AbstractVector) where {T₁, T₂}
    i₁ = Int[]
    i₂ = Int[]

    for (x, y) in enumerate(fac.vars)
        if insorted(y, vars)
            push!(i₁, x)
        else
            push!(i₂, x)
        end
    end
    
    hom = project(fac.hom, i₁, i₂, fac.obs)
    obs = fac.obs[i₁]
    vars = fac.vars[i₁]

    Factor{true, T₁, T₂}(hom, obs, vars)
end



# Construct a morphism by ordering the variables in fac.
function permute(fac::Factor{true}, vars::AbstractVector)
    i = Vector{Int}(undef, length(fac))

    for (x₂, y₂) in enumerate(vars)
        i[x₂] = searchsortedfirst(fac.vars, y₂)
    end

    permute(fac.hom, i, fac.obs)
end


# Construct an identity element
# e
# of type Factor{T₁, T₂}
function unit(::Type{Factor{true, T₁, T₂}}) where {T₁, T₂}
    Factor{true, T₁, T₂}(unit(T₁), [], [])
end


function cpdtype(::Type)
    Union{}
end


function ctxtype(::Type)
    Union{}
end
