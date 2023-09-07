# Let F: CospanΛ → C be a symmetric monoidal functor, and let y be an object in CospanΛ. A
# factor is a pair (i, ψ), where
# i: x ↪ y
# is an injection and
# ψ: I → F(x)
# is a morphism in C.
struct Factor{T₁, T₂}
    hom::T₁
    obs::Vector{T₂}
    vars::Vector{Int}

    function Factor{T₁, T₂}(hom, obs, vars) where {T₁, T₂}
        @assert length(obs) == length(vars)

        new{T₁, T₂}(hom, obs, vars)
    end
end


function Factor(hom::T₁, obs::Vector{T₂}, vars::Vector{Int}) where {T₁, T₂}
    Factor{T₁, T₂}(hom, obs, vars)
end


function Base.convert(::Type{Factor{T₁, T₂}}, fac::Factor) where {T₁, T₂}
    Factor{T₁, T₂}(fac.hom, fac.obs, fac.vars)
end


function Base.length(fac::Factor)
    length(fac.vars)
end


# Compute the combination
# fac₁ ⊗ fac₂
function combine(fac₁::Factor{T₁, T₂}, fac₂::Factor) where {T₁, T₂}
    vars  = fac₁.vars ∪ fac₂.vars

    i₁ = Vector{Int}(undef, length(fac₁))
    i₂ = Vector{Int}(undef, length(fac₂))

    for (x, y) in enumerate(vars)
        for (x₁, y₁) in enumerate(fac₁.vars)
            if y == y₁
                i₁[x₁] = x
            end
        end
        
         for (x₂, y₂) in enumerate(fac₂.vars)
            if y == y₂
                i₂[x₂] = x
            end
        end
    end

    obs = Vector{T₂}(undef, length(vars))
    obs[i₁] = fac₁.obs
    obs[i₂] = fac₂.obs

    hom = combine(fac₁.hom, fac₂.hom, i₁, i₂, obs)

    Factor{T₁, T₂}(hom, obs, vars)
end


# Compute the composite
# hom₁ ⊗ hom₂ ; F([i₁, i₂])
function combine(
    hom₁::GaussianSystem,
    hom₂::GaussianSystem,
    i₁::Vector{Int},
    i₂::Vector{Int},
    obs::Vector{Int})
    
    cms = cumsum(obs)
 
    j₁ = Int[]
    j₂ = Int[]
 
    for y₁ in i₁
        append!(j₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end
  
    for y₂ in i₂
        append!(j₂, cms[y₂] - obs[y₂] + 1:cms[y₂])
    end

    n = sum(obs)

    combine(hom₁, hom₂, j₁, j₂, n)
end


# Compute the projection
# fac₁ ↓ vars₂
function project(fac₁::Factor{T₁, T₂}, vars₂::Vector{Int}) where {T₁, T₂}
    i₁ = Int[]
    i₂ = Int[]

    for (x₁, y₁) in enumerate(fac₁.vars)
        if y₁ in vars₂
            push!(i₁, x₁)
        else
            push!(i₂, x₁)
        end
    end
    
    hom = project(fac₁.hom, i₁, i₂, fac₁.obs)
    obs = fac₁.obs[i₁]
    vars = fac₁.vars[i₁]

    Factor{T₁, T₂}(hom, obs, vars)
end


# Compute the composite
# hom ; F(i₁†)
function project(hom::GaussianSystem, i₁::Vector{Int}, i₂::Vector{Int}, obs::Vector{Int})
    cms = cumsum(obs)

    j₁ = Int[]
    j₂ = Int[] 

    for y₁ in i₁
        append!(j₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end
 
    for y₂ in i₂
        append!(j₂, cms[y₂] - obs[y₂] + 1:cms[y₂])
    end

    first(disintegrate(hom, j₁, j₂))
end


# Construct an identity element
# e
# of type Factor{T₁, T₂}
function Base.zero(::Type{Factor{T₁, T₂}}) where {T₁ <: GaussianSystem, T₂}
    Factor{T₁, T₂}(zero(T₁, 0), [], [])
end


function permute(fac₁::Factor, vars₂::Vector{Int})
    i = Vector{Int}(undef, length(fac₁))
    
    for (x₁, y₁) in enumerate(fac₁.vars)
        for (x₂, y₂) in enumerate(vars₂)
            if y₁ == y₂
                i[x₂] = x₁
            end
        end
    end

    permute(fac₁.hom, i, fac₁.obs)
end


# Compute the composite
# hom ; F(i)
function permute(hom::GaussianSystem, i::Vector{Int}, obs::Vector{Int})
    cms = cumsum(obs)

    j = Int[]

    for y in i
        append!(j, cms[y] - obs[y] + 1:cms[y])
    end

    permute(hom, j) 
end


function observe(fac₁::Factor{T₁, T₂}, hom₂, var₂::Int) where {T₁, T₂}
    i₁ = Int[]
    x₂ = -1

    for (x₁, y₁) in enumerate(fac₁.vars)
        if y₁ == var₂
            x₂ = x₁
        else
            push!(i₁, x₁)
        end
    end

    hom = observe(fac₁.hom, hom₂, i₁, x₂, fac₁.obs)
    obs = fac₁.obs[i₁]
    vars = fac₁.vars[i₁]

    Factor{T₁, T₂}(hom, obs, vars)
end


function observe(
    hom₁::GaussianSystem,
    hom₂::AbstractVector,
    i₁::Vector{Int},
    y₂::Int,
    obs::Vector{Int})

    cms = cumsum(obs)

    j₁ = Int[]
    j₂ = cms[y₂] - obs[y₂] + 1:cms[y₂]

    for y₁ in i₁
        append!(j₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end

    observe(hom₁, hom₂, j₁, j₂)
end
