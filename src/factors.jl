struct Factor{T}
    hom::T
    variables::Vector{Int}
end

function Base.convert(::Type{Factor{T}}, fac::Factor) where T
    Factor{T}(fac.hom, fac.variables)
end

function combine(fac₁::Factor{T}, fac₂::Factor{T}, obs::Vector) where T
    i, j₁, j₂ = combine(fac₁.variables, fac₂.variables)
    Factor{T}(combine(fac₁.hom, fac₂.hom, obs[i], j₁, j₂), i)
end

function combine(
    hom₁::GaussianSystem{T₁, T₂, T₃, T₄, T₅},
    hom₂::GaussianSystem{T₁, T₂, T₃, T₄, T₅},
    obs::Vector{Int},
    i₁::Vector{Int},
    i₂::Vector{Int}) where {T₁, T₂, T₃, T₄, T₅}
    
    cms = cumsum(obs)
 
    is₁ = Int[]
    is₂ = Int[]
 
    for y₁ in i₁
        append!(is₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end
  
    for y₂ in i₂
        append!(is₂, cms[y₂] - obs[y₂] + 1:cms[y₂])
    end

    n = sum(obs)

    P = convert(T₁, Zeros(n, n))
    S = convert(T₂, Zeros(n, n))
    p = convert(T₃, Zeros(n))
    s = convert(T₄, Zeros(n))

    P[is₁, is₁] .+= hom₁.P
    S[is₁, is₁] .+= hom₁.S
    p[is₁] .+= hom₁.p
    s[is₁] .+= hom₁.s

    P[is₂, is₂] .+= hom₂.P
    S[is₂, is₂] .+= hom₂.S
    p[is₂] .+= hom₂.p
    s[is₂] .+= hom₂.s

    σ = hom₁.σ + hom₂.σ

    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ)
end

function combine(i₁::Vector{Int}, i₂::Vector{Int})
    i  = i₁ ∪ i₂
    j₁ = copy(i₁)
    j₂ = copy(i₂)

    for (x, y) in enumerate(i)
        for (x₁, y₁) in enumerate(i₁)
            if y == y₁
                j₁[x₁] = x
            end
        end
        
         for (x₂, y₂) in enumerate(i₂)
            if y == y₂
                j₂[x₂] = x
            end
        end
    end
    
    i, j₁, j₂
end

function project(fac::Factor{T}, s::Vector{Int}, obs::Vector) where T
    i, j₁, j₂ = project(fac.variables, s)
    Factor{T}(project(fac.hom, obs[fac.variables], j₁), i)
end

function project(
    hom::GaussianSystem{T₁, T₂, T₃, T₄, T₅},
    obs::Vector{Int},
    i::Vector{Int}) where {T₁, T₂, T₃, T₄, T₅}

    cms = cumsum(obs)

    is₁ = Int[]
    is₂ = Int[] 

    for y₁ in i
        append!(is₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end
 
    for y₂ in setdiff(eachindex(obs), i)
        append!(is₂, cms[y₂] - obs[y₂] + 1:cms[y₂])
    end
    
    P₁₁ = hom.P[is₁, is₁]; P₁₂ = hom.P[is₁, is₂]
    S₁₁ = hom.S[is₁, is₁]; S₁₂ = hom.S[is₁, is₂]
    P₂₁ = hom.P[is₂, is₁]; P₂₂ = hom.P[is₂, is₂]
    S₂₁ = hom.S[is₂, is₁]; S₂₂ = hom.S[is₂, is₂]

    p₁ = hom.p[is₁]; p₂ = hom.p[is₂]
    s₁ = hom.s[is₁]; s₂ = hom.s[is₂]

    σ₁ = hom.σ

    K = KKT(P₂₂, S₂₂)
    A = solve!(K, P₂₁, S₂₁)
    a = solve!(K, p₂,  s₂)

    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(
        P₁₁ - P₁₂ * A - A' * P₂₁ + A'  * P₂₂ * A,
        S₁₁ - S₁₂ * A,
        p₁  - P₁₂ * a - A' * p₂ + A'  * P₂₂ * a,
        s₁  - S₁₂ * a,
        σ₁  - s₂' * a)
end

function project(i₁::Vector{Int}, i₂::Vector{Int})
    i  = i₁ ∩ i₂
    j₁ = copy(i)
    j₂ = copy(i)

    for (x, y) in enumerate(i)
        for (x₁, y₁) in enumerate(i₁)
            if y == y₁
                j₁[x] = x₁
            end
        end
        
         for (x₂, y₂) in enumerate(i₂)
            if y == y₂
                j₂[x] = x₂
            end
        end
    end
    
    i, j₁, j₂
end

function Base.zero(::Type{Factor{GaussianSystem{T₁, T₂, T₃, T₄, T₅}}}) where {T₁, T₂, T₃, T₄, T₅}
    Factor(zero(GaussianSystem{T₁, T₂, T₃, T₄, T₅}, 0), Int[])
end

function permute(fac::Factor, s::Vector{T}, obs::Vector) where T
    i = permute(fac.variables, s)
    permute(fac.hom, obs[i], i)
end

function permute(
    hom::GaussianSystem{T₁, T₂, T₃, T₄, T₅},
    obs::Vector{Int},
    i::Vector{Int}) where {T₁, T₂, T₃, T₄, T₅}

    cms = cumsum(obs)

    is = Int[]

    for y in i
        append!(is, cms[y] - obs[y] + 1:cms[y])
    end

    n = length(hom)
    
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(
        hom.P[is, is],
        hom.S[is, is],
        hom.p[is],
        hom.s[is],
        hom.σ)
end

function permute(i₁::Vector{Int}, i₂::Vector{Int})
    i = copy(i₁)
    
    for (x₁, y₁) in enumerate(i₁), (x₂, y₂) in enumerate(i₂)
        if y₁ == y₂
            i[x₁] = x₂
        end
    end

    i
end

function context(fac::Factor{T}, hom, var::Int, obs::Vector) where T
    i, j, y = context(fac.variables, var)
    Factor{T}(context(fac.hom, hom, obs[fac.variables], j, y), i)
end


function context(
    hom₁::GaussianSystem{T₁, T₂, T₃, T₄, T₅},
    hom₂::AbstractVector,
    obs::Vector{Int},
    i::Vector{Int},
    y::Int) where {T₁, T₂, T₃, T₄, T₅}

    cms = cumsum(obs)

    is₁ = Int[]
    is₂ = cms[y] - obs[y] + 1:cms[y]

    for y in i
        append!(is₁, cms[y] - obs[y] + 1:cms[y])
    end
 
    GaussianSystem{T₁, T₂, T₃, T₄, T₅}(
        hom₁.P[is₁, is₁],
        hom₁.S[is₁, is₁],
        hom₁.p[is₁] - hom₁.P[is₁, is₂] * hom₂,
        hom₁.s[is₁] - hom₁.S[is₁, is₂] * hom₂,
        hom₁.σ + dot(hom₂, hom₁.S[is₂, is₂] * hom₂ - 2 * hom₁.s[is₂]))
end

function context(i₁::Vector{Int}, y₂::Int)
    i = Int[]
    j = Int[]
    x = nothing

    for (x₁, y₁) in enumerate(i₁)
        if y₁ == y₂
            x = x₁
        else
            push!(i, y₁)
            push!(j, x₁)
        end
    end

    i, j, x::Int
end
