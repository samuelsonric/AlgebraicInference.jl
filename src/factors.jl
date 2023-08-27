struct Factor{T₁, T₂}
    variables::Vector{T₁}
    hom::T₂
end

function Base.convert(::Type{Factor{T₁, T₂}}, fac::Factor) where {T₁, T₂}
    Factor{T₁, T₂}(fac.variables, fac.hom)
end

function combine(fac₁::Factor, fac₂::Factor, obs::Dict)
    i, j₁, j₂ = combine(fac₁.variables, fac₂.variables)
    Factor(i, combine(fac₁.hom, fac₂.hom, [obs[v] for v in i], j₁, j₂))
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

function combine(i₁::Vector, i₂::Vector)
    i  = i₁ ∪ i₂
    j₁ = zeros(Int, length(i₁))
    j₂ = zeros(Int, length(i₂))

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

function project(fac::Factor, s::Vector, obs::Dict)
    i, j₁, j₂ = project(fac.variables, s)
    Factor(i, project(fac.hom, [obs[v] for v in fac.variables], j₁))
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

function project(i₁::Vector, i₂::Vector)
    i  = i₁ ∩ i₂
    j₁ = zeros(Int, length(i))
    j₂ = zeros(Int, length(i))

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

function Base.zero(::Type{Factor{T₁, GaussianSystem{T₂, T₃, T₄, T₅, T₆}}}) where {
    T₁, T₂, T₃, T₄, T₅, T₆}
    Factor(T₁[], zero(GaussianSystem{T₂, T₃, T₄, T₅, T₆}, 0))
end

function permute(fac::Factor, s::Vector, obs::Dict)
    i = permute(fac.variables, s)
    permute(fac.hom, [obs[v] for v in s[i]], i)
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

function permute(i₁::Vector, i₂::Vector)
    i = zeros(Int, length(i₁))
    
    for (x₁, y₁) in enumerate(i₁), (x₂, y₂) in enumerate(i₂)
        if y₁ == y₂
            i[x₁] = x₂
        end
    end

    i
end

function observe(fac::Factor, hom, var, obs::Dict)
    i, j, y = observe(fac.variables, var)
    Factor(i, observe(fac.hom, hom, [obs[v] for v in fac.variables], j, y))
end


function observe(
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

function observe(i₁::Vector, y₂)
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
