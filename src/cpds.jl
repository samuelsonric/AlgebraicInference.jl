struct CPD{T₁, T₂}
    hom::T₁
    srcobs::Vector{T₂}
    tgtobs::Vector{T₂}
    srcvars::Vector{Int}
    tgtvars::Vector{Int}
end


function Factor{T₁, T₂}(cpd::CPD{<:GaussianConditional}) where {T₁ <: GaussianSystem, T₂}
    hom = GaussianSystem(cpd.hom)
    obs = [cpd.srcobs; cpd.tgtobs]
    vars = [cpd.srcvars; cpd.tgtvars]

    Factor{T₁, T₂}(hom, obs, vars)
end


function Base.convert(::Type{CPD{T₁, T₂}}, cpd::CPD) where {T₁, T₂}
    CPD{T₁, T₂}(cpd.hom, cpd.srcobs, cpd.tgtobs, cpd.srcvars, cpd.tgtvars)
end


function Distributions.rand!(rng::AbstractRNG, cpd::CPD, x::Vector{Vector{T}}) where T
    _x = reduce(vcat, x[cpd.srcvars]; init=T[])
    _y = rand(rng, cpd.hom, _x)

    for (v, o, c) in zip(cpd.tgtvars, cpd.tgtobs, cumsum(cpd.tgtobs))
        x[v] = _y[c - o + 1:c]
    end
end


function Distributions.rand!(cpd::CPD, x::Vector{Vector{T}}) where T
    rand(default_rng(), cpd, x)
end


function mean!(cpd::CPD, x::Vector{Vector{T}}) where T
    _x = reduce(vcat, x[cpd.srcvars]; init=T[])
    _y = mean(cpd.hom, _x)

    for (v, o, c) in zip(cpd.tgtvars, cpd.tgtobs, cumsum(cpd.tgtobs))
        x[v] = _y[c - o + 1:c]
    end
end


function cpdtype(::Type{<:GaussianSystem{Matrix{T}}}) where T
    GaussianConditional{DenseGaussianSampler{T}, Matrix{T}}
end


function combine(fac₁::Factor{T₁, T₂}, cpd₂::CPD) where {T₁, T₂}
    fac₂ = Factor{T₁, T₂}(cpd₂)
    combine(fac₁, fac₂)
end


function combine(cpd₁::CPD, fac₂::Factor{T₁, T₂}) where {T₁, T₂}
    fac₁ = Factor{T₁, T₂}(cpd₁)
    combine(fac₁, fac₂)
end


function disintegrate(fac::Factor{T₁, T₂}, vars::Vector{Int}) where {T₁, T₂}
    i₁ = Int[]
    i₂ = Int[]

    for (x, y) in enumerate(fac.vars)
        if y in vars
            push!(i₁, x)
        else
            push!(i₂, x)
        end
    end
    
    hom₁, hom₂ = disintegrate(fac.hom, i₁, i₂, fac.obs)
    srcobs = fac.obs[i₁]
    tgtobs = fac.obs[i₂]
    srcvars = fac.vars[i₁]
    tgtvars = fac.vars[i₂]

    fac = Factor{T₁, T₂}(hom₁, srcobs, srcvars)
    cpd = CPD{cpdtype(T₁), T₂}(hom₂, srcobs, tgtobs, srcvars, tgtvars)

    fac, cpd
end


function disintegrate(hom::GaussianSystem, i₁::Vector{Int}, i₂::Vector{Int}, obs::Vector{Int})
    cms = cumsum(obs)

    j₁ = Int[]
    j₂ = Int[] 

    for y₁ in i₁
        append!(j₁, cms[y₁] - obs[y₁] + 1:cms[y₁])
    end
 
    for y₂ in i₂
        append!(j₂, cms[y₂] - obs[y₂] + 1:cms[y₂])
    end

    disintegrate(hom, j₁, j₂)
end

