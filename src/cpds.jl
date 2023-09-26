# A valuation whose variables are partitioned into a domain and a codomain.
struct CPD{T₁, T₂}
    hom::T₁
    srcobs::Vector{T₂}
    tgtobs::Vector{T₂}
    srcvars::Vector{Int}
    tgtvars::Vector{Int}
end


function Base.convert(::Type{CPD{T₁, T₂}}, cpd::CPD) where {T₁, T₂}
    CPD{T₁, T₂}(cpd.hom, cpd.srcobs, cpd.tgtobs, cpd.srcvars, cpd.tgtvars)
end


function disintegrate(fac::Factor{true, T₁, T₂}, vars::AbstractVector) where {T₁, T₂}
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

    fac = Factor{true, T₁, T₂}(hom₁, srcobs, srcvars)
    cpd = CPD{cpdtype(T₁), T₂}(hom₂, srcobs, tgtobs, srcvars, tgtvars)

    fac, cpd
end


function cpdrand!(rng::AbstractRNG, cpd::CPD, x::Vector)
    x[cpd.tgtvars] = cpdrand(rng, cpd.hom, cpd.srcobs, cpd.tgtobs, x[cpd.srcvars])
end


function cpdmean!(cpd::CPD, x::Vector)
    x[cpd.tgtvars] = cpdmean(cpd.hom, cpd.srcobs, cpd.tgtobs, x[cpd.srcvars])
end
