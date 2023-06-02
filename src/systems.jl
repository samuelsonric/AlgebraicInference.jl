"""
    GaussianSystem{
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix, 
        T₃ <: AbstractVector,
        T₄ <: AbstractVector,
        T₅}

A Gaussian system.

References:
- J. C. Willems, "Open Stochastic Systems," in *IEEE Transactions on Automatic Control*, 
  vol. 58, no. 2, pp. 406-421, Feb. 2013, doi: 10.1109/TAC.2012.2210836.
"""
struct GaussianSystem{
    T₁ <: AbstractMatrix,
    T₂ <: AbstractMatrix,
    T₃ <: AbstractVector,
    T₄ <: AbstractVector,
    T₅}

    P::T₁
    S::T₂
    p::T₃
    s::T₄
    σ::T₅

    function GaussianSystem(P::T₁, S::T₂, p::T₃, s::T₄, σ::T₅) where {
        T₁ <: AbstractMatrix,
        T₂ <: AbstractMatrix,
        T₃ <: AbstractVector,
        T₄ <: AbstractVector,
        T₅}
    
        m = checksquare(P)
        n = checksquare(S)
        @assert m == n == length(p) == length(s)
        new{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ)
    end
end

"""
    canon(J::AbstractMatrix, h::AbstractVector)

Construct a multivariate nornal distribution with information matrix `J` and potential vector
`h`.
"""
function canon(J::AbstractMatrix, h::AbstractVector)
    n = size(J, 1)
    GaussianSystem(J, falses(n, n), h, falses(n), false)
end

"""
    canon(J::AbstractMatrix)

Construct a centered multivariate normal distribution with information matrix `J`.
"""
function canon(J::AbstractMatrix)
    n = size(J, 1)
    canon(J, falses(n))
end

"""
    normal(Σ::AbstractMatrix, μ::AbstractVector)

Construct a multivariate normal distribution with covariance matrix `Σ` and mean vector `μ`.
"""
function normal(Σ::AbstractMatrix, μ::AbstractVector)
    V = nullspace(Σ)
    P = pinv(Σ)
    S = V * V'
    GaussianSystem(P, S, P * μ, S * μ, μ' * S * μ)
end

"""
    normal(Σ::AbstractMatrix)

Construct a centered multivariate normal distribution with covariance matrix `Σ`.
"""
function normal(Σ::AbstractMatrix)
    n = size(Σ, 1)
    normal(Σ, falses(n))
end

"""
    normal(μ::AbstractVector)

Construct a Dirac distribution with mean vector `μ`.
"""
function normal(μ::AbstractVector)
    n = length(μ)
    normal(falses(n, n), μ)
end

"""
    kernel(Σ::AbstractMatrix, μ::AbstractVector, L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\mathcal{N}(Lx + \\mu, \\Sigma).
```
"""
function kernel(Σ::AbstractMatrix, μ::AbstractVector, L::AbstractMatrix)
    normal(Σ, μ) * [-L I]
end

"""
    kernel(Σ::AbstractMatrix, L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\mathcal{N}(Lx, \\Sigma).
```
"""
function kernel(Σ::AbstractMatrix, L::AbstractMatrix)
    n = size(Σ, 1)
    kernel(Σ, falses(n), L)
end

"""
    kernel(L::AbstractMatrix)

Construct a conditional distribution of the form
```math
    y \\mid x \\sim \\delta_{Lx}.
```
"""
function kernel(L::AbstractMatrix)
    n = size(L, 1)
    kernel(falses(n, n), L)
end

"""
    length(Σ::GaussianSystem)

Get the dimension of `Σ`.
"""
function length(Σ::GaussianSystem)
    size(Σ.P, 1)
end

"""
    cov(Σ::GaussianSystem)

Get the covariance matrix of `Σ`.
"""
function cov(Σ::GaussianSystem)
    n = length(Σ)
    A = saddle(Σ.P, [Σ.S; Σ.s'], I(n), zeros(n + 1, n))
    A + A' * (I - Σ.P * A)
end

"""
    mean(Σ::GaussianSystem)

Get the mean vector of `Σ`.
"""
function mean(Σ::GaussianSystem)
    n = length(Σ)
    A = saddle(Σ.P, [Σ.S; Σ.s'], I(n), zeros(n + 1, n))
    a = saddle(Σ.P, [Σ.S; Σ.s'], Σ.p, [Σ.s; Σ.σ])
    a + A' * (Σ.p - Σ.P * a)
end

"""
    invcov(Σ::GaussianSystem)

Get the information matrix of `Σ`.
"""
function invcov(Σ::GaussianSystem)
    Σ.P
end

function ⊗(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    GaussianSystem(
        Σ₁.P ⊕ Σ₂.P,
        Σ₁.S ⊕ Σ₂.S,
        [Σ₁.p; Σ₂.p],
        [Σ₁.s; Σ₂.s],
        Σ₁.σ + Σ₂.σ)
end

function *(Σ::GaussianSystem, M::AbstractMatrix)
    @assert size(M, 1) == length(Σ)
    GaussianSystem(
        M' * Σ.P * M,
        M' * Σ.S * M,
        M' * Σ.p,
        M' * Σ.s,
        Σ.σ)
end

function +(Σ₁::GaussianSystem, Σ₂::GaussianSystem)
    @assert length(Σ₁) == length(Σ₂)    
    GaussianSystem(
        Σ₁.P + Σ₂.P,
        Σ₁.S + Σ₂.S,
        Σ₁.p + Σ₂.p,
        Σ₁.s + Σ₂.s,
        Σ₁.σ + Σ₂.σ)
end

function zero(Σ::GaussianSystem)
    GaussianSystem(
        zero(Σ.P),
        zero(Σ.S),
        zero(Σ.p),
        zero(Σ.s),
        zero(Σ.σ))
end

"""
    oapply(wd::UndirectedWiringDiagram, box_map::AbstractDict{<:Any, <:GaussianSystem})

See [`oapply(wd::UndirectedWiringDiagram, box_map::AbstractVector{<:GaussianSystem})`](@ref).
"""
function oapply(wd::UndirectedWiringDiagram, box_map::AbstractDict{<:Any, <:GaussianSystem})
    boxes = [box_map[x] for x in subpart(wd, :name)]
    oapply(wd, boxes)
end

"""
    oapply(wd::UndirectedWiringDiagram, boxes::AbstractVector{<:GaussianSystem})

Compose Gaussian systems according to the undirected wiring diagram `wd`.
"""
function oapply(wd::UndirectedWiringDiagram, boxes::AbstractVector{<:GaussianSystem})
    @assert nboxes(wd) == length(boxes)
    L = Bool[
        junction(wd, i; outer=false) == j
        for i in ports(wd; outer=false),
            j in junctions(wd)]
    R = Bool[
        junction(wd, i; outer=true ) == j
        for i in ports(wd; outer=true ),
            j in junctions(wd)]
    Σ = reduce(⊗, boxes; init=GaussianSystem(Bool[;;], Bool[;;], Bool[], Bool[], false))
    pushfwd(R, Σ * L)
 end
