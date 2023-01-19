module AlgebraicInference

export GaussianDistribution, GaussRelDom, GaussianRelation
export conjugate, cov, mean, params, pushout
export ∘, ⋅, □, ◊, Δ, ∇, ⊕, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, mcopy, mmerge, mzero, oplus, plus, zero

using Catlab, Catlab.Theories
using LinearAlgebra

import Base: *, +, length
import Catlab.Theories: Hom, Ob
import Catlab.Theories: ∘, ⋅, □, ◊, Δ, ∇, ⊕, dagger, dcounit, dom, dunit, codom, compose, coplus, cozero, create, delete, id, mcopy, mmerge, mzero, oplus, plus, zero
import StatsBase: params
import Statistics: cov, mean

"""
    GaussianDistribution(Σ, μ)

A multivariate Gaussian distribution with covariance matrix ``\\Sigma`` and mean ``\\mu``.
"""
struct GaussianDistribution{T₁ <: AbstractMatrix, T₂ <: AbstractVector}
    Σ::T₁
    μ::T₂
end

"""
    GaussianDistribution(Σ::AbstractMatrix)

Construct a centered Gaussian distribution with covariance matrix ``\\Sigma``.
"""
function GaussianDistribution(Σ::AbstractMatrix)
    n = size(Σ, 1)
    μ = zeros(n)
    GaussianDistribution(Σ, μ)
end

"""
    GaussianDistribution(μ::AbstractVector)

Construct a Dirac distribution with mean ``\\mu``.
"""
function GaussianDistribution(μ::AbstractVector)
    n = length(μ)
    Σ = zeros(n, n)
    GaussianDistribution(Σ, μ)
end

length(ψ::GaussianDistribution) = length(ψ.μ)

"""
    cov(ψ::GaussianDistribution)

Return the covariance matrix corresponding to `ψ`.
"""
cov(ψ::GaussianDistribution) = ψ.Σ

"""
    mean(ψ::GaussianDistributon)

Return the mean vector corresponding to `ψ`.
"""
mean(ψ::GaussianDistribution) = ψ.μ

function *(M::Union{UniformScaling, AbstractMatrix}, ψ::GaussianDistribution)
    Σ = M * ψ.Σ * M'
    μ = M * ψ.μ
    GaussianDistribution(Σ, μ)
end

function +(ψ₁::GaussianDistribution, ψ₂::GaussianDistribution)
    Σ = ψ₁.Σ + ψ₂.Σ
    μ = ψ₁.μ + ψ₂.μ
    GaussianDistribution(Σ, μ)
end

function oplus(ψ₁::GaussianDistribution, ψ₂::GaussianDistribution)
    Σ = ψ₁.Σ ⊕ ψ₂.Σ
    μ = [ψ₁.μ; ψ₂.μ]
    GaussianDistribution(Σ, μ)
end

"""
    GaussRelDom(n)

The domain or codomain of a Gaussian relation.
"""
struct GaussRelDom{T <: Integer}
    n::T
end

"""
    GaussianRelation(L, R, ψ)

A Gaussian relation, i.e. a decorated cospan with legs `L`, `R` and decoration `ψ`.
"""
struct GaussianRelation{T₁ <: AbstractMatrix, T₂ <: AbstractMatrix, T₃, T₄}
    L::T₁
    R::T₂
    ψ::GaussianDistribution{T₃, T₄}
end

"""
    GaussianRelation(L::AbstractMatrix)

Construct a Gaussian relation ``d: m \\to n`` from a linear transformation ``L: \\mathbb{R}^m \\to \\mathbb{R}^n``.
"""
function GaussianRelation(L::AbstractMatrix)
    n = size(L, 1)
    R = Matrix(I, n, n)
    ψ = GaussianDistribution(zeros(n, n), zeros(n))
    GaussianRelation(L, R, ψ)
end

"""
    GaussianRelation(ψ::GaussianDistribution)

Construct a Gaussian relation ``d: 0 \\to n`` from a Gaussian distribution ``\\psi`` on ``\\mathbb{R}^n``.
"""
function GaussianRelation(ψ::GaussianDistribution)
    n = length(ψ)
    L = zeros(n, 0)
    R = Matrix(I, n, n)
    GaussianRelation(L, R, ψ)
end

function pushout(L::AbstractMatrix, R::AbstractMatrix)
    n = size(L, 1)
    K = nullspace([L' -R'])
    L₁ = K[1:n, :]'
    R₁ = K[n+1:end, :]'
    (L₁, R₁)
end

"""
    conjugate(Q::AbstractMatrix, a::AbstractVector, α::Real, B::AbstractMatrix, b::AbstractVector)

Compute the convex conjugate of a partial quadratic function
```math
q(x) = \\langle Qx, x \\rangle + \\langle a, x \\rangle + \\alpha + \\begin{cases}
    0 & Bx = b \\\
    \\infty & \\text{else}
\\end{cases},
```
where ``Q`` is positive semi-definite and ``b \\in \\mathrm{col} B``.

Returns a five-tuple ``(Q_1, a_1, \\alpha_1, B_1, b_1)``.
"""
function conjugate(Q::AbstractMatrix, a::AbstractVector, α::Real, B::AbstractMatrix, b::AbstractVector)
    P = nullspace(B)'
    Q₁ = P' * pinv(P * Q * P') * P
    a₁ = pinv(B) * b - Q₁ * a
    α₁ = -1/2 * dot(a₁, a) - α
    B₁ = nullspace(Q * P')' * P
    b₁ = B₁ * a
    (Q₁, a₁, α₁, B₁, b₁)
end

"""
    params(d::GaussianRelation)

Represent a Gaussian relation ``d: m \\to n`` as an [extended Gaussian distribution](https://arxiv.org/abs/2204.14024) ``\\mathcal{N}(\\mu, \\Sigma) + \\text{null } D`` on ``\\mathbb{R}^{m + n}``.

Returns a five-tuple ``(Σ, μ, D, B, b)``. If ``b \\notin \\text{col } B``, then ``d`` is equal to the failure morphism ``\\bot: m \\to n``.
"""
function params(d::GaussianRelation)
    n = length(d.ψ)
    M = [d.L d.R]
    Q, a, α, B, b = conjugate(d.ψ.Σ, d.ψ.μ, 0, zeros(n, n), zeros(n))
    Σ, μ, _, D, _ = conjugate(M' * Q * M, M' * a, α, B * M, b)
    (Σ, μ, D, B * M, b)
end

length(d::GaussianRelation) = size(d.L, 2) + size(d.R, 2)

"""
    cov(d::GaussianRelation)

Return a representative of the covariance matrix corresponding to `d`.
"""
cov(d::GaussianRelation) = params(d)[1]

"""
    mean(d::GaussianRelation)

Return a representative of the mean vector corresponding to `d`.
"""
mean(d::GaussianRelation) = params(d)[2]

@instance ThAbelianBicategoryRelations{GaussRelDom, GaussianRelation} begin
    mzero(::Type{GaussRelDom}) = GaussRelDom(0)
    oplus(X₁::GaussRelDom, X₂::GaussRelDom) = GaussRelDom(X₁.n + X₂.n)
    dom(d::GaussianRelation) = GaussRelDom(size(d.L, 2))
    codom(d::GaussianRelation) = GaussRelDom(size(d.R, 2))
    dagger(d::GaussianRelation) = GaussianRelation(d.R, d.L, -I * d.ψ)

    function compose(d₁::GaussianRelation, d₂::GaussianRelation)
        Lₚ, Rₚ = pushout(d₁.R, d₂.L)
        L = Lₚ * d₁.L
        R = Rₚ * d₂.R
        ψ = Lₚ * d₁.ψ + Rₚ * d₂.ψ
        GaussianRelation(L, R, ψ)
    end

    function oplus(d₁::GaussianRelation, d₂::GaussianRelation)
        L = oplus(d₁.L, d₂.L)
        R = oplus(d₁.R, d₂.R)
        ψ = oplus(d₁.ψ, d₂.ψ)
        GaussianRelation(L, R, ψ)
    end

    meet(d₁::GaussianRelation, d₂::GaussianRelation) = Δ(X) ⋅ (d₁ ⊕ d₂) ⋅ ∇(X)
    join(d₁::GaussianRelation, d₂::GaussianRelation) = coplus(X) ⋅ (d₁ ⊕ d₂) ⋅ plus(X)

    function id(X::GaussRelDom)
        L = Matrix(I, X.n, X.n)
        GaussianRelation(L)
    end

    function zero(X::GaussRelDom)
        L = zeros(X.n, 0)
        GaussianRelation(L)
    end

    function delete(X::GaussRelDom)
        L = zeros(0, X.n)
        GaussianRelation(L)
    end

    function mcopy(X::GaussRelDom)
        L = [Matrix(I, X.n, X.n); Matrix(I, X.n, X.n)]
        GaussianRelation(L)
    end

    function plus(X::GaussRelDom)
        L = [Matrix(I, X.n, X.n) Matrix(I, X.n, X.n)]
        GaussianRelation(L)
    end

    dunit(X::GaussRelDom) = □(X) ⋅ Δ(X)
    cozero(X::GaussRelDom) = dagger(zero(X))
    create(X::GaussRelDom) = dagger(delete(X))
    mmerge(X::GaussRelDom) = dagger(mcopy(X))
    coplus(X::GaussRelDom) = dagger(plus(X))
    dcounit(X::GaussRelDom) = dagger(dunit(X)) 

    function swap(X₁::GaussRelDom, X₂::GaussRelDom)
        L = [zeros(X₂.n, X₁.n) Matrix(I, X₂.n, X₂.n); Matrix(I, X₁.n, X₁.n) zeros(X₁.n, X₂.n)]
        GaussianRelation(L)
    end

    top(X₁::GaussRelDom, X₂::GaussRelDom) = ◊(X₁) ⋅ □(X₂)
    bottom(X₁::GaussRelDom, X₂::GaussRelDom) = cozero(X₁) ⋅ zero(X₂)
end

end
