"""
    GaussianDistribution(Q, a)

A Gaussian distribution with covariance `Q` and mean `a`.
"""
const GaussianDistribution = QuadraticFunction

"""
    GaussianDistribution(Q::AbstractMatrix)

Construct a centered Gaussian distribution with covariance `Q`.
"""
GaussianDistribution(Q::AbstractMatrix)

"""
    GaussianDistribution(a::AbstractVector)

Construct a Dirac distribution with mean `a`.
"""
GaussianDistribution(a::AbstractVector)

"""
    cov(ψ::GaussianDistribution)

Return the covariance of `ψ`.
"""
cov(ψ::GaussianDistribution) = ψ.Q

"""
    mean(ψ::GaussianDistribution)

Return the mean of `ψ`.
"""
mean(ψ::GaussianDistribution) = ψ.a

"""
    GaussRelDom(n)

The Euclidean space ``\\mathbb{R}^m``.
"""
struct GaussRelDom{T <: Integer}
    n::T
end

"""
    GaussianRelation(rf)

A Gaussian relation with rate function `rf`.

A Gaussian relation ``d: m \\to n`` with rate function ``\\mathcal{I} \\neq \\infty`` is an [extended Gaussian distribution](https://arxiv.org/abs/2204.14024) on ``\\mathbb{R}^{m + n}``.
"""
struct GaussianRelation{T₁, T₂, T₃, T₄}
    rf::QuadraticBifunction{T₁, T₂, T₃, T₄}
end

"""
    GaussianRelation(L::AbstractMatrix)

Construct the extended Gaussian distribution
```math
\\mathcal{N}(0, 0) + \\{ (x, y) \\mid Lx = y \\}.
```
"""
GaussianRelation(L::AbstractMatrix) = GaussianRelation(QuadraticBifunction(L))

"""
    GaussianRelation(ψ::GaussianDistribution)

Construct the extended Gaussian distribution ``\\psi + \\{ 0 \\}``.
"""
GaussianRelation(ψ::GaussianDistribution) = GaussianRelation(QuadraticBifunction(ψ))

"""
    params(d::GaussianRelation)

Represent `d` as an extended Gaussian distribution.

Returns a quadruple ``(Q, a, B, b)``. If ``b = 0``, then
```math
d = \\mathcal{N}(a, Q) + \\text{null } B.
```
Otherwise, `d` is not an extended Gaussian distribution.
"""
function params(d::GaussianRelation)
    Q, a, _, B, b = conjugate(d.rf)
    Q, a, B, b
end

"""
    cov(d::GaussianRelation)

Return the covariance of `d`.
"""
cov(d::GaussianRelation) = params(d)[1]

"""
    mean(d::GaussianRelation)

Return the mean of `d`.
"""
mean(d::GaussianRelation) = params(d)[2]

@instance ThAbelianBicategoryRelations{GaussRelDom, GaussianRelation} begin
    mzero(::Type{GaussRelDom}) = GaussRelDom(0)
    dom(d::GaussianRelation) = GaussRelDom(dom(d.rf).n)
    codom(d::GaussianRelation) = GaussRelDom(codom(d.rf).n)
    oplus(X::GaussRelDom, Y::GaussRelDom) = GaussRelDom(X.n + Y.n)
    dagger(d::GaussianRelation) = GaussianRelation(dagger(d.rf))
    id(X::GaussRelDom) = GaussianRelation(id(QuadDom(X.n)))
    zero(X::GaussRelDom) = GaussianRelation(zero(QuadDom(X.n)))
    delete(X::GaussRelDom) = GaussianRelation(delete(QuadDom(X.n)))
    mcopy(X::GaussRelDom) = GaussianRelation(mcopy(QuadDom(X.n)))
    plus(X::GaussRelDom) = GaussianRelation(plus(QuadDom(X.n)))
    dunit(X::GaussRelDom) = GaussianRelation(dunit(QuadDom(X.n)))
    cozero(X::GaussRelDom) = GaussianRelation(cozero(QuadDom(X.n)))
    create(X::GaussRelDom) = GaussianRelation(create(QuadDom(X.n)))
    mmerge(X::GaussRelDom) = GaussianRelation(mmerge(QuadDom(X.n)))
    coplus(X::GaussRelDom) = GaussianRelation(coplus(QuadDom(X.n)))
    dcounit(X::GaussRelDom) = GaussianRelation(dcounit(QuadDom(X.n)))
    swap(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(swap(QuadDom(X.n), QuadDom(Y.n)))
    top(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(top(QuadDom(X.n), QuadDom(Y.n)))
    bottom(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(bottom(QuadDom(X.n), QuadDom(Y.n)))
    oplus(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(oplus(d₁.rf, d₂.rf))
    compose(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(compose(d₁.rf, d₂.rf))
    meet(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(meet(d₁.rf, d₂.rf))
    join(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(join(d₁.rf, d₂.rf))
end
