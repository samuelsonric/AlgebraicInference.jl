"""
    GaussianDistribution(cgf)

A multivariate Gaussian distribution.
"""
struct GaussianDistribution{T₁, T₂}
    cgf::QuadraticFunction{T₁, T₂}
end

"""
    GaussianDistribution(Σ::AbstractMatrix, μ::AbstractVector)

Construct a Gaussian distribution with covariance `Σ` and mean `μ`.
"""
GaussianDistribution(Σ::AbstractMatrix, μ::AbstractVector) = GaussianDistribution(QuadraticFunction(Σ, μ))

"""
    GaussianDistribution(Σ::AbstractMatrix)

Construct a centered Gaussian distribution with covariance `Σ`.
"""
GaussianDistribution(Σ::AbstractMatrix) = GaussianDistribution(QuadraticFunction(Σ))

"""
    GaussianDistribution(μ::AbstractVector)

Construct a Dirac distribution with mean `μ`.
"""
GaussianDistribution(μ::AbstractVector) = GaussianDistribution(QuadraticFunction(μ))

"""
    cov(ψ::GaussianDistribution)

Get the covariance of `ψ`.
"""
cov(ψ::GaussianDistribution) = ψ.cgf.Q

"""
    mean(ψ::GaussianDistribution)

Get the mean of `ψ`.
"""
mean(ψ::GaussianDistribution) = ψ.cgf.a

"""
    GaussRelDom(n)

The Euclidean space ``\\mathbb{R}^n``.
"""
struct GaussRelDom{T}
    n::QuadDom{T}
end

GaussRelDom(n::Integer) = GaussRelDom(QuadDom(n))

"""
    GaussianRelation(rf)

A Gaussian relation is a morphism in the hypergraph category ``\\text{Cond}(\\text{GaussEx})``. Internally, it is represented as an equivalence class of partial quadratic bifunctions, where 
```math
F_1 \\sim F_2 \\iff F_1 = F_2 + \\alpha \\text{ for some } \\alpha \\in \\mathbb{R}.
```
```
struct GaussianRelation{T₁, T₂, T₃, T₄}
    rf::QuadraticBifunction{T₁, T₂, T₃, T₄}
end
```
Every Gaussian relation ``d: m \\to n`` is either an extended Gaussian distribution on ``\\mathbb{R}^{m + n}`` or a failure state ``\\bot_{m + n}``.

References:
- Stein & Staton (2021), "Compositional Semantics for Probabilistic Programs with Exact Conditioning" ([arXiv:2101.11351](https://arxiv.org/abs/2101.11351))
- Stein (2022), "Decorated Linear Relations: Extending Gaussian Probability with Uninformative Priors" ([arXiv:2204.14024](https://arxiv.org/abs/2204.14024))
- Stein (2022), "A Hypergraph Category for Exact Gaussian Inference", ([https://msp.cis.strath.ac.uk/act2022/papers/ACT2022_paper_3601.pdf](https://msp.cis.strath.ac.uk/act2022/papers/ACT2022_paper_3601.pdf))
"""
struct GaussianRelation{T₁, T₂, T₃, T₄}
    rf::QuadraticBifunction{T₁, T₂, T₃, T₄}
end

GaussianRelation(L::AbstractMatrix, R::AbstractMatrix, ψ::GaussianDistribution) = GaussianRelation(QuadraticBifunction(L, R, ψ.cgf))

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
GaussianRelation(ψ::GaussianDistribution) = GaussianRelation(QuadraticBifunction(ψ.cgf))

"""
    params(d::GaussianRelation)

Represent `d` as an extended Gaussian distribution.

Returns a quadruple ``(Q, a, B, b)``. If ``b \\neq 0``, then ``d = \\bot``. Otherwise,
```math
d = \\mathcal{N}(a, Q) + \\text{null } B.
```
"""
function params(d::GaussianRelation)
    Q, a, _, B, b = conjugate(d.rf)
    Q, a, B, b
end

"""
    cov(d::GaussianRelation)

Get the covariance of `d`.
"""
cov(d::GaussianRelation) = params(d)[1]

"""
    mean(d::GaussianRelation)

Get the mean of `d`.
"""
mean(d::GaussianRelation) = params(d)[2]

@instance ThAbelianBicategoryRelations{GaussRelDom, GaussianRelation} begin
    mzero(::Type{GaussRelDom}) = GaussRelDom(mzero(QuadDom))
    dom(d::GaussianRelation) = GaussRelDom(dom(d.rf))
    codom(d::GaussianRelation) = GaussRelDom(codom(d.rf))
    oplus(X::GaussRelDom, Y::GaussRelDom) = GaussRelDom(X.n ⊕ Y.n)
    dagger(d::GaussianRelation) = GaussianRelation(dagger(d.rf))
    id(X::GaussRelDom) = GaussianRelation(id(X.n))
    zero(X::GaussRelDom) = GaussianRelation(zero(X.n))
    delete(X::GaussRelDom) = GaussianRelation(delete(X.n))
    mcopy(X::GaussRelDom) = GaussianRelation(mcopy(X.n))
    plus(X::GaussRelDom) = GaussianRelation(plus(X.n))
    dunit(X::GaussRelDom) = GaussianRelation(dunit(X.n))
    cozero(X::GaussRelDom) = GaussianRelation(cozero(X.n))
    create(X::GaussRelDom) = GaussianRelation(create(X.n))
    mmerge(X::GaussRelDom) = GaussianRelation(mmerge(X.n))
    coplus(X::GaussRelDom) = GaussianRelation(coplus(X.n))
    dcounit(X::GaussRelDom) = GaussianRelation(dcounit(X.n))
    swap(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(swap(X.n, Y.n))
    top(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(top(X.n, Y.n))
    bottom(X::GaussRelDom, Y::GaussRelDom) = GaussianRelation(bottom(X.n, Y.n))
    oplus(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(oplus(d₁.rf, d₂.rf))
    compose(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(compose(d₁.rf, d₂.rf))
    meet(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(meet(d₁.rf, d₂.rf))
    join(d₁::GaussianRelation, d₂::GaussianRelation) = GaussianRelation(join(d₁.rf, d₂.rf))
end
