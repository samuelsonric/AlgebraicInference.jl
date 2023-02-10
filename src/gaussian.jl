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
    GaussDom(n)

The Euclidean space ``\\mathbb{R}^n``.
"""
struct GaussDom{T}
    n::QuadDom{T}
end

GaussDom(n::Integer) = GaussDom(QuadDom(n))

"""
    OpenGaussianDistribution(rf)

An open Gaussian distribution is a decorated cospan
```math
\\left(
\\begin{aligned}
\\begin{CD}
m   @>L>> k @<R<< n
\\end{CD}
\\end{aligned},
\\enspace
\\begin{aligned}
\\mathcal{N}(\\mu, \\Sigma)
\\end{aligned}
\\right),
```
where ``L``, ``R`` are matrices and ``\\mathcal{N}(\\mu, \\Sigma)`` is a Gaussian distribution on ``\\mathbb{R}^k``. This data determines a morphism
```math
d: m \\to n
```
in the hypergraph category ``\\text{Cond}(\\text{GaussEx})``. In particular, if ``m = 0``, then ``d`` represents the posterior distribution
```math
X \\mid Y = 0
```
of a Bayesian model
```math
\\begin{align*}
X           &\\sim \\mathcal{U} \\\\
\\epsilon   &\\sim \\mathcal{N}(\\mu, \\Sigma) \\\\
Y           &= RX - \\epsilon,
\\end{align*}
```
where ``\\mathcal{U}`` is an improper uniform prior over ``\\mathbb{R}^n`` and ``X \\perp \\epsilon``.

References:
- Stein & Staton (2021), "Compositional Semantics for Probabilistic Programs with Exact Conditioning" ([arXiv:2101.11351](https://arxiv.org/abs/2101.11351))
- Stein (2022), "Decorated Linear Relations: Extending Gaussian Probability with Uninformative Priors" ([arXiv:2204.14024](https://arxiv.org/abs/2204.14024))
- Stein (2022), "A Hypergraph Category for Exact Gaussian Inference", ([https://msp.cis.strath.ac.uk/act2022/papers/ACT2022_paper_3601.pdf](https://msp.cis.strath.ac.uk/act2022/papers/ACT2022_paper_3601.pdf))
"""
struct OpenGaussianDistribution{T₁, T₂, T₃, T₄}
    rf::OpenQuadraticFunction{T₁, T₂, T₃, T₄}
end

"""
    OpenGaussianDistribution(L::AbstractMatrix, R::AbstractMatrix, ψ::GaussianDistribution)

Construct an open Gaussian distribution with legs (`L`, `R`) and decoration `ψ`.
"""
OpenGaussianDistribution(L::AbstractMatrix, R::AbstractMatrix, ψ::GaussianDistribution) = OpenGaussianDistribution(OpenQuadraticFunction(L, R, ψ.cgf))

"""
    OpenGaussianDistribution(L::AbstractMatrix, ψ::GaussianDistribution)

Construct the Gaussian map
```math
x \\mapsto Lx + \\psi
```
"""
OpenGaussianDistribution(L::AbstractMatrix, ψ::GaussianDistribution) = OpenGaussianDistribution(OpenQuadraticFunction(L, ψ.cgf))

"""
    OpenGaussianDistribution(L::AbstractMatrix)

Construct the linear map
```math
x \\mapsto Lx
```
"""
OpenGaussianDistribution(L::AbstractMatrix) = OpenGaussianDistribution(OpenQuadraticFunction(L))

"""
    OpenGaussianDistribution(ψ::GaussianDistribution)

Construct the Gaussian map
```math
* \\mapsto \\psi
```
"""
OpenGaussianDistribution(ψ::GaussianDistribution) = OpenGaussianDistribution(OpenQuadraticFunction(ψ.cgf))

"""
    params(d::OpenGaussianDistribution)

Compute the cumulant-generating function ``K`` of `d`.

Returns a quadruple ``(Q, a, B, b)``. If ``b \\neq 0``, then ``K = -\\infty``. Otherwise,
```math
K(x, y) =
\\begin{cases}
\\langle (x, y), \\frac{1}{2}Q(x, y) + a \\rangle   & B(x, y) = 0 \\\\
\\infty                                             & \\text{else}
\\end{cases}.
```
"""
function params(d::OpenGaussianDistribution)
    Q, a, _, B, b = conjugate(d.rf)
    Q, a, B, b
end

"""
    cov(d::OpenGaussianDistribution)

Get the covariance of `d`.
"""
cov(d::OpenGaussianDistribution) = params(d)[1]

"""
    mean(d::OpenGaussianDistribution)

Get the mean of `d`.
"""
mean(d::OpenGaussianDistribution) = params(d)[2]

@instance ThAbelianBicategoryRelations{GaussDom, OpenGaussianDistribution} begin

    """
        mzero(::Type{GaussDom})

    Construct the Euclidean space ``\\mathbb{R}^0``.
    """
    mzero(::Type{GaussDom}) = GaussDom(mzero(QuadDom))
    
    """
        dom(d::OpenGaussianDistribution)
    """
    dom(d::OpenGaussianDistribution) = GaussDom(dom(d.rf))

    """
        codom(d::OpenGaussianDistribution)
    """
    codom(d::OpenGaussianDistribution) = GaussDom(codom(d.rf))

    """
        oplus(X::GaussDom, Y::GassDom)

    Compute direct sum of `X` and `Y`.
    """
    oplus(X::GaussDom, Y::GaussDom) = GaussDom(X.n ⊕ Y.n)

    """
        dagger(d::OpenGaussianDistribution)
    """
    dagger(d::OpenGaussianDistribution) = OpenGaussianDistribution(dagger(d.rf))

    """
        id(X::GaussDom)
    """
    id(X::GaussDom) = OpenGaussianDistribution(id(X.n))

    """
        zero(X::GaussDom)

    Construct the linear map
    ```math
    * \\mapsto 0.
    ```
    """
    zero(X::GaussDom) = OpenGaussianDistribution(zero(X.n))

    """
        delete(X::GaussDom)

    Construct the linear map
    ```math
    x \\mapsto *.
    ```
    """
    delete(X::GaussDom) = OpenGaussianDistribution(delete(X.n))

    """
        mcopy(X::GaussDom)
    
    Construct the linear map
    ```math
    x \\mapsto (x, x).
    ```
    """
    mcopy(X::GaussDom) = OpenGaussianDistribution(mcopy(X.n))

    """
        plus(X::GaussDom)

    Construct the linear map
    ```math
    (x, y) \\mapsto x + y.
    ```
    """
    plus(X::GaussDom) = OpenGaussianDistribution(plus(X.n))

    """
        dunit(X::GaussDom)

    Construct the extended Gaussian map
    ```math
    * \\mapsto \\{(y, y) \\mid y \\}.
    ```
    """
    dunit(X::GaussDom) = OpenGaussianDistribution(dunit(X.n))

    """
        cozero(X::GaussDom)
    """
    cozero(X::GaussDom) = OpenGaussianDistribution(cozero(X.n))

    """
        create(X::GaussDom)
    
    Construct the extended Gaussian map
    ```math
    * \\mapsto \\mathbb{R}^n.
    ```
    """
    create(X::GaussDom) = OpenGaussianDistribution(create(X.n))

    """
        mmerge(X::GaussDom)
    """
    mmerge(X::GaussDom) = OpenGaussianDistribution(mmerge(X.n))

    """
        coplus(X::GaussDom)
    """
    coplus(X::GaussDom) = OpenGaussianDistribution(coplus(X.n))

    """
        dcounit(X::GaussDom)
    """
    dcounit(X::GaussDom) = OpenGaussianDistribution(dcounit(X.n))
    
    """
        swap(X::GaussDom, Y::GaussDom)

    Construct the linear map
    ```math
    (x_1, x_2) \\mapsto (x_2, x_1).
    ```
    """
    swap(X::GaussDom, Y::GaussDom) = OpenGaussianDistribution(swap(X.n, Y.n))
    
    """
        top(X::GaussDom, Y::GaussDom)

    Construct the extended Gaussian map
    ```math
    x \\mapsto \\mathbb{R}^n.
    ```
    """
    top(X::GaussDom, Y::GaussDom) = OpenGaussianDistribution(top(X.n, Y.n))

    """
        bottom(X::GaussDom, Y::GaussDom)
    """
    bottom(X::GaussDom, Y::GaussDom) = OpenGaussianDistribution(bottom(X.n, Y.n))

    """
        oplus(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution)
    """
    oplus(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution) = OpenGaussianDistribution(oplus(d₁.rf, d₂.rf))

    """
        compose(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution)
    """
    compose(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution) = OpenGaussianDistribution(compose(d₁.rf, d₂.rf))

    """
        meet(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution)
    """
    meet(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution) = OpenGaussianDistribution(meet(d₁.rf, d₂.rf))

    """
        join(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution)
    """
    join(d₁::OpenGaussianDistribution, d₂::OpenGaussianDistribution) = OpenGaussianDistribution(join(d₁.rf, d₂.rf))
end

"""
    oapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: OpenGaussianDistribution}

Compose open Gaussian distributions according to an undirected wiring wiring diagram.
"""
function oapply(composite::UndirectedWiringDiagram, hom_map::AbstractDict{T₁, T₂}) where {T₁, T₂ <: OpenGaussianDistribution}
    hom_map = Dict(x => y.rf for (x, y) in hom_map)
    rf = oapply(composite, hom_map)
    OpenGaussianDistribution(rf)
end

"""
    oapply(composite::UndirectedWiringDiagram, cospans::AbstractVector{T}) where T <: OpenGaussianDistribution

Compose open Gaussian distributions according to an undirected wiring wiring diagram.
"""
function oapply(composite::UndirectedWiringDiagram, cospans::AbstractVector{T}) where T <: OpenGaussianDistribution
    cospans = [x.rf for x in cospans]
    rf = oapply(composite, cospans)
    OpenGaussianDistribution(rf)
end
