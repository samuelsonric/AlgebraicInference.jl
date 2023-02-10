# Library Reference

## Gaussian Distributions

```@docs
GaussianDistribution
GaussianDistribution(Σ::AbstractMatrix, μ::AbstractVector)
GaussianDistribution(Σ::AbstractMatrix)
GaussianDistribution(μ::AbstractVector)

OpenGaussianDistribution
OpenGaussianDistribution(L::AbstractMatrix, R::AbstractMatrix, ψ::GaussianDistribution)
OpenGaussianDistribution(L::AbstractMatrix, ψ::GaussianDistribution)
OpenGaussianDistribution(L::AbstractMatrix)
OpenGaussianDistribution(ψ::GaussianDistribution)

GaussDom

params
cov
mean
```

## Quadratic Functions

```@docs
QuadraticFunction
QuadraticFunction(Q::AbstractMatrix)
QuadraticFunction(a::AbstractVector)

OpenQuadraticFunction
OpenQuadraticFunction(L::AbstractMatrix, f::QuadraticFunction)
OpenQuadraticFunction(L::AbstractMatrix)
OpenQuadraticFunction(f::QuadraticFunction)

QuadDom

*
conjugate
```

## Composition

```@docs
compose
oplus
meet
join
oapply
```

## Construction

```@docs
mzero
id
swap
zero
cozero
delete
create
mcopy
mmerge
plus
coplus
dunit
dcounit
top
bottom
```

## Other

```@docs
dom
codom
dagger
```
