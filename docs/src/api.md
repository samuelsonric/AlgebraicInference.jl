# Library Reference

## Gaussian Relations

```@docs
GaussianDistribution
GaussianDistribution(Σ::AbstractMatrix, μ::AbstractVector)
GaussianDistribution(Σ::AbstractMatrix)
GaussianDistribution(μ::AbstractVector)

GaussianRelation
GaussianRelation(L::AbstractMatrix)
GaussianRelation(ψ::GaussianDistribution)

GaussRelDom

params
cov
mean
```

## Quadratic Functions

```@docs
QuadraticFunction
QuadraticFunction(Q::AbstractMatrix)
QuadraticFunction(a::AbstractVector)

QuadraticBifunction
QuadraticBifunction(L::AbstractMatrix)
QuadraticBifunction(f::QuadraticFunction)

QuadDom

*
conjugate

dom
codom
mzero
dagger
compose
oplus
meet
join
id
zero
delete
mcopy
plus
dunit
cozero
create
mmerge
coplus
dcounit
swap
top
bottom
```
