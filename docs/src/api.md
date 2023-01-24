# Library Reference

## Gaussian Relations

```@docs
GaussianDistribution
GaussianDistribution(Q::AbstractMatrix)
GaussianDistribution(a::AbstractVector)

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
adjoint

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
