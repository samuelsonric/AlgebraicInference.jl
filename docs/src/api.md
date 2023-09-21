e# Library Reference
## Systems

```@docs
GaussianSystem
CanonicalForm
DenseGaussianSystem
DenseCanonicalForm

GaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Real)
CanonicalForm(::AbstractMatrix, ::AbstractVector)

normal
kernel

length(::GaussianSystem)
cov(::GaussianSystem)
invcov(::GaussianSystem)
var(::GaussianSystem)
mean(::GaussianSystem)
```

## Problems
```@docs
InferenceProblem

InferenceProblem(::RelationDiagram, ::AbstractDict, ::AbstractDict)
InferenceProblem(::BayesNet, ::AbstractVector, ::AbstractDict)

solve(::InferenceProblem, ::EliminationAlgorithm, ::SupernodeType, ::ArchitectureType)
init(::InferenceProblem, ::EliminationAlgorithm, ::SupernodeType, ::ArchitectureType)
```

## Solvers
```@docs
InferenceSolver

solve!(::InferenceSolver)
mean(::InferenceSolver)
rand(::AbstractRNG, ::InferenceSolver)
```

## Elimination
```@docs
EliminationAlgorithm
MinDegree
MinFill
CuthillMcKeeJL_RCM
AMDJL_AMD
MetisJL_ND

SupernodeType
Node
MaximalSupernode
```

## Architectures
```@docs
ArchitectureType
ShenoyShafer
LauritzenSpiegelhalter
```
