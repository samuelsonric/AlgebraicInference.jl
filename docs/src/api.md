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
mean(::InferenceSolver{AncestralSampler()})
rand(::InferenceSolver{AncestralSampler()})
rand(::AbstractRNG, ::InferenceSolver{AncestralSampler()})
```

## Elimination
```@docs
EliminationAlgorithm
MaxCardinality
MinDegree
MinFill
ChordalGraph
CuthillMcKeeJL_RCM
AMDJL_AMD
MetisJL_ND

ischordal
```

## Supernodes
```@docs
SupernodeType
Node
MaximalSupernode
FundamentalSupernode
```

## Architectures
```@docs
ArchitectureType
ShenoyShafer
LauritzenSpiegelhalter
HUGIN
Idempotent
AncestralSampler
```
