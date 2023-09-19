# AlgebraicInference.jl

[![Build Status](https://github.com/samuelsonric/AlgebraicInference.jl/workflows/Tests/badge.svg)](https://github.com/samuelsonric/AlgebraicInference.jl/actions?query=workflow%3ATests)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://samuelsonric.github.io/AlgebraicInference.jl/dev/)
[![Code Coverage](https://codecov.io/gh/samuelsonric/AlgebraicInference.jl/branch/master/graph/badge.svg?token=FJJQQCTUCF)](https://codecov.io/gh/samuelsonric/AlgebraicInference.jl)

AlgebraicInference.jl is a library for performing Bayesian inference on wiring diagrams,
building on [Catlab.jl](https://algebraicjulia.github.io/Catlab.jl/dev/). See the
[documentation](https://samuelsonric.github.io/AlgebraicInference.jl/dev/) for example
notebooks and an API.

## How to Use

```julia
using AlgebraicInference
using Catlab.Programs

wd = @relation (X,) where (X::m, Y::n) begin
    prior(X)
    likelihood(X, Y)
    evidence(Y)
end

hom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(
    :prior => normal(0, 1),           # p(X) = N(0, 1)
    :likelihood => kernel([1], 0, 1), # p(Y | X = x) = N(x, 1)
    :evidence => normal(2, 0))        # Y = 2

ob_map = Dict(
    :m => 1, # X ∈ ℝ¹
    :n => 1) # Y ∈ ℝ¹

problem = InferenceProblem(wd, hom_map, ob_map)

Σ = solve(problem)
```

![inference](./inference.svg)
