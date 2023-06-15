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

wd = @relation (x,) begin
    prior(x)
    likelihood(x, y)
    evidence(y)
end

hm = Dict(
    :prior => normal([1;;], [0]),             # x ~ N(0, 1)
    :likelihood => kernel([1;;], [0], [1;;]), # y | x ~ N(x, 1)
    :evidence => normal([0;;], [2]))          # y = 2

# Solve directly
Σ = oapply(wd, hm) 

# Solve using belief propagation.
T = DenseGaussianSystem{Float64}
Σ = solve(InferenceProblem{T}(wd, hm), MinFill())
```

![inference](./inference.svg)
