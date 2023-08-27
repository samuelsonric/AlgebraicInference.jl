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

wd = @relation (x,) where (x::X, y::Y) begin
    prior(x)
    likelihood(x, y)
    evidence(y)
end

hom_map = Dict(
    :prior => normal(0, 1),           # x ~ N(0, 1)
    :likelihood => kernel([1], 0, 1), # y | x ~ N(x, 1)
    :evidence => normal(2, 0))        # y = 2

ob_map = Dict(
    :X => 1, # x ∈ ℝ¹
    :Y => 1) # y ∈ ℝ¹

ob_attr = :junction_type

# Solve directly.
Σ = oapply(wd, hom_map, ob_map; ob_attr)

# Solve using belief propagation.
T₁ = Int
T₂ = DenseGaussianSystem{Float64}
T₃ = Int
T₄ = Vector{Float64}

ip = InferenceProblem{T₁, T₂, T₃, T₄}(wd, hom_map, ob_map; ob_attr)
alg = MinFill()

Σ = solve(ip, alg)
```

![inference](./inference.svg)
