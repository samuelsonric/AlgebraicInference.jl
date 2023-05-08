module AlgebraicInference

# Gaussian Systems
export AbstractSystem, ClassicalSystem, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export LabeledBox, Valuation
export ⊗, ↓, ↑, d, construct_elimination_sequence, construct_inference_problem, fusion_algorithm

using Base.Iterators
using Catlab, Catlab.CategoricalAlgebra, Catlab.WiringDiagrams
using LinearAlgebra
using OrderedCollections

import Base: ==, \, *, -, convert, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./utils.jl")
include("./valuations.jl")

end
