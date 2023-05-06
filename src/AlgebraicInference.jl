module AlgebraicInference

# Gaussian Systems
export AbstractSystem, ClassicalSystem, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export LabeledBox, Valuation
export ⊗, ↓, ↑, d, fusion_algorithm, join_tree_construction

using Base.Iterators
using Catlab, Catlab.CategoricalAlgebra, Catlab.Graphs, Catlab.WiringDiagrams
using LinearAlgebra
using OrderedCollections

import Base: ==, \, *, -, convert, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./utilities.jl")
include("./valuations.jl")

end
