module AlgebraicInference

# Graphs
export construct_elimination_sequence, construct_join_tree

# Systems
export AbstractSystem, ClassicalSystem, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export IdentityValuation, LabeledBox, LabeledBoxVariable, Valuation, Variable
export combine, construct_inference_problem, construct_factors, collect_algorithm,       
       domain, eliminate, fusion_algorithm, neutral_valuation, project,
       shenoy_shafer_architecture!

using AbstractTrees
using Catlab, Catlab.CategoricalAlgebra, Catlab.WiringDiagrams
using JunctionTrees: Node
using LinearAlgebra

import Base: ==, \, *, convert, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./valuations.jl")
include("./graphs.jl")
include("./utils.jl")

end
