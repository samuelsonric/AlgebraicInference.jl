module AlgebraicInference

# Graphs
export LabeledGraph, child, construct_elimination_sequence, construct_join_tree

# Systems
export AbstractSystem, ClassicalSystem, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export LabeledBox, LabeledBoxVariable, Valuation, Variable
export combine, construct_inference_problem, construct_join_tree_factors, collect_algorithm,       
       domain, eliminate, fusion_algorithm, neutral_element, project

using Base.Iterators
using Catlab, Catlab.CategoricalAlgebra, Catlab.WiringDiagrams
using LinearAlgebra

import Base: ==, \, *, convert, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./graphs.jl")
include("./systems.jl")
include("./utils.jl")
include("./valuations.jl")

end
