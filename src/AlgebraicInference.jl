module AlgebraicInference

# Hypergraphs
export JoinTree
export osla_sc, osla_ffi, primal_graph, construct_join_tree

# Systems
export AbstractSystem, ClassicalSystem, Kernel, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export IdentityValuation, LabeledBox, LabeledBoxVariable, Valuation, Variable
export combine, construct_inference_problem, construct_factors!, collect_algorithm,       
       domain, eliminate, fusion_algorithm, neutral_valuation, project,
       shenoy_shafer_architecture!

using AbstractTrees
using Catlab, Catlab.CategoricalAlgebra, Catlab.WiringDiagrams
using LinearAlgebra
using OrderedCollections

import AbstractTrees: ChildIndexing, NodeType, ParentLinks, children, nodetype, nodevalue,
                      parent
import Base: \, *, length
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./valuations.jl")
include("./join_trees.jl")
include("./hypergraphs.jl")
include("./utils.jl")
end
