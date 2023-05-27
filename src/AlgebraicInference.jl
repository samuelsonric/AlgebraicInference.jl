module AlgebraicInference

# Architectures
export Architecture
export answer_query, answer_query!, construct_factors!, construct_join_tree

# Hypergraphs
export osla_ffi, osla_sc, primal_graph

# Systems
export AbstractSystem, ClassicalSystem, OpenProgram, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export IdentityValuation, LabeledBox, Valuation
export combine, construct_inference_problem, domain, eliminate, neutral_valuation, project

using AbstractTrees
using Base.Iterators: take, drop
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
include("./architectures.jl")
include("./hypergraphs.jl")
include("./utils.jl")

end
