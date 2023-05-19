module AlgebraicInference

# Architectures
export Architecture
export answer_query, answer_query!, construct_architecture, construct_factors! 

# Hypergraphs
export osla_ffi, primal_graph

# Systems
export AbstractSystem, ClassicalSystem, Kernel, System
export ⊗, cov, dof, fiber, mean, oapply 

# Valuations
export IdentityValuation, LabeledBox, LabeledBoxVariable, Valuation, Variable
export combine, construct_inference_problem, domain, eliminate, fusion_algorithm,
       neutral_valuation, project

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
include("./architectures.jl")
include("./hypergraphs.jl")
include("./utils.jl")

end
