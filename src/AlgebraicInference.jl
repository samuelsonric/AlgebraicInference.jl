module AlgebraicInference

# Architectures
export Architecture
export answer_query, answer_query!, architecture

# Graphs
export minfill!, minwidth!, primal_graph

# Systems
export GaussianSystem
export ⊗, canon, cov, invcov, marginal, normal, kernel, mean, oapply, pushfwd

# Valuations
export IdentityValuation, LabeledBox, Valuation
export combine, inference_problem, domain, project

using AbstractTrees
using Base.Iterators: take, drop
using Catlab, Catlab.CategoricalAlgebra, Catlab.Programs, Catlab.WiringDiagrams
using Graphs
using LinearAlgebra
using LinearSolve
using MetaGraphsNext

using Graphs: add_edge!, add_vertex!, has_edge, neighbors, nv, vertices
using LinearAlgebra: checksquare

import AbstractTrees: ChildIndexing, NodeType, ParentLinks, children, nodetype, nodevalue,
                      parent
import Base: *, +, length, zero
import Catlab.Theories: ⊗
import Catlab.WiringDiagrams: oapply
import CommonSolve: solve!
import StatsBase: dof
import Statistics: cov, mean

include("./systems.jl")
include("./valuations.jl")
include("./graphs.jl")
include("./architectures.jl")
include("./utils.jl")

end
